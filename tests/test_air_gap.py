"""
tests/test_air_gap.py

Tests for the Panda Guard air-gap sanitization layer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from panda_guard.air_gap import (
    scan_for_sensitive_data,
    redact_sensitive_text,
    hash_vectors_for_upload,
    sanitize_embedding_store,
    AirGapReport,
)


# ── Sensitive Data Detection ─────────────────────────────────────────────────

class TestScanForSensitiveData:
    def test_detects_openai_key(self):
        texts = ["config: OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz"]
        matches = scan_for_sensitive_data(texts)
        assert len(matches) > 0
        pattern_names = [m.pattern_name for m in matches]
        assert "api_key_openai" in pattern_names

    def test_detects_airtable_key(self):
        texts = ["AIRTABLE_API_KEY=patAbcDefGhiJklMnoPqrStUvWxYz0123456789"]
        matches = scan_for_sensitive_data(texts)
        assert any(m.pattern_name == "api_key_airtable" for m in matches)

    def test_detects_email(self):
        texts = ["contact: user@example.com for support"]
        matches = scan_for_sensitive_data(texts)
        assert any(m.pattern_name == "email_address" for m in matches)

    def test_detects_github_token(self):
        texts = ["token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"]
        matches = scan_for_sensitive_data(texts)
        assert any(m.pattern_name == "github_token" for m in matches)

    def test_detects_aws_key(self):
        texts = ["aws_key = AKIAIOSFODNN7EXAMPLE"]
        matches = scan_for_sensitive_data(texts)
        assert any(m.pattern_name == "aws_key" for m in matches)

    def test_detects_private_key(self):
        texts = ["-----BEGIN PRIVATE KEY-----\nMIIEvgI..."]
        matches = scan_for_sensitive_data(texts)
        assert any(m.pattern_name == "private_key_header" for m in matches)

    def test_detects_connection_string(self):
        texts = ["DATABASE_URL=postgres://user:pass@host:5432/dbname"]
        matches = scan_for_sensitive_data(texts)
        assert any(m.pattern_name == "connection_string" for m in matches)

    def test_clean_text_no_matches(self):
        texts = [
            "def hello():\n    return 'world'\n",
            "# This is a normal comment\nimport os\n",
        ]
        matches = scan_for_sensitive_data(texts)
        assert len(matches) == 0

    def test_truncated_preview(self):
        """Matched text should be truncated for safety."""
        texts = ["key=sk-very_long_secret_key_that_should_be_truncated_for_safety"]
        matches = scan_for_sensitive_data(texts)
        for m in matches:
            # Preview should not contain the full secret
            assert len(m.matched_text) <= 16

    def test_multiple_matches_in_one_text(self):
        text = (
            "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\n"
            "EMAIL=admin@company.com\n"
        )
        matches = scan_for_sensitive_data([text])
        assert len(matches) >= 2

    def test_chunk_keys_in_matches(self):
        texts = ["sk-abcdefghijklmnopqrstuvwxyz"]
        keys = ["config.py:0"]
        matches = scan_for_sensitive_data(texts, keys)
        assert matches[0].chunk_key == "config.py:0"


# ── Text Redaction ────────────────────────────────────────────────────────────

class TestRedactSensitiveText:
    def test_redacts_api_key(self):
        text = "api_key = sk-abcdefghijklmnopqrstuvwxyz"
        redacted = redact_sensitive_text(text)
        assert "sk-abcdefg" not in redacted
        assert "[REDACTED:" in redacted

    def test_redacts_email(self):
        text = "contact user@example.com for help"
        redacted = redact_sensitive_text(text)
        assert "user@example.com" not in redacted

    def test_preserves_non_sensitive(self):
        text = "def hello():\n    return 42\n"
        redacted = redact_sensitive_text(text)
        assert redacted == text

    def test_redacts_multiple(self):
        text = (
            "key=sk-abcdefghijklmnopqrstuvwxyz "
            "email=user@test.com"
        )
        redacted = redact_sensitive_text(text)
        assert "sk-" not in redacted
        assert "@test.com" not in redacted


# ── Vector Hashing ────────────────────────────────────────────────────────────

class TestHashVectorsForUpload:
    def test_returns_hex_hashes(self):
        vectors = np.random.randn(5, 10).astype(np.float32)
        hashes = hash_vectors_for_upload(vectors)
        assert len(hashes) == 5
        assert all(len(h) == 64 for h in hashes)  # SHA-256 hex

    def test_no_float_data_in_hashes(self):
        vectors = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        hashes = hash_vectors_for_upload(vectors)
        # Hash should only contain hex characters
        assert all(c in "0123456789abcdef" for c in hashes[0])

    def test_deterministic(self):
        vectors = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        h1 = hash_vectors_for_upload(vectors)
        h2 = hash_vectors_for_upload(vectors)
        assert h1 == h2

    def test_different_vectors_different_hashes(self):
        v1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        v2 = np.array([[1.0, 2.0, 3.1]], dtype=np.float32)
        assert hash_vectors_for_upload(v1) != hash_vectors_for_upload(v2)


# ── Full Store Sanitization ──────────────────────────────────────────────────

class TestSanitizeEmbeddingStore:
    def test_clean_store(self, tmp_path: Path):
        """A clean store should produce is_safe=True."""
        store_path = tmp_path / "clean_store.npz"
        vectors = np.random.randn(10, 8).astype(np.float32)
        texts = np.array([f"def func_{i}(): pass" for i in range(10)])
        keys = np.array([f"file_{i}.py:0" for i in range(10)])
        np.savez(str(store_path), vectors=vectors, texts=texts, keys=keys)

        report = sanitize_embedding_store(store_path)
        assert report.is_safe
        assert report.total_chunks == 10
        assert report.merkle_root != ""
        assert len(report.sensitive_matches) == 0

    def test_dirty_store_detected(self, tmp_path: Path):
        """A store with secrets should produce is_safe=False."""
        store_path = tmp_path / "dirty_store.npz"
        vectors = np.random.randn(3, 8).astype(np.float32)
        texts = np.array([
            "normal code here",
            "api_key = sk-abcdefghijklmnopqrstuvwxyz",
            "email: admin@secret.com",
        ])
        keys = np.array(["a.py:0", "config.py:0", "readme.md:0"])
        np.savez(str(store_path), vectors=vectors, texts=texts, keys=keys)

        report = sanitize_embedding_store(store_path, redact=False)
        assert not report.is_safe
        assert len(report.sensitive_matches) > 0

    def test_redaction_creates_clean_copy(self, tmp_path: Path):
        """Redaction should remove sensitive data from the saved store."""
        store_path = tmp_path / "dirty_store.npz"
        clean_path = tmp_path / "clean_store.npz"
        vectors = np.random.randn(2, 4).astype(np.float32)
        texts = np.array([
            "api_key = sk-verylongsecretkeyvalue1234567890",
            "normal code",
        ])
        keys = np.array(["config.py:0", "main.py:0"])
        np.savez(str(store_path), vectors=vectors, texts=texts, keys=keys)

        report = sanitize_embedding_store(store_path, redact=True, output_path=clean_path)
        assert report.redacted_chunks > 0

        # Verify the clean copy doesn't have the secret
        clean_data = np.load(str(clean_path), allow_pickle=True)
        clean_texts = clean_data["texts"]
        assert all("sk-" not in str(t) for t in clean_texts)

    def test_report_summary(self, tmp_path: Path):
        """AirGapReport summary should be readable."""
        store_path = tmp_path / "store.npz"
        vectors = np.random.randn(5, 4).astype(np.float32)
        texts = np.array(["clean code"] * 5)
        keys = np.array([f"f{i}.py:0" for i in range(5)])
        np.savez(str(store_path), vectors=vectors, texts=texts, keys=keys)

        report = sanitize_embedding_store(store_path)
        summary = report.summary
        assert "SAFE" in summary
        assert "5" in summary  # chunk count
