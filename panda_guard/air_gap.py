"""
panda_guard/air_gap.py

Air-gap sanitization layer for embedding stores.

Scans embedding store chunk text for sensitive patterns (API keys,
emails, secrets) and ensures that only hashed binary representations
of vectors are ever uploaded — never raw floats or sensitive text.

This module is the "guard" half of the Panda Guard system.
The "merkle" half (merkle.py) provides integrity verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from panda_guard.merkle import MerkleTree, hash_vector, hash_text


# ── Sensitive Patterns ────────────────────────────────────────────────────────
# Patterns that should NEVER appear in uploaded embedding chunk text.

_SENSITIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("api_key_openai", re.compile(r"sk-[a-zA-Z0-9]{20,}", re.ASCII)),
    ("api_key_airtable", re.compile(r"pat[a-zA-Z0-9.]{30,}", re.ASCII)),
    ("api_key_perplexity", re.compile(r"pplx-[a-zA-Z0-9]{30,}", re.ASCII)),
    ("api_key_generic", re.compile(
        r"(?:api[_-]?key|secret|token|password)\s*[=:]\s*['\"]?[\w\-./+]{16,}",
        re.IGNORECASE,
    )),
    ("email_address", re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    )),
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}", re.ASCII)),
    ("github_token", re.compile(r"gh[ps]_[a-zA-Z0-9]{36,}", re.ASCII)),
    ("jwt_token", re.compile(
        r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}",
    )),
    ("private_key_header", re.compile(
        r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
    )),
    ("connection_string", re.compile(
        r"(?:postgres|mysql|mongodb|redis)://[^\s]{20,}",
        re.IGNORECASE,
    )),
]


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class SensitiveMatch:
    """A detected sensitive data match."""
    pattern_name: str
    chunk_key: str
    matched_text: str  # Truncated to first/last 4 chars for safety
    line_number: int = 0


@dataclass
class AirGapReport:
    """Report from scanning an embedding store for sensitive data."""
    total_chunks: int = 0
    scanned_chunks: int = 0
    sensitive_matches: list[SensitiveMatch] = field(default_factory=list)
    redacted_chunks: int = 0
    merkle_root: str = ""
    is_safe: bool = True

    @property
    def summary(self) -> str:
        status = "SAFE" if self.is_safe else "UNSAFE — SENSITIVE DATA FOUND"
        return (
            f"Panda Guard Scan: {status}\n"
            f"  Chunks scanned: {self.scanned_chunks}/{self.total_chunks}\n"
            f"  Sensitive matches: {len(self.sensitive_matches)}\n"
            f"  Redacted chunks: {self.redacted_chunks}\n"
            f"  Merkle root: {self.merkle_root[:16]}..."
        )


# ── Air Gap Functions ─────────────────────────────────────────────────────────

def scan_for_sensitive_data(
    texts: list[str] | np.ndarray,
    keys: list[str] | np.ndarray | None = None,
) -> list[SensitiveMatch]:
    """
    Scan chunk texts for sensitive patterns.

    Args:
        texts: Array of chunk text strings
        keys: Optional array of chunk keys for identification

    Returns:
        List of SensitiveMatch objects for each detection
    """
    matches: list[SensitiveMatch] = []

    for i, text in enumerate(texts):
        text_str = str(text)
        chunk_key = str(keys[i]) if keys is not None else f"chunk_{i}"

        for pattern_name, pattern in _SENSITIVE_PATTERNS:
            for match in pattern.finditer(text_str):
                matched = match.group()
                # Truncate for safety — show first/last 4 chars only
                if len(matched) > 12:
                    safe_preview = f"{matched[:4]}...{matched[-4:]}"
                else:
                    safe_preview = matched[:4] + "..."

                matches.append(SensitiveMatch(
                    pattern_name=pattern_name,
                    chunk_key=chunk_key,
                    matched_text=safe_preview,
                    line_number=text_str[:match.start()].count("\n") + 1,
                ))

    return matches


def redact_sensitive_text(text: str) -> str:
    """
    Redact sensitive patterns from chunk text.

    Replaces sensitive matches with [REDACTED:<pattern_name>] tokens.
    This preserves the surrounding context while removing secrets.
    """
    result = text
    for pattern_name, pattern in _SENSITIVE_PATTERNS:
        result = pattern.sub(f"[REDACTED:{pattern_name}]", result)
    return result


def hash_vectors_for_upload(vectors: np.ndarray) -> list[str]:
    """
    Convert float vectors → binary → SHA-256 hash.

    Only the hashes leave the machine, never the original vectors.
    This is the "air gap" — the hash function is the one-way gate.

    Args:
        vectors: (N, D) float32 matrix

    Returns:
        List of N hex-encoded SHA-256 hashes
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    return [hash_vector(v) for v in vectors]


def sanitize_embedding_store(
    npz_path: str | Path,
    redact: bool = True,
    output_path: str | Path | None = None,
) -> AirGapReport:
    """
    Full air-gap scan and sanitization of an embedding store.

    1. Loads the .npz file
    2. Scans all chunk texts for sensitive patterns
    3. Optionally redacts sensitive text and saves a clean copy
    4. Builds a Merkle tree over the (original) vectors
    5. Returns an AirGapReport

    Args:
        npz_path: Path to the embedding store .npz file
        redact: If True, create a sanitized copy with redacted text
        output_path: Where to save the sanitized copy (default: overwrite)

    Returns:
        AirGapReport with scan results and Merkle root
    """
    npz_path = Path(npz_path)
    report = AirGapReport()

    # Load store
    data = np.load(str(npz_path), allow_pickle=True)
    vectors = data["vectors"]
    texts = data["texts"]
    keys = data["keys"]

    report.total_chunks = len(vectors)
    report.scanned_chunks = len(texts)

    # Scan for sensitive data
    matches = scan_for_sensitive_data(texts, keys)
    report.sensitive_matches = matches
    report.is_safe = len(matches) == 0

    # Build Merkle tree over vectors (hashed binary)
    tree = MerkleTree.from_vectors(vectors)
    report.merkle_root = tree.root_hash

    # Redact if requested and sensitive data found
    if redact and matches:
        clean_texts = np.array([redact_sensitive_text(str(t)) for t in texts])
        report.redacted_chunks = sum(
            1 for orig, clean in zip(texts, clean_texts) if str(orig) != str(clean)
        )

        save_path = Path(output_path) if output_path else npz_path
        np.savez(
            str(save_path),
            keys=keys,
            vectors=vectors,
            texts=clean_texts,
        )

    return report


def verify_store_integrity(
    npz_path: str | Path,
    expected_root: str,
) -> bool:
    """
    Verify that an embedding store hasn't been tampered with.

    Rebuilds the Merkle tree from the vectors and checks the root hash
    against the expected value.

    Args:
        npz_path: Path to the .npz embedding store
        expected_root: Expected Merkle root hash

    Returns:
        True if the store is intact
    """
    tree = MerkleTree.from_npz(str(npz_path))
    return tree.root_hash == expected_root
