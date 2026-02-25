"""
tests/test_webhook_server.py

Tests for the webhook server.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from webhooks.webhook_server import (
    validate_github_signature,
    parse_github_event,
)


# ── HMAC Signature Validation ─────────────────────────────────────────────────

class TestValidateGithubSignature:
    def test_valid_signature(self):
        """Correct HMAC should validate."""
        secret = "test-secret-123"
        payload = b'{"action": "opened"}'
        sig = "sha256=" + hmac.new(
            secret.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()

        assert validate_github_signature(payload, sig, secret)

    def test_invalid_signature(self):
        """Wrong HMAC should fail."""
        secret = "test-secret-123"
        payload = b'{"action": "opened"}'
        bad_sig = "sha256=0000000000000000000000000000000000000000000000000000000000000000"

        assert not validate_github_signature(payload, bad_sig, secret)

    def test_no_prefix(self):
        """Signature without sha256= prefix should fail."""
        payload = b'{"action": "opened"}'
        assert not validate_github_signature(payload, "invalid", "secret")

    def test_empty_secret_skips_validation(self):
        """No secret configured → skip validation (dev mode)."""
        payload = b'{"action": "opened"}'
        assert validate_github_signature(payload, "", "")

    def test_tampered_payload(self):
        """Signature for original payload should fail for tampered payload."""
        secret = "mysecret"
        original = b'{"action": "opened"}'
        tampered = b'{"action": "closed"}'
        sig = "sha256=" + hmac.new(
            secret.encode("utf-8"), original, hashlib.sha256
        ).hexdigest()

        assert validate_github_signature(original, sig, secret)
        assert not validate_github_signature(tampered, sig, secret)


# ── Event Parsing ─────────────────────────────────────────────────────────────

class TestParseGithubEvent:
    def test_push_event(self):
        headers = {"x-github-event": "push"}
        body = {
            "ref": "refs/heads/main",
            "after": "abc1234567890",
            "sender": {"login": "dev-user"},
            "repository": {"full_name": "org/repo"},
        }
        event = parse_github_event(headers, body)

        assert event["event_type"] == "push"
        assert event["ref"] == "refs/heads/main"
        assert event["head_sha"] == "abc1234"  # truncated to 7
        assert event["sender"] == "dev-user"
        assert event["repository"] == "org/repo"

    def test_pull_request_event(self):
        headers = {"x-github-event": "pull_request"}
        body = {
            "action": "opened",
            "number": 42,
            "sender": {"login": "contributor"},
            "repository": {"full_name": "org/repo"},
            "pull_request": {
                "head": {"sha": "def5678901234567890"},
            },
        }
        event = parse_github_event(headers, body)

        assert event["event_type"] == "pull_request"
        assert event["action"] == "opened"
        assert event["pr_number"] == 42
        assert event["head_sha"] == "def5678"

    def test_workflow_run_event(self):
        headers = {"x-github-event": "workflow_run"}
        body = {
            "action": "completed",
            "sender": {"login": "github-actions[bot]"},
            "repository": {"full_name": "org/repo"},
        }
        event = parse_github_event(headers, body)
        assert event["event_type"] == "workflow_run"
        assert event["action"] == "completed"

    def test_unknown_event(self):
        headers = {}
        body = {}
        event = parse_github_event(headers, body)
        assert event["event_type"] == "unknown"
        assert event["sender"] == "unknown"

    def test_missing_fields(self):
        """Missing fields should not crash, just use defaults."""
        headers = {"x-github-event": "push"}
        body = {"ref": "refs/heads/feature"}
        event = parse_github_event(headers, body)
        assert event["event_type"] == "push"
        assert event["sender"] == "unknown"
        assert event["head_sha"] == ""
