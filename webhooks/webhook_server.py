"""
webhooks/webhook_server.py

Inbound webhook receiver for GitHub and Airtable events.
Validates HMAC signatures, parses payloads, and dispatches
to GitHub Actions via workflow_dispatch or repository_dispatch.

Usage:
    python webhooks/webhook_server.py                  # start server on :9090
    python webhooks/webhook_server.py --port 8888      # custom port
    python webhooks/webhook_server.py --dry-run        # print config and exit
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


# ── Configuration ─────────────────────────────────────────────────────────────

_GITHUB_WEBHOOK_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_GITHUB_REPO = os.environ.get("GITHUB_REPOSITORY", "adaptco-main/A2A_MCP")
_LISTEN_HOST = os.environ.get("WEBHOOK_HOST", "0.0.0.0")
_LISTEN_PORT = int(os.environ.get("WEBHOOK_PORT", "9090"))


# ── HMAC Signature Validation ─────────────────────────────────────────────────

def validate_github_signature(
    payload: bytes,
    signature: str,
    secret: str = "",
) -> bool:
    """
    Validate a GitHub webhook HMAC-SHA256 signature.

    GitHub sends the signature in the X-Hub-Signature-256 header
    as "sha256=<hex_digest>".
    """
    secret = secret or _GITHUB_WEBHOOK_SECRET
    if not secret:
        # No secret configured — skip validation (dev mode)
        return True

    if not signature.startswith("sha256="):
        return False

    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(expected, signature)


# ── Event Dispatch ────────────────────────────────────────────────────────────

async def dispatch_workflow(
    event_type: str,
    payload: dict[str, Any],
    repo: str = "",
) -> dict[str, Any]:
    """
    Dispatch a repository_dispatch event to GitHub Actions.

    This triggers workflows that listen for repository_dispatch events,
    passing the payload as client_payload.
    """
    repo = repo or _GITHUB_REPO
    token = _GITHUB_TOKEN

    if not token or not httpx:
        return {
            "dispatched": False,
            "reason": "No GITHUB_TOKEN or httpx not installed",
            "event_type": event_type,
        }

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    body = {
        "event_type": event_type,
        "client_payload": payload,
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        return {
            "dispatched": resp.status_code == 204,
            "status_code": resp.status_code,
            "event_type": event_type,
        }


def parse_github_event(
    headers: dict[str, str],
    body: dict[str, Any],
) -> dict[str, Any]:
    """
    Parse a GitHub webhook event into a normalized structure.

    Returns a dict with:
        event_type: The GitHub event type (push, pull_request, etc.)
        action: The sub-action (opened, closed, synchronize, etc.)
        ref: The git ref (for push events)
        pr_number: The PR number (for pull_request events)
        sender: The GitHub username that triggered the event
    """
    event_type = headers.get("x-github-event", "unknown")

    return {
        "event_type": event_type,
        "action": body.get("action", ""),
        "ref": body.get("ref", ""),
        "pr_number": body.get("number", 0),
        "sender": body.get("sender", {}).get("login", "unknown"),
        "repository": body.get("repository", {}).get("full_name", ""),
        "head_sha": _extract_sha(event_type, body),
    }


def _extract_sha(event_type: str, body: dict[str, Any]) -> str:
    """Extract the relevant commit SHA from a webhook payload."""
    if event_type == "push":
        return body.get("after", "")[:7]
    elif event_type == "pull_request":
        return body.get("pull_request", {}).get("head", {}).get("sha", "")[:7]
    return ""


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class WebhookHandler(BaseHTTPRequestHandler):
    """HTTP request handler for webhook payloads."""

    def do_POST(self) -> None:
        """Handle POST requests (webhook payloads)."""
        content_length = int(self.headers.get("Content-Length", 0))
        payload = self.rfile.read(content_length)

        # Validate signature
        signature = self.headers.get("X-Hub-Signature-256", "")
        if not validate_github_signature(payload, signature):
            self.send_error(401, "Invalid signature")
            return

        # Parse body
        try:
            body = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # Parse event
        headers_dict = {k.lower(): v for k, v in self.headers.items()}
        event = parse_github_event(headers_dict, body)

        # Log
        print(
            f"[webhook] {event['event_type']}"
            f" action={event['action']}"
            f" sender={event['sender']}"
            f" sha={event['head_sha']}"
        )

        # Route
        response = self._route_event(event, body)

        # Respond
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_GET(self) -> None:
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "ok",
            "service": "a2a-digital-twin-webhook",
        }).encode("utf-8"))

    def _route_event(
        self, event: dict[str, Any], raw_body: dict[str, Any]
    ) -> dict[str, Any]:
        """Route parsed event to the appropriate handler."""
        event_type = event["event_type"]
        action = event["action"]

        if event_type == "pull_request" and action in ("opened", "synchronize"):
            return {
                "routed_to": "pr_validation",
                "pr_number": event["pr_number"],
                "sha": event["head_sha"],
            }
        elif event_type == "push":
            return {
                "routed_to": "main_pipeline",
                "ref": event["ref"],
                "sha": event["head_sha"],
            }
        elif event_type == "workflow_run":
            return {
                "routed_to": "ci_status_update",
                "action": action,
            }
        else:
            return {
                "routed_to": "unhandled",
                "event_type": event_type,
                "action": action,
            }

    def log_message(self, format: str, *args: Any) -> None:
        """Override to use consistent log format."""
        print(f"[webhook-http] {args[0]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A2A Digital Twin Webhook Server"
    )
    parser.add_argument(
        "--port", type=int, default=_LISTEN_PORT, help="Port to listen on"
    )
    parser.add_argument(
        "--host", type=str, default=_LISTEN_HOST, help="Host to bind to"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print config and exit"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("A2A Digital Twin — Webhook Server Configuration")
        print(f"  Host:        {args.host}")
        print(f"  Port:        {args.port}")
        print(f"  Secret:      {'configured' if _GITHUB_WEBHOOK_SECRET else 'NOT SET'}")
        print(f"  GitHub repo: {_GITHUB_REPO}")
        print(f"  GitHub token: {'configured' if _GITHUB_TOKEN else 'NOT SET'}")
        print(f"  httpx:       {'available' if httpx else 'NOT INSTALLED'}")
        print("  Status:      dry-run — exiting")
        return

    server = HTTPServer((args.host, args.port), WebhookHandler)
    print(f"A2A Digital Twin webhook server listening on {args.host}:{args.port}")
    print("  POST /  → process webhook payload")
    print("  GET  /  → health check")
    print(f"  HMAC validation: {'enabled' if _GITHUB_WEBHOOK_SECRET else 'disabled (no secret)'}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down webhook server")
        server.server_close()


if __name__ == "__main__":
    main()
