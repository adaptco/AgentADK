"""
integrations/office/graph_checkpoint.py

Microsoft Graph API integration for Office 365 checkpoints.
Writes stage completion reports to Word, Excel, and Outlook.

Falls back to returning mock results when credentials are absent,
so tests and local dev work without Azure AD setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


_GRAPH_URL = "https://graph.microsoft.com/v1.0"


@dataclass
class CheckpointResult:
    """Result of a stage checkpoint operation."""
    word_doc_url: str = ""
    excel_url: str = ""
    email_sent: bool = False
    mock: bool = False


def _get_token() -> str | None:
    """
    Obtain an access token using client_credentials flow.
    Returns None if credentials are not configured.
    """
    tenant = os.environ.get("OFFICE_TENANT_ID", "")
    client_id = os.environ.get("OFFICE_CLIENT_ID", "")
    client_secret = os.environ.get("OFFICE_CLIENT_SECRET", "")

    if not all([tenant, client_id, client_secret, httpx]):
        return None

    token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": "https://graph.microsoft.com/.default",
    }

    resp = httpx.post(token_url, data=payload, timeout=15.0)
    resp.raise_for_status()
    return resp.json().get("access_token", "")


async def write_stage_checkpoint(
    task_name: str,
    agent_id: str,
    stage: str,
    summary: str,
    acceptance_criteria: list[str] | None = None,
    fossil_hashes: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
    checkpoint_type: str = "all",
    handoff_email: str = "",
) -> dict[str, Any]:
    """
    Write a stage checkpoint to Microsoft Office 365.

    checkpoint_type:
        "word"    — Write Word doc only
        "excel"   — Write Excel row only
        "outlook" — Send email only
        "all"     — All three

    Returns a dict with URLs/status for each action taken.
    """
    token = _get_token()
    if not token:
        # Offline stub — return mock result
        return asdict(CheckpointResult(
            word_doc_url="(offline — no Office credentials)",
            excel_url="(offline — no Office credentials)",
            email_sent=False,
            mock=True,
        ))

    result = CheckpointResult()
    user_email = os.environ.get("OFFICE_USER_EMAIL", "")
    handoff = handoff_email or os.environ.get("HANDOFF_EMAIL", "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # ── Word Document ─────────────────────────────────────────────────
        if checkpoint_type in ("word", "all"):
            doc_content = _build_word_content(
                task_name, agent_id, stage, summary,
                acceptance_criteria or [], fossil_hashes or [],
            )
            try:
                resp = await client.put(
                    f"{_GRAPH_URL}/users/{user_email}/drive/root:"
                    f"/A2A_Checkpoints/{task_name}.docx:/content",
                    headers={**headers, "Content-Type": "text/plain"},
                    content=doc_content.encode(),
                )
                if resp.status_code < 300:
                    result.word_doc_url = resp.json().get("webUrl", "")
            except Exception:
                pass

        # ── Excel Row ─────────────────────────────────────────────────────
        if checkpoint_type in ("excel", "all"):
            if metrics:
                row_values = [[str(v) for v in metrics.values()]]
                try:
                    resp = await client.post(
                        f"{_GRAPH_URL}/users/{user_email}/drive/root:"
                        f"/A2A_Checkpoints/metrics.xlsx:/workbook/"
                        f"worksheets/Sheet1/tables/Table1/rows/add",
                        headers=headers,
                        json={"values": row_values},
                    )
                    if resp.status_code < 300:
                        result.excel_url = "(row added)"
                except Exception:
                    pass

        # ── Outlook Email ─────────────────────────────────────────────────
        if checkpoint_type in ("outlook", "all") and handoff:
            email_body = (
                f"Stage Checkpoint: {stage}\n\n"
                f"Task: {task_name}\n"
                f"Agent: {agent_id}\n"
                f"Summary: {summary}\n"
            )
            try:
                await client.post(
                    f"{_GRAPH_URL}/users/{user_email}/sendMail",
                    headers=headers,
                    json={
                        "message": {
                            "subject": f"[A2A Twin] {stage} — {task_name}",
                            "body": {"contentType": "Text", "content": email_body},
                            "toRecipients": [
                                {"emailAddress": {"address": handoff}}
                            ],
                        }
                    },
                )
                result.email_sent = True
            except Exception:
                pass

    return asdict(result)


def _build_word_content(
    task_name: str,
    agent_id: str,
    stage: str,
    summary: str,
    acceptance_criteria: list[str],
    fossil_hashes: list[str],
) -> str:
    """Build plain-text content for the Word checkpoint document."""
    lines = [
        f"A2A Digital Twin — Stage Checkpoint",
        f"{'=' * 40}",
        f"",
        f"Task:     {task_name}",
        f"Agent:    {agent_id}",
        f"Stage:    {stage}",
        f"",
        f"Summary:",
        f"  {summary}",
        f"",
    ]

    if acceptance_criteria:
        lines.append("Acceptance Criteria:")
        for i, ac in enumerate(acceptance_criteria, 1):
            lines.append(f"  {i}. {ac}")
        lines.append("")

    if fossil_hashes:
        lines.append("Fossil Chain Hashes:")
        for h in fossil_hashes:
            lines.append(f"  - {h}")
        lines.append("")

    return "\n".join(lines)
