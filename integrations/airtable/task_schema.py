"""
integrations/airtable/task_schema.py

Airtable API client for the A2A Digital Twin.
Manages Tasks, Roles, Workflows, and Actions tables.

Falls back to mock data when AIRTABLE_API_KEY is not set,
so tests and CI can run without live credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


# ── Enums ─────────────────────────────────────────────────────────────────────

class TaskStatus(Enum):
    BACKLOG = "Backlog"
    READY = "Ready"
    IN_PROGRESS = "In Progress"
    IN_REVIEW = "In Review"
    DONE = "Done"
    BLOCKED = "Blocked"


class AgentRole(Enum):
    MANAGING_AGENT = "managing_agent"
    ORCHESTRATION_AGENT = "orchestration_agent"
    ARCHITECTURE_AGENT = "architecture_agent"
    CODER = "coder"
    TESTER = "tester"
    RESEARCHER = "researcher"
    JUDGE = "judge"
    DIGITAL_TWIN = "digital_twin"


class WorkflowStage(Enum):
    INTAKE = "1-Intake"
    RESEARCH = "2-Research"
    ARCHITECT = "3-Architect"
    IMPLEMENT = "4-Implement"
    VERIFY = "5-Verify"
    CHECKPOINT = "6-Checkpoint"
    DEPLOY = "7-Deploy"


class WorkflowTrigger(Enum):
    MANUAL = "manual"
    PUSH = "push"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class AirtableTask:
    """A task record from the Airtable Tasks table."""
    record_id: str
    name: str
    status: TaskStatus = TaskStatus.BACKLOG
    agent_role: str = ""
    workflow_stage: str = "1-Intake"
    description: str = ""
    acceptance_criteria: str = ""
    browser_steps: list[str] = field(default_factory=list)
    github_action: str = ""
    office_checkpoint: str = ""
    related_tasks: list[str] = field(default_factory=list)


@dataclass
class AirtableRole:
    """A role record from the Airtable Roles table."""
    record_id: str
    name: str
    agent_class: str = ""
    system_prompt: str = ""
    tools: list[str] = field(default_factory=list)
    mcp_tools: list[str] = field(default_factory=list)


@dataclass
class AirtableWorkflow:
    """A workflow record from the Airtable Workflows table."""
    record_id: str
    name: str
    stages: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    trigger: str = "manual"
    github_action_file: str = ""


@dataclass
class AirtableAction:
    """A GitHub Actions tracking record."""
    record_id: str
    name: str
    run_id: str = ""
    status: str = "in_progress"
    triggered_by: list[str] = field(default_factory=list)
    run_url: str = ""
    timestamp: str = ""


# ── Client ────────────────────────────────────────────────────────────────────

_AIRTABLE_API_URL = "https://api.airtable.com/v0"


class AirtableClient:
    """
    Airtable REST API client.

    Reads AIRTABLE_API_KEY and AIRTABLE_BASE_ID from environment.
    When credentials are absent, all methods return mock data so
    tests and local development work without a live Airtable base.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_id: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("AIRTABLE_API_KEY", "")
        self._base_id = base_id or os.environ.get("AIRTABLE_BASE_ID", "")
        self._is_live = bool(self._api_key and self._base_id and httpx)

    @property
    def is_live(self) -> bool:
        """True if we have valid credentials and httpx is available."""
        return self._is_live

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def list_tasks(self) -> list[AirtableTask]:
        """List all tasks from the Tasks table."""
        if not self._is_live:
            return _mock_tasks()

        url = f"{_AIRTABLE_API_URL}/{self._base_id}/Tasks"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        tasks = []
        for record in data.get("records", []):
            fields = record.get("fields", {})
            browser_text = fields.get("Browser Steps", "")
            browser_steps = [
                s.strip() for s in browser_text.split("\n") if s.strip()
            ] if browser_text else []

            tasks.append(AirtableTask(
                record_id=record["id"],
                name=fields.get("Name", ""),
                status=TaskStatus(fields.get("Status", "Backlog")),
                agent_role=fields.get("Agent Role", ""),
                workflow_stage=fields.get("Workflow Stage", "1-Intake"),
                description=fields.get("Description", ""),
                acceptance_criteria=fields.get("Acceptance Criteria", ""),
                browser_steps=browser_steps,
                github_action=fields.get("GitHub Action", ""),
                office_checkpoint=fields.get("Office Checkpoint", ""),
                related_tasks=fields.get("Related Tasks", []),
            ))
        return tasks

    async def update_task_status(
        self, record_id: str, status: TaskStatus
    ) -> dict[str, Any]:
        """Update a task's status in Airtable."""
        if not self._is_live:
            return {"id": record_id, "status": status.value, "mock": True}

        url = f"{_AIRTABLE_API_URL}/{self._base_id}/Tasks/{record_id}"
        payload = {"fields": {"Status": status.value}}
        async with httpx.AsyncClient() as client:
            resp = await client.patch(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()

    async def list_roles(self) -> list[AirtableRole]:
        """List all roles from the Roles table."""
        if not self._is_live:
            return _mock_roles()

        url = f"{_AIRTABLE_API_URL}/{self._base_id}/Roles"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()

        return [
            AirtableRole(
                record_id=r["id"],
                name=r["fields"].get("Name", ""),
                agent_class=r["fields"].get("Agent Class", ""),
                system_prompt=r["fields"].get("System Prompt", ""),
                tools=[
                    t.strip()
                    for t in r["fields"].get("Tools", "").split(",")
                    if t.strip()
                ],
                mcp_tools=[
                    t.strip()
                    for t in r["fields"].get("MCP Tools", "").split(",")
                    if t.strip()
                ],
            )
            for r in data.get("records", [])
        ]

    async def create_action(
        self,
        name: str,
        run_id: str,
        status: str,
        run_url: str,
        triggered_by: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new Actions record to track a GitHub Actions run."""
        if not self._is_live:
            return {"name": name, "run_id": run_id, "mock": True}

        url = f"{_AIRTABLE_API_URL}/{self._base_id}/Actions"
        payload = {
            "fields": {
                "Name": name,
                "Run ID": run_id,
                "Status": status,
                "Run URL": run_url,
            }
        }
        if triggered_by:
            payload["fields"]["Triggered By"] = triggered_by

        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            return resp.json()


# ── Mock Data (for offline / test use) ────────────────────────────────────────

def _mock_tasks() -> list[AirtableTask]:
    """Return sample tasks for offline development."""
    return [
        AirtableTask(
            record_id="recMOCK001",
            name="Implement Postgres ArtifactStore",
            status=TaskStatus.IN_PROGRESS,
            agent_role="coder",
            workflow_stage="4-Implement",
            description="Replace InMemoryArtifactStore with Postgres-backed version.",
            acceptance_criteria="All existing tests pass\nNew integration test added",
            browser_steps=["navigate to /artifacts", "verify table renders"],
        ),
        AirtableTask(
            record_id="recMOCK002",
            name="Add RAG embedding search",
            status=TaskStatus.DONE,
            agent_role="researcher",
            workflow_stage="5-Verify",
            description="Build vertical tensor slice for semantic repo search.",
        ),
        AirtableTask(
            record_id="recMOCK003",
            name="Set up CI/CD pipeline",
            status=TaskStatus.BACKLOG,
            agent_role="orchestration_agent",
            workflow_stage="1-Intake",
            description="Configure GitHub Actions for the A2A Digital Twin.",
        ),
    ]


def _mock_roles() -> list[AirtableRole]:
    """Return sample roles for offline development."""
    return [
        AirtableRole(
            record_id="recROLE001",
            name="Coder",
            agent_class="CoderAgent",
            system_prompt="You are a senior software engineer...",
            tools=["read_file", "write_file", "run_tests", "git_commit"],
            mcp_tools=["search_repo"],
        ),
        AirtableRole(
            record_id="recROLE002",
            name="Researcher",
            agent_class="ResearcherAgent",
            system_prompt="You are a research agent...",
            tools=["search_web", "search_repo"],
            mcp_tools=["search_web"],
        ),
    ]