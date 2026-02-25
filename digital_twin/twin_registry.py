"""
digital_twin/twin_registry.py

Central state management for the A2A Digital Twin.
Stores task state, agent status, and CI results in a JSON file
that is committed to the repo by GitHub Actions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class TaskTwinNode:
    """A single task tracked by the Digital Twin."""
    task_id: str
    name: str
    airtable_record_id: str = ""
    status: str = "Backlog"
    agent_role: str = ""
    workflow_stage: str = "1-Intake"
    browser_actions_total: int = 0
    browser_actions_done: int = 0
    github_action: str = ""
    office_checkpoint: str = ""


@dataclass
class AgentTwinNode:
    """An agent registered with the Digital Twin."""
    agent_id: str
    agent_class: str
    status: str = "idle"
    current_task: str = ""
    system_prompt_hash: str = ""


@dataclass
class CIState:
    """CI/CD pipeline state."""
    last_run_sha: str = ""
    last_run_status: str = ""  # success | failure | in_progress
    last_run_url: str = ""
    passed_tests: int = 0
    failed_tests: int = 0
    coverage_pct: float = 0.0
    embedding_store_hash: str = ""


@dataclass
class TwinState:
    """Full Digital Twin state."""
    tasks: dict[str, TaskTwinNode] = field(default_factory=dict)
    agents: dict[str, AgentTwinNode] = field(default_factory=dict)
    ci: CIState = field(default_factory=CIState)
    version: str = "1.0.0"


# ── Registry ──────────────────────────────────────────────────────────────────

_DEFAULT_STATE_PATH = Path(
    os.environ.get("TWIN_STATE_FILE", "digital_twin/twin_state.json")
)


class TwinRegistry:
    """
    Load, mutate, and persist the Digital Twin state.

    Usage:
        twin = TwinRegistry()
        twin.load()
        twin.get().tasks["rec123"] = TaskTwinNode(...)
        twin.update_ci(sha, "success", 42, 0, 95.5, "https://...")
        twin.save()
    """

    def __init__(self, state_path: Path | str | None = None) -> None:
        self._path = Path(state_path) if state_path else _DEFAULT_STATE_PATH
        self._state = TwinState()

    def get(self) -> TwinState:
        """Return the current TwinState."""
        return self._state

    def load(self) -> None:
        """Load state from JSON file. No-op if file doesn't exist."""
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._state = _deserialize_twin_state(raw)
        except (json.JSONDecodeError, KeyError):
            # Corrupted file — start fresh
            self._state = TwinState()

    def save(self) -> None:
        """Persist state to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(_serialize_twin_state(self._state), f, indent=2)

    def update_ci(
        self,
        sha: str,
        status: str,
        passed: int,
        failed: int,
        coverage: float,
        run_url: str,
    ) -> None:
        """Update CI state from a GitHub Actions run."""
        self._state.ci.last_run_sha = sha
        self._state.ci.last_run_status = status
        self._state.ci.last_run_url = run_url
        self._state.ci.passed_tests = passed
        self._state.ci.failed_tests = failed
        self._state.ci.coverage_pct = coverage

    def get_summary(self) -> dict[str, Any]:
        """Return a human-readable summary of the twin state."""
        tasks = list(self._state.tasks.values())
        done = [t for t in tasks if t.status == "Done"]
        in_progress = [t for t in tasks if t.status == "In Progress"]
        blocked = [t for t in tasks if t.status == "Blocked"]
        agents = list(self._state.agents.values())
        active_agents = [a for a in agents if a.status == "active"]

        return {
            "total_tasks": len(tasks),
            "tasks_done": len(done),
            "tasks_in_progress": len(in_progress),
            "tasks_blocked": len(blocked),
            "total_agents": len(agents),
            "active_agents": len(active_agents),
            "ci_status": self._state.ci.last_run_status or "never_run",
            "ci_sha": self._state.ci.last_run_sha,
            "ci_coverage": self._state.ci.coverage_pct,
            "ci_passed": self._state.ci.passed_tests,
            "ci_failed": self._state.ci.failed_tests,
        }

    def sync_files(self, repo_root: Path) -> None:
        """
        Scan repo for agent definition files and register them.
        This is a lightweight discovery — just finds Python files in agents/.
        """
        agents_dir = repo_root / "agents"
        if not agents_dir.exists():
            return
        for py_file in agents_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            agent_id = py_file.stem
            if agent_id not in self._state.agents:
                self._state.agents[agent_id] = AgentTwinNode(
                    agent_id=agent_id,
                    agent_class=agent_id,
                )


# ── Serialization ─────────────────────────────────────────────────────────────

def _serialize_twin_state(state: TwinState) -> dict:
    """Convert TwinState to a JSON-serializable dict."""
    return {
        "version": state.version,
        "tasks": {k: asdict(v) for k, v in state.tasks.items()},
        "agents": {k: asdict(v) for k, v in state.agents.items()},
        "ci": asdict(state.ci),
    }


def _deserialize_twin_state(raw: dict) -> TwinState:
    """Reconstruct TwinState from a JSON dict."""
    state = TwinState(version=raw.get("version", "1.0.0"))

    for k, v in raw.get("tasks", {}).items():
        state.tasks[k] = TaskTwinNode(**v)

    for k, v in raw.get("agents", {}).items():
        state.agents[k] = AgentTwinNode(**v)

    ci_raw = raw.get("ci", {})
    if ci_raw:
        state.ci = CIState(**ci_raw)

    return state
