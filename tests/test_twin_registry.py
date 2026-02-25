"""
tests/test_twin_registry.py

Tests for the Digital Twin state management.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from digital_twin.twin_registry import (
    TwinRegistry,
    TwinState,
    TaskTwinNode,
    AgentTwinNode,
    CIState,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_state_file(tmp_path: Path) -> Path:
    return tmp_path / "twin_state.json"


@pytest.fixture
def registry(tmp_state_file: Path) -> TwinRegistry:
    return TwinRegistry(state_path=tmp_state_file)


@pytest.fixture
def populated_registry(registry: TwinRegistry) -> TwinRegistry:
    """Registry with sample data pre-loaded."""
    registry.get().tasks["task1"] = TaskTwinNode(
        task_id="task1",
        name="Implement feature X",
        status="In Progress",
        agent_role="coder",
    )
    registry.get().tasks["task2"] = TaskTwinNode(
        task_id="task2",
        name="Write tests",
        status="Done",
        agent_role="tester",
    )
    registry.get().tasks["task3"] = TaskTwinNode(
        task_id="task3",
        name="Deploy to prod",
        status="Blocked",
    )
    registry.get().agents["coder"] = AgentTwinNode(
        agent_id="coder",
        agent_class="CoderAgent",
        status="active",
        current_task="task1",
    )
    return registry


# ── Load / Save Round-Trip ────────────────────────────────────────────────────

class TestLoadSave:
    def test_save_and_load_empty(self, registry: TwinRegistry, tmp_state_file: Path):
        """Save empty state, reload, verify it matches."""
        registry.save()
        assert tmp_state_file.exists()

        loaded = TwinRegistry(state_path=tmp_state_file)
        loaded.load()
        assert loaded.get().version == "1.0.0"
        assert len(loaded.get().tasks) == 0

    def test_save_and_load_with_data(
        self, populated_registry: TwinRegistry, tmp_state_file: Path
    ):
        """Round-trip with tasks and agents."""
        populated_registry.save()

        loaded = TwinRegistry(state_path=tmp_state_file)
        loaded.load()

        assert len(loaded.get().tasks) == 3
        assert loaded.get().tasks["task1"].name == "Implement feature X"
        assert loaded.get().tasks["task1"].status == "In Progress"
        assert loaded.get().tasks["task2"].status == "Done"
        assert len(loaded.get().agents) == 1
        assert loaded.get().agents["coder"].agent_class == "CoderAgent"

    def test_load_missing_file(self, registry: TwinRegistry):
        """Load from non-existent file should not crash."""
        registry.load()
        assert len(registry.get().tasks) == 0

    def test_load_corrupted_file(self, tmp_state_file: Path):
        """Load from corrupted JSON should reset to empty state."""
        tmp_state_file.write_text("not valid json {{{")
        registry = TwinRegistry(state_path=tmp_state_file)
        registry.load()
        assert len(registry.get().tasks) == 0

    def test_json_structure(self, populated_registry: TwinRegistry, tmp_state_file: Path):
        """Verify the saved JSON has the expected structure."""
        populated_registry.save()
        with open(tmp_state_file) as f:
            data = json.load(f)

        assert "version" in data
        assert "tasks" in data
        assert "agents" in data
        assert "ci" in data
        assert data["tasks"]["task1"]["name"] == "Implement feature X"


# ── CI State Updates ──────────────────────────────────────────────────────────

class TestUpdateCI:
    def test_update_ci_success(self, registry: TwinRegistry):
        registry.update_ci(
            sha="abc1234",
            status="success",
            passed=42,
            failed=0,
            coverage=95.5,
            run_url="https://github.com/example/actions/runs/123",
        )
        ci = registry.get().ci
        assert ci.last_run_sha == "abc1234"
        assert ci.last_run_status == "success"
        assert ci.passed_tests == 42
        assert ci.failed_tests == 0
        assert ci.coverage_pct == 95.5
        assert "123" in ci.last_run_url

    def test_update_ci_failure(self, registry: TwinRegistry):
        registry.update_ci("def5678", "failure", 38, 4, 72.3, "https://x")
        ci = registry.get().ci
        assert ci.last_run_status == "failure"
        assert ci.failed_tests == 4

    def test_ci_round_trip(self, registry: TwinRegistry, tmp_state_file: Path):
        """CI state survives save/load."""
        registry.update_ci("abc", "success", 10, 0, 80.0, "https://x")
        registry.save()

        loaded = TwinRegistry(state_path=tmp_state_file)
        loaded.load()
        assert loaded.get().ci.last_run_sha == "abc"
        assert loaded.get().ci.coverage_pct == 80.0


# ── Summary ───────────────────────────────────────────────────────────────────

class TestGetSummary:
    def test_empty_summary(self, registry: TwinRegistry):
        summary = registry.get_summary()
        assert summary["total_tasks"] == 0
        assert summary["tasks_done"] == 0
        assert summary["ci_status"] == "never_run"

    def test_populated_summary(self, populated_registry: TwinRegistry):
        populated_registry.update_ci("sha", "success", 10, 0, 85.0, "url")
        summary = populated_registry.get_summary()

        assert summary["total_tasks"] == 3
        assert summary["tasks_done"] == 1
        assert summary["tasks_in_progress"] == 1
        assert summary["tasks_blocked"] == 1
        assert summary["total_agents"] == 1
        assert summary["active_agents"] == 1
        assert summary["ci_status"] == "success"
        assert summary["ci_coverage"] == 85.0


# ── Sync Files ────────────────────────────────────────────────────────────────

class TestSyncFiles:
    def test_sync_discovers_agents(self, registry: TwinRegistry, tmp_path: Path):
        """sync_files should discover Python files in agents/."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "coder.py").write_text("# agent")
        (agents_dir / "tester.py").write_text("# agent")
        (agents_dir / "__init__.py").write_text("# skip")

        registry.sync_files(tmp_path)
        assert "coder" in registry.get().agents
        assert "tester" in registry.get().agents
        assert "__init__" not in registry.get().agents

    def test_sync_no_agents_dir(self, registry: TwinRegistry, tmp_path: Path):
        """sync_files should not crash if agents/ doesn't exist."""
        registry.sync_files(tmp_path)
        assert len(registry.get().agents) == 0
