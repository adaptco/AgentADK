"""
tests/test_adk_subagent_spawner.py

Tests for the A2A Sub-Agent Spawner.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock
import pytest

# Import from the parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.adk_subagent_spawner import A2ASubagentSpawner, SpawnResult
from digital_twin.twin_registry import TwinRegistry, AgentTwinNode


class TestA2ASubagentSpawner:
    
    @pytest.fixture
    def mock_slicer(self):
        """Mock the VerticalTensorSlicer for semantic routing."""
        slicer = MagicMock()
        # Default behavior: return a specific agent and score
        slicer.route_to_agent.return_value = ("researcher", 0.95)
        return slicer

    @pytest.fixture
    def mock_twin(self):
        """Mock the TwinRegistry."""
        twin = MagicMock(spec=TwinRegistry)
        # Mock the internal state structure
        state_mock = MagicMock()
        state_mock.agents = {}
        twin.get.return_value = state_mock
        return twin

    def test_list_agents(self):
        """Should list default agents."""
        spawner = A2ASubagentSpawner()
        agents = spawner.list_agents()
        
        assert len(agents) > 0
        ids = [a["agent_id"] for a in agents]
        assert "coder" in ids
        assert "tester" in ids
        assert "managing_agent" in ids

    @pytest.mark.asyncio
    async def test_spawn_specific_agent(self):
        """Spawning with a specific ID should bypass routing."""
        spawner = A2ASubagentSpawner()
        result = await spawner.spawn("Fix the bug", agent_id="coder")
        
        assert result.agent_id == "coder"
        assert result.status == "completed"  # In-process default
        assert result.routing_score == 1.0

    @pytest.mark.asyncio
    async def test_spawn_invalid_agent(self):
        """Spawning a non-existent agent should return error."""
        spawner = A2ASubagentSpawner()
        result = await spawner.spawn("Task", agent_id="non_existent_agent")
        
        assert result.status == "error"
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_routing_keyword_fallback(self):
        """Should use keyword matching when no slicer is provided."""
        spawner = A2ASubagentSpawner(slicer=None)
        
        # "test" -> tester
        res_tester = await spawner.spawn("write unit tests for the module")
        assert res_tester.agent_id == "tester"
        
        # "design" -> architecture_agent
        res_arch = await spawner.spawn("design the system architecture")
        assert res_arch.agent_id == "architecture_agent"
        
        # "research" -> researcher
        res_research = await spawner.spawn("research the best library for X")
        assert res_research.agent_id == "researcher"

    @pytest.mark.asyncio
    async def test_routing_semantic(self, mock_slicer):
        """Should use slicer for routing when available."""
        spawner = A2ASubagentSpawner(slicer=mock_slicer)
        
        # Slicer is mocked to return "researcher"
        result = await spawner.spawn("some ambiguous task")
        
        assert result.agent_id == "researcher"
        assert result.routing_score == 0.95
        mock_slicer.route_to_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_twin_state_updates(self, mock_twin):
        """Should update twin state to active then idle."""
        spawner = A2ASubagentSpawner(twin=mock_twin)
        
        # We need a real dict for agents to track state changes if we want to be precise,
        # but with MagicMock we can verify the get() calls.
        # Let's use a real TwinRegistry for this test to verify logic.
        real_twin = TwinRegistry()
        # Pre-populate agent to avoid key error if logic checks existence
        real_twin.get().agents["coder"] = AgentTwinNode(agent_id="coder", agent_class="CoderAgent")
        
        spawner = A2ASubagentSpawner(twin=real_twin)
        await spawner.spawn("write code", agent_id="coder")
        
        # Since spawn is awaited, it finishes immediately in in-process mode.
        # The agent should be back to "idle"
        assert real_twin.get().agents["coder"].status == "idle"