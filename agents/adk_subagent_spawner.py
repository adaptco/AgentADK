"""
agents/adk_subagent_spawner.py

A2A Sub-Agent Spawner using Normalized Dot Product routing.
Routes task descriptions to the best-matching agent based on
semantic similarity between the task embedding and agent capability vectors.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpawnResult:
    """Result of spawning a sub-agent."""
    agent_id: str
    agent_class: str
    task: str
    status: str = "spawned"
    routing_score: float = 0.0
    output: str = ""
    error: str = ""


# ── Default agent definitions ─────────────────────────────────────────────────
# These represent the built-in agent roles from SETUP.md

_DEFAULT_AGENTS: dict[str, dict[str, str]] = {
    "managing_agent": {
        "class": "ManagingAgent",
        "prompt": (
            "You are the managing agent. You break down high-level goals into "
            "actionable tasks, assign them to specialist agents, and track "
            "progress to completion."
        ),
    },
    "orchestration_agent": {
        "class": "OrchestrationAgent",
        "prompt": (
            "You are the orchestration agent. You coordinate multi-step workflows, "
            "manage dependencies between tasks, and ensure the pipeline executes "
            "in the correct order."
        ),
    },
    "architecture_agent": {
        "class": "ArchitectureAgent",
        "prompt": (
            "You are the architecture agent. You design system architecture, "
            "evaluate technical trade-offs, define interfaces, and review "
            "code for structural integrity."
        ),
    },
    "coder": {
        "class": "CoderAgent",
        "prompt": (
            "You are a senior software engineer. You implement features, "
            "fix bugs, write clean production-quality code, and create unit tests. "
            "You use search_repo to find relevant code, write_file to create code, "
            "and run_tests to verify your work."
        ),
    },
    "tester": {
        "class": "TesterAgent",
        "prompt": (
            "You are the testing agent. You write comprehensive test suites, "
            "perform integration testing, validate acceptance criteria, "
            "and report coverage metrics."
        ),
    },
    "researcher": {
        "class": "ResearcherAgent",
        "prompt": (
            "You are the research agent. You search documentation, explore "
            "APIs, gather technical information, and synthesize findings "
            "into actionable recommendations."
        ),
    },
    "judge": {
        "class": "JudgeAgent",
        "prompt": (
            "You are the judge agent. You review code quality, validate "
            "that acceptance criteria are met, verify fossil chain integrity, "
            "and approve or reject task completions."
        ),
    },
    "digital_twin": {
        "class": "DigitalTwinAgent",
        "prompt": (
            "You are the digital twin agent. You maintain the twin state, "
            "synchronize Airtable records, update CI status, and provide "
            "system-wide observability."
        ),
    },
}


class A2ASubagentSpawner:
    """
    Routes tasks to agents via NDP (Normalized Dot Product) scoring
    and spawns them for execution.

    Uses the VerticalTensorSlicer for semantic routing when available,
    falls back to keyword matching otherwise.

    Modes:
        "in-process" — Execute the agent logic in the current process
        "a2a"        — Send task via A2A protocol to a remote agent
    """

    def __init__(
        self,
        slicer: Any | None = None,
        twin: Any | None = None,
        mode: str = "in-process",
        agents: dict[str, dict[str, str]] | None = None,
    ) -> None:
        self._slicer = slicer
        self._twin = twin
        self._mode = mode
        self._agents = agents or _DEFAULT_AGENTS

    def list_agents(self) -> list[dict[str, str]]:
        """List all registered agents with their roles and status."""
        result = []
        for agent_id, info in self._agents.items():
            status = "idle"
            current_task = ""

            # Check twin for active status
            if self._twin is not None:
                try:
                    twin_agents = self._twin.get().agents
                    if agent_id in twin_agents:
                        status = twin_agents[agent_id].status
                        current_task = twin_agents[agent_id].current_task
                except Exception:
                    pass

            result.append({
                "agent_id": agent_id,
                "agent_class": info["class"],
                "status": status,
                "current_task": current_task,
            })
        return result

    async def spawn(
        self,
        task: str,
        agent_id: str | None = None,
    ) -> SpawnResult:
        """
        Spawn an agent for a task.

        If agent_id is provided, use that agent directly.
        Otherwise, use NDP routing to find the best match.
        """
        # Route to best agent if not specified
        if agent_id is None:
            agent_id, score = self._route(task)
        else:
            score = 1.0

        if agent_id not in self._agents:
            return SpawnResult(
                agent_id=agent_id or "unknown",
                agent_class="",
                task=task,
                status="error",
                error=f"Agent '{agent_id}' not found",
            )

        agent_info = self._agents[agent_id]

        # Update twin state
        if self._twin is not None:
            try:
                from digital_twin.twin_registry import AgentTwinNode
                self._twin.get().agents[agent_id] = AgentTwinNode(
                    agent_id=agent_id,
                    agent_class=agent_info["class"],
                    status="active",
                    current_task=task[:100],
                )
            except Exception:
                pass

        # Execute based on mode
        if self._mode == "in-process":
            result = await self._run_in_process(agent_id, agent_info, task)
        else:
            result = await self._run_a2a(agent_id, agent_info, task)

        result.routing_score = score

        # Update twin — mark agent as idle
        if self._twin is not None:
            try:
                if agent_id in self._twin.get().agents:
                    self._twin.get().agents[agent_id].status = "idle"
                    self._twin.get().agents[agent_id].current_task = ""
            except Exception:
                pass

        return result

    def _route(self, task: str) -> tuple[str, float]:
        """
        Route a task to the best agent using NDP scoring.

        Falls back to keyword matching if no slicer is available.
        """
        # Try semantic routing via VerticalTensorSlicer
        if self._slicer is not None:
            try:
                agent_prompts = {
                    aid: info["prompt"] for aid, info in self._agents.items()
                }
                agent_id, score = self._slicer.route_to_agent(task, agent_prompts)
                return agent_id, score
            except Exception:
                pass

        # Fallback: keyword matching
        task_lower = task.lower()

        keyword_map = {
            "coder": ["implement", "code", "write", "create", "build", "fix", "bug"],
            "tester": ["test", "verify", "validate", "assert", "coverage"],
            "researcher": ["research", "search", "find", "explore", "investigate"],
            "architecture_agent": ["design", "architect", "interface", "schema"],
            "judge": ["review", "approve", "reject", "quality", "criteria"],
            "orchestration_agent": ["orchestrate", "coordinate", "pipeline", "workflow"],
            "managing_agent": ["manage", "plan", "assign", "track", "goal"],
            "digital_twin": ["twin", "state", "sync", "airtable", "ci"],
        }

        best_id = "coder"  # default
        best_score = 0.0

        for aid, keywords in keyword_map.items():
            matches = sum(1 for kw in keywords if kw in task_lower)
            score = matches / len(keywords) if keywords else 0
            if score > best_score:
                best_score = score
                best_id = aid

        return best_id, best_score

    async def _run_in_process(
        self,
        agent_id: str,
        agent_info: dict[str, str],
        task: str,
    ) -> SpawnResult:
        """Execute agent logic in-process (simulated)."""
        # In a full implementation, this would instantiate the agent class
        # and run its execute() method. For now, return a placeholder.
        return SpawnResult(
            agent_id=agent_id,
            agent_class=agent_info["class"],
            task=task,
            status="completed",
            output=(
                f"Agent {agent_info['class']} processed task: {task[:80]}... "
                f"(in-process mode — implement {agent_info['class']}.execute() "
                f"for full agent logic)"
            ),
        )

    async def _run_a2a(
        self,
        agent_id: str,
        agent_info: dict[str, str],
        task: str,
    ) -> SpawnResult:
        """Send task via A2A protocol to a remote agent."""
        # Placeholder for A2A protocol communication
        return SpawnResult(
            agent_id=agent_id,
            agent_class=agent_info["class"],
            task=task,
            status="dispatched",
            output=(
                f"Task dispatched to {agent_info['class']} via A2A protocol. "
                f"(Remote agent endpoint not yet configured)"
            ),
        )