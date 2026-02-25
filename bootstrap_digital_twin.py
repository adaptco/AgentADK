"""
bootstrap_digital_twin.py

Drop this in the root of your A2A_MCP repo alongside bootstrap.py.
Run it once to wire all the new components onto the existing MCPHub.

Usage:
    python bootstrap_digital_twin.py                  # start full system
    python bootstrap_digital_twin.py --build-rag       # rebuild embeddings only
    python bootstrap_digital_twin.py --sync-airtable   # sync Airtable only
    python bootstrap_digital_twin.py --test-tools      # test all MCP tools
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# ── Ensure A2A_MCP root is on sys.path (mirrors existing bootstrap.py) ───────
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))


async def build_rag(force: bool = False) -> None:
    print("── Building Vertical Tensor Slice ────────────────────────────────")
    from rag.vertical_tensor_slice import build_embedding_store
    out = REPO_ROOT / "rag" / "embedding_store.npz"
    if out.exists() and not force:
        print(f"  Store exists: {out} — use --build-rag --force to rebuild")
        return
    build_embedding_store(repo=REPO_ROOT, out_path=out)


async def sync_airtable() -> None:
    print("── Syncing Airtable → Digital Twin ───────────────────────────────")
    from integrations.airtable.task_schema import AirtableClient, TaskStatus
    from digital_twin.twin_registry import TwinRegistry, TaskTwinNode

    client = AirtableClient()
    twin = TwinRegistry()
    twin.load()

    tasks = await client.list_tasks()
    for t in tasks:
        twin.get().tasks[t.record_id] = TaskTwinNode(
            task_id=t.record_id,
            name=t.name,
            airtable_record_id=t.record_id,
            status=t.status.value,
            browser_actions_total=len(t.browser_steps),
        )
    twin.sync_files(REPO_ROOT)
    twin.save()
    print(f"  Synced {len(tasks)} tasks. Twin state: {twin.get_summary()}")


async def test_tools() -> None:
    print("── Testing MCP Tools ─────────────────────────────────────────────")
    from mcp_extensions.claude_code_mcp_server import (
        load_embedding_store, get_embedding, ndp_search
    )
    load_embedding_store(REPO_ROOT / "rag" / "embedding_store.npz")

    q = "how does IntentEngine route tasks between agents"
    vec = get_embedding(q)
    results = ndp_search(vec, top_k=3)

    print(f"  Query: {q}")
    print(f"  Embedding dim: {len(vec)} | norm: {sum(v**2 for v in vec)**0.5:.4f}")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['path']}")


async def start_mcp_server() -> None:
    """
    Extend the existing MCPHub with all Digital Twin tools,
    then start the server.
    """
    print("── Starting Extended MCP Server ──────────────────────────────────")

    from fastmcp import FastMCP
    from mcp_extensions.claude_code_mcp_server import (
        register_claude_code_tools, load_embedding_store
    )
    from integrations.perplexity.search_agent import PerplexitySearchAgent
    from rag.vertical_tensor_slice import VerticalTensorSlicer
    from digital_twin.twin_registry import TwinRegistry
    from agents.adk_subagent_spawner import A2ASubagentSpawner

    # ── Load existing MCPHub if available ─────────────────────────────────────
    try:
        from orchestrator.main import MCPHub
        print("  ✓ Existing MCPHub loaded")
    except ImportError:
        MCPHub = None
        print("  ⚠ MCPHub not found — starting standalone MCP server")

    # ── Build extension layer ─────────────────────────────────────────────────
    mcp = FastMCP("a2a-digital-twin")

    # 1. Claude Code replacement tools
    register_claude_code_tools(mcp)
    print("  ✓ Claude Code tools registered (read_file, write_file, search_repo, run_tests, git_commit)")

    # 2. Load embedding store
    store_path = REPO_ROOT / "rag" / "embedding_store.npz"
    load_embedding_store(store_path)
    print(f"  ✓ Embedding store loaded: {store_path}")

    # 3. Perplexity search tool
    slicer = None
    if store_path.exists():
        slicer = VerticalTensorSlicer(store_path)
    search_agent = PerplexitySearchAgent(slicer=slicer)

    @mcp.tool()
    async def search_web(query: str, agent_filter: str = "") -> dict:
        """NDP repo search + Perplexity fallthrough for web knowledge."""
        return await search_agent._tool_fn(query, agent_filter)

    print("  ✓ Perplexity search tool registered")

    # 4. Digital Twin tools
    twin = TwinRegistry()
    twin.load()

    @mcp.tool()
    async def get_twin_state() -> dict:
        """Get current Digital Twin summary — tasks, agents, CI, coverage."""
        return twin.get_summary()

    @mcp.tool()
    async def get_twin_tasks(status: str = "") -> dict:
        """List tasks from the Digital Twin, optionally filtered by status."""
        tasks = list(twin.get().tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        from dataclasses import asdict
        return {"tasks": [asdict(t) for t in tasks]}

    @mcp.tool()
    async def list_agents() -> dict:
        """List all registered A2A agents and their current status."""
        spawner = A2ASubagentSpawner(slicer=slicer, twin=twin)
        return {"agents": spawner.list_agents()}

    @mcp.tool()
    async def spawn_agent(task: str, agent_id: str = "") -> dict:
        """Spawn an A2A subagent for a task using NDP routing."""
        spawner = A2ASubagentSpawner(slicer=slicer, twin=twin, mode="in-process")
        result = await spawner.spawn(task, agent_id=agent_id or None)
        from dataclasses import asdict
        return asdict(result)

    print("  ✓ Digital Twin tools registered (get_twin_state, spawn_agent, list_agents)")

    # 5. Airtable tool
    @mcp.tool()
    async def sync_tasks() -> dict:
        """Sync Airtable tasks into the Digital Twin."""
        await sync_airtable()
        return twin.get_summary()

    print("  ✓ Airtable sync tool registered")
    print()
    print("  Tools available:")
    print("    read_file       write_file      search_repo     run_tests")
    print("    git_commit      list_directory  get_repo_status search_web")
    print("    get_twin_state  get_twin_tasks  spawn_agent     list_agents")
    print("    sync_tasks")
    print()

    # ── Start server ──────────────────────────────────────────────────────────
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8080"))

    if transport == "http":
        print(f"  Starting HTTP server on {host}:{port}")
        mcp.run(transport="http", host=host, port=port)
    else:
        print("  Starting stdio server (use MCP_TRANSPORT=http for HTTP mode)")
        mcp.run(transport="stdio")


def main() -> None:
    parser = argparse.ArgumentParser(description="A2A Digital Twin Bootstrap")
    parser.add_argument("--build-rag", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if store exists")
    parser.add_argument("--sync-airtable", action="store_true")
    parser.add_argument("--test-tools", action="store_true")
    args = parser.parse_args()

    if args.build_rag:
        asyncio.run(build_rag(force=args.force))
    elif args.sync_airtable:
        asyncio.run(sync_airtable())
    elif args.test_tools:
        asyncio.run(test_tools())
    else:
        asyncio.run(start_mcp_server())


if __name__ == "__main__":
    main()
