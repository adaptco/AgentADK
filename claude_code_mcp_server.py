"""
mcp_extensions/claude_code_mcp_server.py

Drop-in Claude Code replacement MCP server for A2A_MCP.
Extends the existing mcp_server.py with:
  - Normalized dot product vector search (not cosine — faster, equivalent when vectors are L2-normalised)
  - Git operations as MCP tools
  - Test runner as MCP tool
  - Repo RAG as MCP tool

Drop this next to your existing mcp_server.py and import it:
    from mcp_extensions.claude_code_mcp_server import register_claude_code_tools

Then call register_claude_code_tools(mcp) inside your FastMCP setup.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastmcp import FastMCP

# ── Repo root is always the A2A_MCP checkout ────────────────────────────────
REPO_ROOT = Path(os.environ.get("A2A_REPO_ROOT", Path(__file__).parent.parent))

# ── Shared embedding store (populated by rag/vertical_tensor_slice.py) ──────
_EMBEDDING_STORE: dict[str, np.ndarray] = {}   # path → L2-normalised d=1536 vector
_CHUNK_TEXT: dict[str, str] = {}                # path → raw text chunk


def load_embedding_store(store_path: str | Path = "rag/embedding_store.npz") -> None:
    """Load pre-built embedding store into memory."""
    global _EMBEDDING_STORE, _CHUNK_TEXT
    p = Path(store_path)
    if not p.exists():
        return
    data = np.load(p, allow_pickle=True)
    _EMBEDDING_STORE = {k: data["vectors"][i] for i, k in enumerate(data["keys"])}
    _CHUNK_TEXT = dict(zip(data["keys"], data["texts"]))


# ── Normalized dot product retrieval ────────────────────────────────────────
def ndp_search(query_vector: np.ndarray, top_k: int = 8) -> list[dict[str, Any]]:
    """
    Normalized Dot Product search.

    When both query and corpus vectors are L2-normalised,
    dot(q, v) == cosine_similarity(q, v).
    Using np.dot is ~3x faster than computing cosine explicitly.

    Returns list of {path, score, text} sorted descending by score.
    """
    if not _EMBEDDING_STORE:
        return []

    keys = list(_EMBEDDING_STORE.keys())
    matrix = np.stack([_EMBEDDING_STORE[k] for k in keys])  # (N, 1536)

    # Single matrix multiply — O(N·d), no sqrt needed
    scores: np.ndarray = matrix @ query_vector                # (N,)

    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    return [
        {
            "path": keys[i],
            "score": float(scores[i]),
            "text": _CHUNK_TEXT.get(keys[i], ""),
        }
        for i in top_idx
    ]


def get_embedding(text: str) -> np.ndarray:
    """
    Embed text using OpenAI text-embedding-3-small (d=1536) and L2-normalise.
    Falls back to a deterministic hash embedding for offline/test use.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
    except Exception:
        # Offline fallback: SHA-256 hash → 1536 floats (deterministic, not semantic)
        import hashlib
        raw = hashlib.sha256(text.encode()).digest() * (1536 // 32 + 1)
        vec = np.frombuffer(raw[:1536 * 4], dtype=np.float32).copy()

    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── Tool registration ─────────────────────────────────────────────────────────
def register_claude_code_tools(mcp: FastMCP) -> None:
    """
    Register all Claude Code replacement tools on a FastMCP instance.
    Call this inside your existing mcp_server.py after creating the FastMCP app.
    """

    # ── 1. read_file ─────────────────────────────────────────────────────────
    @mcp.tool()
    async def read_file(path: str, start_line: int = 0, end_line: int = -1) -> dict:
        """Read a file from the repo. Optionally specify line range."""
        full_path = REPO_ROOT / path
        if not full_path.exists():
            return {"ok": False, "error": f"File not found: {path}"}
        lines = full_path.read_text(encoding="utf-8").splitlines(keepends=True)
        if end_line == -1:
            end_line = len(lines)
        content = "".join(lines[start_line:end_line])
        return {"ok": True, "path": path, "content": content, "total_lines": len(lines)}

    # ── 2. write_file ────────────────────────────────────────────────────────
    @mcp.tool()
    async def write_file(path: str, content: str, create_dirs: bool = True) -> dict:
        """Write content to a file. Creates parent directories if needed."""
        full_path = REPO_ROOT / path
        if create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path, "bytes_written": len(content.encode())}

    # ── 3. search_repo (NDP-powered) ─────────────────────────────────────────
    @mcp.tool()
    async def search_repo(query: str, top_k: int = 8) -> dict:
        """
        Semantic search over the repo using Normalized Dot Product.
        Embeds the query, searches L2-normalised corpus vectors.
        Returns top_k most relevant file chunks with similarity scores.
        """
        query_vec = get_embedding(query)
        results = ndp_search(query_vec, top_k=top_k)
        if not results:
            return {
                "ok": True,
                "results": [],
                "note": "Embedding store empty — run: python rag/vertical_tensor_slice.py",
            }
        return {"ok": True, "query": query, "results": results}

    # ── 4. run_tests ─────────────────────────────────────────────────────────
    @mcp.tool()
    async def run_tests(
        pattern: str = "",
        package: str = "",
        timeout: int = 120,
    ) -> dict:
        """
        Run pytest on the repo. Equivalent to: pytest tests/ -v [pattern].
        Returns pass/fail counts and any failure output.
        """
        report_path = Path(tempfile.gettempdir()) / "a2a_test_report.json"
        
        cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "--json-report",
               f"--json-report-file={report_path}"]
        if pattern:
            cmd.extend(["-k", pattern])
        if package:
            cmd = ["python", "-m", "pytest", package, "-v", "--tb=short"]

        try:
            result = subprocess.run(
                cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout
            )
            report = {}
            try:
                with open(report_path) as f:
                    report = json.load(f)
            except Exception:
                pass

            return {
                "ok": result.returncode == 0,
                "return_code": result.returncode,
                "passed": report.get("summary", {}).get("passed", 0),
                "failed": report.get("summary", {}).get("failed", 0),
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-2000:],
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Tests timed out after {timeout}s"}

    # ── 5. git_commit ─────────────────────────────────────────────────────────
    @mcp.tool()
    async def git_commit(message: str, files: list[str] | None = None, push: bool = False) -> dict:
        """
        Stage files (or all changes) and create a git commit.
        Optionally push to origin/main.
        """
        try:
            if files:
                subprocess.run(["git", "add"] + files, cwd=str(REPO_ROOT), check=True)
            else:
                subprocess.run(["git", "add", "-A"], cwd=str(REPO_ROOT), check=True)

            result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=str(REPO_ROOT), capture_output=True, text=True
            )
            if result.returncode != 0 and "nothing to commit" not in result.stdout:
                return {"ok": False, "error": result.stderr}

            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO_ROOT)
            ).decode().strip()

            pushed = False
            if push:
                push_result = subprocess.run(
                    ["git", "push", "origin", "HEAD"],
                    cwd=str(REPO_ROOT), capture_output=True, text=True
                )
                pushed = push_result.returncode == 0

            return {"ok": True, "sha": sha, "message": message, "pushed": pushed}
        except subprocess.CalledProcessError as e:
            return {"ok": False, "error": str(e)}

    # ── 6. list_directory ────────────────────────────────────────────────────
    @mcp.tool()
    async def list_directory(path: str = ".", depth: int = 2) -> dict:
        """List directory structure up to given depth."""
        full_path = REPO_ROOT / path
        if not full_path.exists():
            return {"ok": False, "error": f"Path not found: {path}"}

        def _walk(p: Path, d: int) -> dict:
            if d == 0 or not p.is_dir():
                return str(p.relative_to(REPO_ROOT))
            return {
                p.name: [_walk(c, d - 1) for c in sorted(p.iterdir())
                         if not c.name.startswith(".") and c.name != "__pycache__"]
            }

        return {"ok": True, "tree": _walk(full_path, depth)}

    # ── 7. get_repo_status ───────────────────────────────────────────────────
    @mcp.tool()
    async def get_repo_status() -> dict:
        """Get current git status, branch, and last 5 commits."""
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(REPO_ROOT)
            ).decode().strip()
            status = subprocess.check_output(
                ["git", "status", "--short"], cwd=str(REPO_ROOT)
            ).decode().strip()
            log = subprocess.check_output(
                ["git", "log", "--oneline", "-5"], cwd=str(REPO_ROOT)
            ).decode().strip()
            return {"ok": True, "branch": branch, "status": status, "recent_commits": log}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ── Standalone entry point (for testing this MCP server directly) ────────────
if __name__ == "__main__":
    import os
    mcp = FastMCP("claude-code-replacement")
    register_claude_code_tools(mcp)
    load_embedding_store()
    print("Claude Code MCP server running on stdio")
    mcp.run(transport="stdio")
