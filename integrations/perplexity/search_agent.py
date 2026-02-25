"""
integrations/perplexity/search_agent.py

Perplexity AI search agent for the A2A Digital Twin.
Provides web knowledge fallback when the local RAG store
doesn't have sufficient context for a query.

Falls back gracefully when PERPLEXITY_API_KEY is not set.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


_PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"


class PerplexitySearchAgent:
    """
    Web search agent using the Perplexity API.

    Optionally integrates with a VerticalTensorSlicer to first check
    the local embedding store before falling back to web search.
    """

    def __init__(
        self,
        api_key: str | None = None,
        slicer: Any | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self._slicer = slicer
        self._is_live = bool(self._api_key and httpx)

    @property
    def is_live(self) -> bool:
        """True if we have a valid API key and httpx is available."""
        return self._is_live

    async def search(
        self,
        query: str,
        agent_filter: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Search for information.

        1. First searches the local RAG store (if slicer is available)
        2. Falls back to Perplexity web search if local results are weak
        """
        result: dict[str, Any] = {
            "query": query,
            "local_results": [],
            "web_results": [],
            "source": "none",
        }

        # Step 1: Local RAG search
        if self._slicer is not None:
            try:
                local = self._slicer.query(
                    query, top_k=top_k, agent_filter=agent_filter or None
                )
                result["local_results"] = local
                # If top local score is high enough, skip web search
                if local and local[0].get("score", 0) > 0.75:
                    result["source"] = "local_rag"
                    return result
            except Exception:
                pass

        # Step 2: Web search via Perplexity
        if self._is_live:
            try:
                web = await self._perplexity_search(query)
                result["web_results"] = web
                result["source"] = "perplexity"
            except Exception as e:
                result["web_error"] = str(e)
                result["source"] = "local_rag" if result["local_results"] else "none"
        else:
            result["source"] = "local_rag" if result["local_results"] else "offline"

        return result

    async def _perplexity_search(self, query: str) -> list[dict[str, str]]:
        """Call the Perplexity API for web search."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Provide concise, "
                        "factual answers with source URLs."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 1024,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                _PERPLEXITY_API_URL, headers=headers, json=payload
            )
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return []

        content = choices[0].get("message", {}).get("content", "")
        return [{"text": content, "source": "perplexity"}]

    async def _tool_fn(
        self, query: str, agent_filter: str = ""
    ) -> dict[str, Any]:
        """MCP tool entry point — called by bootstrap_digital_twin.py."""
        return await self.search(query, agent_filter=agent_filter)
