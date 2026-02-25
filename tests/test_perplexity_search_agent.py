"""
tests/test_perplexity_search_agent.py

Tests for the Perplexity Search Agent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

# Import from the parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.perplexity.search_agent import PerplexitySearchAgent


class TestPerplexitySearchAgent:
    
    @pytest.fixture
    def mock_slicer(self):
        """Mock the VerticalTensorSlicer."""
        slicer = MagicMock()
        slicer.query.return_value = []
        return slicer

    def test_init_offline(self):
        """Should be offline without API key."""
        agent = PerplexitySearchAgent(api_key=None)
        assert not agent.is_live

    def test_init_live(self):
        """Should be live with API key (assuming httpx installed)."""
        agent = PerplexitySearchAgent(api_key="test_key")
        try:
            import httpx
            assert agent.is_live
        except ImportError:
            assert not agent.is_live

    @pytest.mark.asyncio
    async def test_search_local_fallback(self, mock_slicer):
        """Should return local results if slicer provided and score is high."""
        # Mock high score local result
        mock_slicer.query.return_value = [{"score": 0.9, "text": "local info"}]
        
        agent = PerplexitySearchAgent(api_key=None, slicer=mock_slicer)
        result = await agent.search("query")
        
        assert result["source"] == "local_rag"
        assert len(result["local_results"]) == 1
        # Should not have tried web search (offline anyway, but logic holds)

    @pytest.mark.asyncio
    async def test_search_web_call(self):
        """Should call Perplexity API if live and local results are weak."""
        # Patch httpx to simulate web request
        with patch("integrations.perplexity.search_agent.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            
            # Mock response
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "Web result"}}]
            }
            mock_client.post.return_value = mock_resp

            agent = PerplexitySearchAgent(api_key="key")
            result = await agent.search("query")
            
            assert result["source"] == "perplexity"
            assert result["web_results"][0]["text"] == "Web result"

    @pytest.mark.asyncio
    async def test_search_offline_fallback(self):
        """Should return offline status if no key and no local results."""
        agent = PerplexitySearchAgent(api_key=None)
        result = await agent.search("query")
        
        assert result["source"] == "offline"