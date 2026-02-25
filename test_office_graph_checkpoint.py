"""
tests/test_office_graph_checkpoint.py

Tests for the Office 365 Graph API integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import os

# Import from the parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.office.graph_checkpoint import (
    write_stage_checkpoint,
    CheckpointResult,
    _get_token
)


class TestOfficeGraphCheckpoint:

    def test_get_token_missing_creds(self):
        """Should return None if env vars are missing."""
        # Ensure env vars are cleared for this test
        with patch.dict(os.environ, {}, clear=True):
            token = _get_token()
            assert token is None

    def test_get_token_success(self):
        """Should return token if creds exist and request succeeds."""
        env_vars = {
            "OFFICE_TENANT_ID": "tenant",
            "OFFICE_CLIENT_ID": "client",
            "OFFICE_CLIENT_SECRET": "secret"
        }
        with patch.dict(os.environ, env_vars):
            # Patch the module-level httpx to ensure it's treated as present
            with patch("integrations.office.graph_checkpoint.httpx") as mock_httpx:
                # Mock the response
                mock_resp = MagicMock()
                mock_resp.json.return_value = {"access_token": "fake_token"}
                mock_resp.status_code = 200
                mock_httpx.post.return_value = mock_resp

                token = _get_token()
                assert token == "fake_token"
                mock_httpx.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_checkpoint_offline(self):
        """Should return mock result if offline (no token)."""
        # Mock _get_token to return None
        with patch("integrations.office.graph_checkpoint._get_token", return_value=None):
            result = await write_stage_checkpoint(
                task_name="Task",
                agent_id="Agent",
                stage="Stage",
                summary="Summary"
            )
            assert result["mock"] is True
            assert "offline" in result["word_doc_url"]

    @pytest.mark.asyncio
    async def test_write_checkpoint_live(self):
        """Should make Graph API calls if token is present."""
        with patch("integrations.office.graph_checkpoint._get_token", return_value="valid_token"):
            with patch.dict(os.environ, {"OFFICE_USER_EMAIL": "user@example.com"}):
                # Mock httpx.AsyncClient
                with patch("integrations.office.graph_checkpoint.httpx.AsyncClient") as mock_client_cls:
                    mock_client = AsyncMock()
                    mock_client_cls.return_value.__aenter__.return_value = mock_client
                    
                    # Mock responses for Word (PUT) and Excel/Outlook (POST)
                    mock_word_resp = MagicMock()
                    mock_word_resp.status_code = 201
                    mock_word_resp.json.return_value = {"webUrl": "http://word/doc"}
                    mock_client.put.return_value = mock_word_resp

                    mock_post_resp = MagicMock()
                    mock_post_resp.status_code = 202
                    mock_client.post.return_value = mock_post_resp

                    result = await write_stage_checkpoint(
                        task_name="Task",
                        agent_id="Agent",
                        stage="Stage",
                        summary="Summary",
                        metrics={"metric": 1},
                        handoff_email="boss@example.com",
                        checkpoint_type="all"
                    )

                    assert result["mock"] is False
                    assert result["word_doc_url"] == "http://word/doc"
                    assert result["email_sent"] is True
                    
                    # Verify calls: 1 PUT (Word), 2 POSTs (Excel + Outlook)
                    assert mock_client.put.call_count == 1
                    assert mock_client.post.call_count == 2
