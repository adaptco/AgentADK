"""
tests/test_airtable_client.py

Tests for the Airtable integration client.
"""

from __future__ import annotations

import pytest

# Import from the parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.airtable.task_schema import (
    AirtableClient, 
    TaskStatus, 
    AirtableTask
)


class TestAirtableClient:
    
    def test_init_offline_by_default(self):
        """Client should be offline if no API key is provided."""
        client = AirtableClient(api_key=None, base_id=None)
        assert not client.is_live

    def test_init_live_requires_creds(self):
        """Client should be live if creds are provided (and httpx installed)."""
        # We assume httpx is installed in the test env
        client = AirtableClient(api_key="key", base_id="base")
        # If httpx is present, it should be live
        try:
            import httpx
            assert client.is_live
        except ImportError:
            assert not client.is_live

    @pytest.mark.asyncio
    async def test_list_tasks_offline(self):
        """Offline mode should return mock tasks."""
        client = AirtableClient(api_key=None)
        tasks = await client.list_tasks()
        
        assert len(tasks) > 0
        assert isinstance(tasks[0], AirtableTask)
        assert tasks[0].record_id.startswith("recMOCK")

    @pytest.mark.asyncio
    async def test_update_task_status_offline(self):
        """Offline mode should return mock success response."""
        client = AirtableClient(api_key=None)
        result = await client.update_task_status("rec123", TaskStatus.DONE)
        
        assert result["mock"] is True
        assert result["status"] == "Done"

    @pytest.mark.asyncio
    async def test_list_roles_offline(self):
        """Offline mode should return mock roles."""
        client = AirtableClient(api_key=None)
        roles = await client.list_roles()
        
        assert len(roles) > 0
        assert any(r.name == "Coder" for r in roles)