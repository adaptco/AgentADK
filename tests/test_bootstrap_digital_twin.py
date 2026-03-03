"""Tests for bootstrap_digital_twin CLI dispatch behavior."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import bootstrap_digital_twin


@pytest.fixture()
def mocks():
    """Patch async entrypoints and asyncio.run to validate main() routing."""
    with patch.object(bootstrap_digital_twin, "build_rag", new_callable=MagicMock) as build_rag, \
        patch.object(bootstrap_digital_twin, "sync_airtable", new_callable=MagicMock) as sync_airtable, \
        patch.object(bootstrap_digital_twin, "test_tools", new_callable=MagicMock) as test_tools, \
        patch.object(bootstrap_digital_twin, "start_mcp_server", new_callable=MagicMock) as start_mcp_server, \
        patch.object(bootstrap_digital_twin.asyncio, "run", new_callable=MagicMock) as run:
        yield {
            "build_rag": build_rag,
            "sync_airtable": sync_airtable,
            "test_tools": test_tools,
            "start_mcp_server": start_mcp_server,
            "run": run,
        }


class TestMain:
    def test_default_starts_server(self, mocks):
        with patch.object(sys, "argv", ["bootstrap_digital_twin.py"]):
            bootstrap_digital_twin.main()

        mocks["start_mcp_server"].assert_called_once_with()
        mocks["build_rag"].assert_not_called()
        mocks["sync_airtable"].assert_not_called()
        mocks["test_tools"].assert_not_called()
        mocks["run"].assert_called_once_with(mocks["start_mcp_server"].return_value)

    def test_build_rag_argument(self, mocks):
        with patch.object(sys, "argv", ["bootstrap_digital_twin.py", "--build-rag"]):
            bootstrap_digital_twin.main()

        mocks["build_rag"].assert_called_once_with(force=False)
        mocks["sync_airtable"].assert_not_called()
        mocks["test_tools"].assert_not_called()
        mocks["start_mcp_server"].assert_not_called()
        mocks["run"].assert_called_once_with(mocks["build_rag"].return_value)

    def test_build_rag_with_force(self, mocks):
        with patch.object(
            sys, "argv", ["bootstrap_digital_twin.py", "--build-rag", "--force"]
        ):
            bootstrap_digital_twin.main()

        mocks["build_rag"].assert_called_once_with(force=True)
        mocks["sync_airtable"].assert_not_called()
        mocks["test_tools"].assert_not_called()
        mocks["start_mcp_server"].assert_not_called()
        mocks["run"].assert_called_once_with(mocks["build_rag"].return_value)

    def test_sync_airtable_argument(self, mocks):
        with patch.object(sys, "argv", ["bootstrap_digital_twin.py", "--sync-airtable"]):
            bootstrap_digital_twin.main()

        mocks["sync_airtable"].assert_called_once_with()
        mocks["build_rag"].assert_not_called()
        mocks["test_tools"].assert_not_called()
        mocks["start_mcp_server"].assert_not_called()
        mocks["run"].assert_called_once_with(mocks["sync_airtable"].return_value)

    def test_test_tools_argument(self, mocks):
        with patch.object(sys, "argv", ["bootstrap_digital_twin.py", "--test-tools"]):
            bootstrap_digital_twin.main()

        mocks["test_tools"].assert_called_once_with()
        mocks["build_rag"].assert_not_called()
        mocks["sync_airtable"].assert_not_called()
        mocks["start_mcp_server"].assert_not_called()
        mocks["run"].assert_called_once_with(mocks["test_tools"].return_value)

