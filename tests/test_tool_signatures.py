"""Startup validation for MCP wrapper/engine signature drift."""

from types import ModuleType

import pytest

from yantrikdb_mcp.server import _validate_tool_signatures


def deliberately_mismatched_wrapper(ctx=None):
    db = ctx
    return db.correct("rid", correction_note="outdated wrapper")


class _Engine:
    def correct(self, rid, reason):
        pass


def test_signature_check_names_tool_and_forwarded_parameter(caplog):
    tools = ModuleType("deliberately_mismatched_tools")
    tools.deliberately_mismatched_wrapper = deliberately_mismatched_wrapper

    with pytest.raises(RuntimeError, match="1 mismatch"):
        _validate_tool_signatures(_Engine, tools)

    assert "tool 'deliberately_mismatched_wrapper'" in caplog.text
    assert "parameter 'correction_note'" in caplog.text
    assert "YantrikDB.correct()" in caplog.text
