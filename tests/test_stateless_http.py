"""Stateless streamable-http — the configuration that survives a restart.

THE PROBLEM, measured rather than assumed:

Both SSE and STATEFUL streamable-http keep a session table in memory. Restart
the server and it is empty, so the client's session id is unknown:

    SSE                      -> HTTP 404  "Could not find session"
    streamable-http (2.x)    -> HTTP 404  -32600 "Session not found"

The server is behaving CORRECTLY in both cases — it genuinely has no such
session. The client is the half that cannot recover: most surface the failure
as an opaque JSON-RPC error (Claude Code reports `-32602 Invalid request
parameters`) and keep using the dead session instead of re-initialising. That
is why "restart your client after every server upgrade" became folklore.

Worth recording because it corrects a wrong diagnosis this project carried for
a while: the fix is NOT to make the server emit a clearer rejection. It already
emits a clear, correct one. The fix is to have no session to lose.

STATELESS MODE removes the session entirely — every request carries its own
context. Verified end to end (see the repro in this file's git history): with
YANTRIKDB_STATELESS_HTTP=1, `initialize` returns NO Mcp-Session-Id, and a
tools/list issued after the server has been killed and restarted returns
HTTP 200 with real data. No client restart, no reconnect logic.
"""
from __future__ import annotations

import os

import pytest

from yantrikdb_mcp._compat import MCP_MAJOR, build_network_app


@pytest.fixture
def clean_env():
    old = os.environ.get("YANTRIKDB_STATELESS_HTTP")
    yield
    if old is None:
        os.environ.pop("YANTRIKDB_STATELESS_HTTP", None)
    else:
        os.environ["YANTRIKDB_STATELESS_HTTP"] = old


def test_stateless_app_builds_on_this_sdk_line(clean_env):
    from yantrikdb_mcp.server import mcp

    os.environ["YANTRIKDB_STATELESS_HTTP"] = "1"
    app = build_network_app(mcp, "streamable-http", "0.0.0.0", 8420)
    assert app is not None and callable(app)


def test_stateless_is_off_unless_asked(clean_env):
    """Opt-in: stateless costs per-request re-initialisation and rules out
    features needing a durable session, so it must never switch on by
    accident."""
    from yantrikdb_mcp.server import mcp

    os.environ.pop("YANTRIKDB_STATELESS_HTTP", None)
    app = build_network_app(mcp, "streamable-http", "0.0.0.0", 8420)
    assert app is not None
    if MCP_MAJOR == 1:
        assert mcp.settings.stateless_http is False


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes"])
def test_truthy_spellings_all_enable_it(clean_env, val):
    from yantrikdb_mcp.server import mcp

    os.environ["YANTRIKDB_STATELESS_HTTP"] = val
    build_network_app(mcp, "streamable-http", "0.0.0.0", 8420)
    if MCP_MAJOR == 1:
        assert mcp.settings.stateless_http is True


def test_sse_plus_stateless_is_refused_not_ignored(clean_env):
    """SSE has no stateless mode. Silently ignoring the flag would leave an
    operator believing their restarts are survivable when they are not — the
    exact false-confidence this feature exists to remove."""
    from yantrikdb_mcp.server import mcp

    os.environ["YANTRIKDB_STATELESS_HTTP"] = "1"
    with pytest.raises(ValueError, match="streamable-http"):
        build_network_app(mcp, "sse", "0.0.0.0", 8420)
