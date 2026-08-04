"""The SSE / streamable-http path must build on BOTH SDK lines.

WHY THIS FILE EXISTS
--------------------
v0.12.0 shipped dual mcp 1.x/2.x support with a 12-leg CI matrix — and still
broke every NETWORK deployment on 2.x, because every e2e case drives *stdio*.
The network path was never executed on either line.

The defect: 1.x configures the server by MUTATING `server.settings.host`,
`server.settings.transport_security.*`. On 2.x `Settings` has no such fields,
so the assignment raises

    ValueError: "Settings" object has no field "host"

i.e. the server does not start at all. Silent to a stdio suite; fatal to the
actual deployments, which run SSE behind bearer auth.

These tests build the ASGI app the way `_run_network` does, on whichever line
is installed. Both CI legs run them, so the pair is covered.
"""
from __future__ import annotations

import pytest

from yantrikdb_mcp._compat import MCP_MAJOR, build_network_app, sdk_line


@pytest.mark.parametrize("transport", ["sse", "streamable-http"])
def test_network_app_builds(transport: str) -> None:
    """The exact call `_run_network` makes. On 2.x this raised ValueError
    before v0.12.1 and the process died before uvicorn ever bound."""
    from yantrikdb_mcp.server import mcp

    app = build_network_app(mcp, transport, "0.0.0.0", 8420)
    assert app is not None, f"{transport} app must build on {sdk_line()}"
    # Starlette ASGI apps are callable with (scope, receive, send).
    assert callable(app), f"{transport} app must be an ASGI callable"


def test_run_network_uses_the_compat_builder() -> None:
    """Guard the regression at its source: `_run_network` must not go back to
    poking `mcp.settings.host` directly, which only works on 1.x."""
    import inspect

    from yantrikdb_mcp import _run_network

    src = inspect.getsource(_run_network)
    assert "build_network_app" in src, (
        "_run_network must build its ASGI app through _compat.build_network_app"
    )
    for forbidden in ("settings.host", "settings.port", "settings.transport_security"):
        assert forbidden not in src, (
            f"_run_network touches `{forbidden}` directly — that is 1.x-only "
            f"state and raises ValueError on mcp 2.x. Route it through "
            f"_compat.build_network_app instead."
        )


def test_transport_security_is_permissive_by_design() -> None:
    """This server is reached over LAN/tunnel and gated by bearer-token auth,
    not host/origin allowlists. Pin that intent so a future refactor doesn't
    silently start rejecting the deployed clients."""
    from mcp.server.transport_security import TransportSecuritySettings

    fields = set(TransportSecuritySettings.model_fields)
    # Identical on both majors — only where you attach it changed.
    assert {"allowed_hosts", "allowed_origins",
            "enable_dns_rebinding_protection"} <= fields


@pytest.mark.skipif(MCP_MAJOR != 2, reason="2.x-only shape check")
def test_2x_settings_really_lacks_host() -> None:
    """Documents WHY the shim exists: assigning host on 2.x is an error, not
    a no-op. If a future 2.x restores the field this test fails loudly and the
    shim can be simplified — a deliberate tripwire, not a bug."""
    from yantrikdb_mcp.server import mcp

    with pytest.raises((ValueError, AttributeError)):
        mcp.settings.host = "0.0.0.0"
