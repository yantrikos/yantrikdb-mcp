"""MCP SDK compatibility layer — supports both the 1.x and 2.x SDK lines.

WHY THIS EXISTS
---------------
mcp 2.0.0 removed `mcp.server.fastmcp`. Every symbol this package imports at
module scope moved, so `import yantrikdb_mcp` raised ModuleNotFoundError on
the new SDK — and because the dependency was pinned `>=1.2.0` with no
ceiling, already-published releases silently re-shipped themselves broken the
day 2.0.0 landed (see tests/test_dependency_pins.py).

The migration turned out to be a SYMBOL-LOCATION change, not a semantic one.
Verified empirically against mcp 2.0.0 before this module was written:

  - `MCPServer(name, instructions=..., lifespan=...)` accepts the same
    arguments we pass to `FastMCP`
  - `@server.tool(annotations=ToolAnnotations(...))` keeps the same shape,
    and the annotations survive to the wire in tools/list
  - a `lifespan` async-context-manager yielding a dict is still reachable as
    `ctx.request_context.lifespan_context`
  - `ToolError` still renders as `isError: true` on the result envelope
  - `mcp.types.ToolAnnotations` did not move at all
  - `.run(transport="stdio"|"sse"|"streamable-http")` is unchanged

Because the semantics match, ONE codebase can serve both lines: this module
resolves the symbols and the rest of the package imports them from here.
Nothing else in the package may import from `mcp.server.*` directly — the
dependency-pin suite asserts that, so a future contributor can't reintroduce
a version-specific import path that only fails on the other line.

WHICH LINE AM I ON?
-------------------
`MCP_MAJOR` is 1 or 2, resolved by probing the import, never by parsing a
version string. That is the same discipline the engine contract suites use:
two builds can report the same version and behave differently, so the probe
is the truth and the version is a backstop.
"""
from __future__ import annotations

# ToolAnnotations lives in mcp.types on BOTH lines — no shim needed, but it's
# re-exported here so callers have exactly one import site for MCP symbols.
from mcp.types import ToolAnnotations

try:
    # ── mcp 2.x ──
    from mcp.server import MCPServer as _ServerClass
    from mcp.server.mcpserver import Context
    from mcp.server.mcpserver.exceptions import ToolError

    MCP_MAJOR = 2
except ImportError:  # pragma: no cover - exercised on the 1.x CI leg
    # ── mcp 1.x ──
    # FastMCP is the 1.x name for the same object; alias it so the rest of
    # the package speaks one vocabulary regardless of the installed line.
    from mcp.server.fastmcp import Context, FastMCP as _ServerClass
    from mcp.server.fastmcp.exceptions import ToolError

    MCP_MAJOR = 1

# The canonical name used throughout this package. On 1.x this IS FastMCP;
# on 2.x it IS MCPServer. Callers must not care which.
MCPServer = _ServerClass

__all__ = [
    "MCPServer",
    "Context",
    "ToolError",
    "ToolAnnotations",
    "MCP_MAJOR",
    "build_network_app",
    "sdk_line",
]


def sdk_line() -> str:
    """Human-readable SDK line, for --help / diagnostics."""
    return f"mcp {MCP_MAJOR}.x"


def build_network_app(server, transport: str, host: str, port: int):
    """Build the ASGI app for `sse` / `streamable-http` on either SDK line.

    THE SECOND REAL DIVERGENCE between the majors, and the one that actually
    bites deployments rather than tests:

      1.x  configuration is MUTABLE STATE on the server —
             server.settings.host = ...
             server.settings.transport_security.allowed_hosts = [...]
           then `server.sse_app()` reads it back.

      2.x  `Settings` has NO host/port/transport_security fields at all;
           assigning one raises `ValueError: "Settings" object has no field
           "host"`. Host and security are ARGUMENTS to the app factory:
             server.sse_app(host=..., transport_security=...)

    So the 1.x code path doesn't merely misconfigure on 2.x — it raises, and
    the server never starts. That is invisible to a stdio test suite, which is
    exactly how it shipped in v0.12.0: every e2e case drives stdio, while the
    deployed SSE servers take this path. Fixed in v0.12.1, with
    `tests/test_network_transport_compat.py` covering both lines.

    TransportSecuritySettings itself is identical on both majors — only the
    place you attach it moved.

    Binding note: uvicorn is what actually binds host:port (see
    `_run_network`). On 1.x the settings assignment additionally feeds the
    SDK's own URL construction, so it is still set there for fidelity.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    # This server is deliberately permissive: it is normally reached over a
    # LAN/tunnel and fronted by bearer-token auth (see auth.BearerTokenMiddleware),
    # not by host/origin allowlisting.
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=["*"],
        allowed_origins=["*"],
    )

    if MCP_MAJOR == 1:
        server.settings.host = host
        server.settings.port = port
        server.settings.transport_security = security
        return server.sse_app() if transport == "sse" else server.streamable_http_app()

    # 2.x — pass configuration in rather than mutating settings.
    if transport == "sse":
        return server.sse_app(host=host, transport_security=security)
    return server.streamable_http_app(host=host, transport_security=security)
