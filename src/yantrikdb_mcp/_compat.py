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

__all__ = ["MCPServer", "Context", "ToolError", "ToolAnnotations", "MCP_MAJOR"]


def sdk_line() -> str:
    """Human-readable SDK line, for --help / diagnostics."""
    return f"mcp {MCP_MAJOR}.x"
