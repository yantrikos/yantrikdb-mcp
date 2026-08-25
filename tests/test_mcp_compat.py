"""`_compat` must resolve to the SDK line that is ACTUALLY installed.

A compat shim that imports successfully but silently picked the wrong branch
is worse than no shim: every downstream symbol would still resolve, and the
mismatch would only surface as strange runtime behaviour on one of the two
lines. So these tests cross-check the shim's verdict against an independent
probe of the environment, rather than trusting `MCP_MAJOR` on its own.

The independent probe is the INSTALLED DISTRIBUTION's major version
(`importlib.metadata.version("mcp")`), not the presence of a module path.
Module presence stopped being a discriminator at mcp 2.1: 2.0.0 removed
`mcp.server.fastmcp`, and 2.1.x ships it again alongside the canonical
`mcp.server.mcpserver`. A probe that read "fastmcp importable" as "1.x"
failed on 2.1.1 while the shim (which imports the canonical 2.x names first)
was right. Presence of `mcp.server.fastmcp` therefore proves nothing about
the line; the canonical 2.x module is what distinguishes 2.x, and the
distribution version is what the environment actually says.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util

import pytest

from yantrikdb_mcp._compat import (
    MCP_MAJOR,
    Context,
    MCPServer,
    ToolAnnotations,
    ToolError,
    sdk_line,
)


def _installed_major() -> int:
    """Independent probe: the major of the `mcp` distribution that pip
    installed — read from package metadata, not from which modules import."""
    return int(importlib.metadata.version("mcp").split(".", 1)[0])


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _canonical_2x_server_present() -> bool:
    """The 2.x-only module tree (`mcp.server.mcpserver`)."""
    return _module_present("mcp.server.mcpserver")


def _fastmcp_present() -> bool:
    """The 1.x module tree — ALSO present on 2.1.x, so never a discriminator
    on its own. Kept only to describe the environment in assertion messages
    and to pin the 2.1.x shape below."""
    return _module_present("mcp.server.fastmcp")


def test_major_matches_an_independent_probe() -> None:
    """The shim's verdict must agree with the installed distribution, not just
    be self-consistent."""
    expected = _installed_major()
    assert expected in (1, 2), f"unexpected installed mcp major: {expected}"
    assert MCP_MAJOR == expected, (
        f"_compat resolved {sdk_line()} but the installed mcp distribution is "
        f"{importlib.metadata.version('mcp')} — the shim picked the wrong branch."
    )


def test_canonical_server_module_agrees_with_the_major() -> None:
    """On 2.x the canonical `mcp.server.mcpserver` tree must exist (that is
    what the shim imports first); on 1.x it must not. This is the module-level
    cross-check that survives 2.1.x — unlike fastmcp presence."""
    assert _canonical_2x_server_present() == (MCP_MAJOR == 2), (
        f"{sdk_line()} resolved, but mcp.server.mcpserver "
        f"{'IS' if _canonical_2x_server_present() else 'is NOT'} importable"
    )


@pytest.mark.skipif(
    not (_fastmcp_present() and _canonical_2x_server_present()),
    reason="regression shape needs BOTH module trees importable (mcp 2.1.x)",
)
def test_both_module_trees_importable_resolves_to_2x() -> None:
    """Regression for mcp 2.1.1: `mcp.server.fastmcp` AND
    `mcp.server.mcpserver` are both importable. The shim must choose 2.x
    (canonical names first) and the tests must not infer 1.x from the
    reappearance of the fastmcp path."""
    assert MCP_MAJOR == 2, (
        f"both module trees present (mcp {importlib.metadata.version('mcp')}) "
        f"but the shim resolved {sdk_line()}"
    )
    assert _installed_major() == 2


def test_all_exported_symbols_are_usable() -> None:
    assert MCP_MAJOR in (1, 2)
    assert callable(MCPServer), "MCPServer must be constructible"
    assert issubclass(ToolError, Exception)
    assert Context is not None
    # ToolAnnotations is a pydantic model on both lines; the server passes
    # these exact fields, so a rename here would break every tool's metadata.
    for field in ("title", "readOnlyHint", "destructiveHint",
                  "idempotentHint", "openWorldHint"):
        ToolAnnotations(**{field: True} if field != "title" else {"title": "x"})


def test_server_object_is_the_line_appropriate_class() -> None:
    """On 1.x this must be FastMCP; on 2.x it must be MCPServer. Guards
    against the shim aliasing something plausible-but-wrong."""
    if MCP_MAJOR == 1:
        from mcp.server.fastmcp import FastMCP

        assert MCPServer is FastMCP
    else:
        from mcp.server import MCPServer as Real

        assert MCPServer is Real


def test_the_live_server_object_was_built_from_the_shim() -> None:
    """End-to-end: the module-level server really is an instance of the
    class the shim resolved."""
    from yantrikdb_mcp.server import mcp as live

    assert isinstance(live, MCPServer)


def test_static_resources_declare_no_context_parameter() -> None:
    """mcp 2.x REFUSES to register a static resource whose handler takes a
    Context, and the failure is at import time — it takes down the whole
    server, not just that resource.

    This is the divergence that the symbol-level shim could not paper over,
    so pin it: any static (non-templated) resource handler must be
    Context-free. Templated resources are exempt — 2.x allows Context there.
    """
    import inspect

    from yantrikdb_mcp import resources as res

    offenders = []
    for name, fn in vars(res).items():
        if not callable(fn) or name.startswith("_"):
            continue
        if not name.endswith("_resource"):
            continue
        params = inspect.signature(fn).parameters
        has_ctx = any(
            p.annotation is Context or getattr(p.annotation, "__name__", "") == "Context"
            for p in params.values()
        )
        # A handler with no non-ctx params is bound to a static URI here.
        templated = any(p.name != "ctx" for p in params.values())
        if has_ctx and not templated:
            offenders.append(name)
    assert not offenders, (
        f"static resource handlers must not declare Context — mcp 2.x refuses "
        f"to register them and the server dies at import: {offenders}"
    )


@pytest.mark.skipif(MCP_MAJOR != 2, reason="2.x-specific guard")
def test_no_static_resource_context_regression_on_2x() -> None:
    """On 2.x, importing the package at all proves static-resource
    registration succeeded — this is the canary for that whole class."""
    import yantrikdb_mcp.resources  # noqa: F401
