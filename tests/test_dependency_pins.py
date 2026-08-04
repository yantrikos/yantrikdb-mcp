"""Runtime dependency pins must be UPPER-BOUNDED.

Written after mcp 2.0.0 shipped and removed `mcp.server.fastmcp`. The pin was
`mcp[cli]>=1.2.0` with no ceiling, so every fresh `pip install yantrikdb-mcp`
resolved to 2.0.0 and produced a server that died on import:

    ModuleNotFoundError: No module named 'mcp.server.fastmcp'

CI caught it on the v0.11.0 release PR, but it had ALREADY broken the
published v0.10.0 for new installs — nothing in this repo changed, upstream
simply cut a major and our release re-shipped itself broken.

An unbounded pin on a hard runtime dependency is a release that silently
re-ships itself every time upstream cuts a major. The engine pin already
carried this firewall (with a comment explaining the identical lesson from
the v0.9.0/v0.9.1 signature drift); this test enforces the rule across ALL
runtime deps so the lesson can't be applied to one dependency and forgotten
on the next.

Scope: install_requires only. Optional extras are opt-in — a user asking for
[onnx] has chosen that surface — so they are exempt.
"""
from __future__ import annotations

import pathlib
import re

import pytest

try:  # py3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.10
    import tomli as tomllib  # type: ignore

PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"


def _runtime_deps() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def _name_of(spec: str) -> str:
    # "mcp[cli]>=1.2.0,<2.0.0" -> "mcp"
    return re.split(r"[\[<>=!~;\s]", spec, maxsplit=1)[0]


# Deps deliberately left unbounded, each with a recorded reason. Adding a name
# here is a REVIEWED decision, not a way to silence the test.
_UNBOUNDED_ALLOWED: dict[str, str] = {
    # Stable, ubiquitous, and strong semver track record; a breaking 3.x would
    # break most of the Python ecosystem simultaneously and be caught instantly.
    "requests": "ecosystem-stable; breakage would be industry-wide and immediate",
    "click": "ecosystem-stable; breakage would be industry-wide and immediate",
}


@pytest.mark.parametrize("spec", _runtime_deps())
def test_runtime_dependency_has_upper_bound(spec: str) -> None:
    name = _name_of(spec)
    if name in _UNBOUNDED_ALLOWED:
        pytest.skip(f"{name}: {_UNBOUNDED_ALLOWED[name]}")
    assert "<" in spec, (
        f"runtime dependency {spec!r} has no upper bound. The day {name} cuts a "
        f"breaking major, every fresh `pip install yantrikdb-mcp` — including "
        f"ALREADY-PUBLISHED versions — resolves to it and ships broken. This is "
        f"exactly how mcp 2.0.0 broke v0.10.0 post-release. Bound it to the "
        f"tested major, or add it to _UNBOUNDED_ALLOWED with a reason."
    )


def test_mcp_pin_excludes_the_fastmcp_removal() -> None:
    """Specifically pin the regression: mcp 2.x removed mcp.server.fastmcp,
    which server.py imports at module scope."""
    spec = next((d for d in _runtime_deps() if _name_of(d) == "mcp"), None)
    assert spec is not None, "mcp is a hard runtime dependency and must be declared"
    assert "<2.0.0" in spec.replace(" ", ""), (
        f"mcp pin {spec!r} must exclude 2.x until the FastMCP-2 import surface "
        f"is ported and contract-tested — 2.0.0 removed `mcp.server.fastmcp`."
    )


def test_the_imports_the_server_actually_needs_are_importable() -> None:
    """The rejection surface for this whole class of failure: not 'is a version
    installed' but 'do the symbols server.py imports at module scope exist'."""
    from mcp.server.fastmcp import FastMCP  # noqa: F401
    from mcp.server.fastmcp.exceptions import ToolError  # noqa: F401
    from mcp.types import ToolAnnotations  # noqa: F401
