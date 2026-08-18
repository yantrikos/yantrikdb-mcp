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

import json
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


def test_mcp_pin_is_bounded_at_the_next_untested_major() -> None:
    """v0.11.0 pinned <2.0.0 because 2.x removed `mcp.server.fastmcp`. v0.12.0
    ports that surface behind `_compat`, so the ceiling moves to <3.0.0 — the
    next major we have NOT tested. The ceiling moves only when a major is
    actually exercised in CI, never speculatively."""
    spec = next((d for d in _runtime_deps() if _name_of(d) == "mcp"), None)
    assert spec is not None, "mcp is a hard runtime dependency and must be declared"
    assert "<3.0.0" in spec.replace(" ", ""), (
        f"mcp pin {spec!r} must stop at the first UNTESTED major. Both 1.x and "
        f"2.x are supported via yantrikdb_mcp._compat and covered in CI; 3.x is "
        f"not, so it must stay excluded until it is."
    )


def test_the_symbols_the_server_actually_needs_resolve() -> None:
    """The rejection surface for this whole class of failure: not 'is a version
    installed' but 'do the symbols the server imports at module scope exist'.

    Goes through `_compat`, which is the only place allowed to know which SDK
    line is installed."""
    from yantrikdb_mcp._compat import (  # noqa: F401
        MCP_MAJOR,
        Context,
        MCPServer,
        ToolAnnotations,
        ToolError,
    )

    assert MCP_MAJOR in (1, 2), f"unexpected MCP SDK line: {MCP_MAJOR}"
    assert callable(MCPServer)
    assert issubclass(ToolError, Exception)


def test_no_module_bypasses_the_compat_layer() -> None:
    """A direct `from mcp.server.fastmcp import ...` anywhere outside _compat
    silently breaks the other SDK line — and would only be caught on whichever
    CI leg happens to run it. Enforce the single import site structurally."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "yantrikdb_mcp"
    offenders: list[str] = []
    for py in src.glob("*.py"):
        if py.name == "_compat.py":
            continue
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # mcp.types is stable across both majors; the server/* tree is not.
            if re.search(r"\bfrom\s+mcp\.server[.\s]|\bimport\s+mcp\.server\b", stripped):
                offenders.append(f"{py.name}:{i}: {stripped}")
    assert not offenders, (
        "these modules import from `mcp.server.*` directly instead of going "
        "through yantrikdb_mcp._compat — that path moved between SDK majors "
        "and will break one of the two supported lines:\n  "
        + "\n  ".join(offenders)
    )


def test_server_json_version_matches_the_package() -> None:
    """server.json is published to the official MCP registry by
    .github/workflows/mcp-registry.yml on every release, so a stale version
    there tells the registry to advertise a package that is not what shipped.

    It drifted from 0.14.0 to 0.19.2 unnoticed — five minor releases — because
    no release commit touches this file and nothing compared it to pyproject.
    A version surface with no test is a version surface that will drift.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    expected = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)

    server_json = json.loads((root / "server.json").read_text(encoding="utf-8"))
    assert server_json["version"] == expected, (
        f"server.json version {server_json['version']!r} != pyproject "
        f"{expected!r}; the MCP registry would advertise the wrong release"
    )
    for pkg in server_json["packages"]:
        assert pkg["version"] == expected, (
            f"server.json packages[{pkg.get('identifier')!r}] version "
            f"{pkg['version']!r} != pyproject {expected!r}"
        )
