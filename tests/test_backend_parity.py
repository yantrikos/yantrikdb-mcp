"""Backend parity contract test — would have caught both the _Conflict
(v0.5.0) and _Stats (v0.5.1) regressions automatically.

Fails CI if:

1. `HttpBackend` is missing any method that `tools.py` calls on the `db`
   object — the v0.5.0 class of regression where `tools.py` invoked
   `db.get_conflict(...)` and `HttpBackend` had no such method, raising
   `AttributeError` instead of a clean `RemoteUnsupportedError`.

2. Any wrapper class in `http_backend.py` (e.g. `_Stats`, `_Conflict`)
   doesn't expose both subscript access (`x["key"]`) and attribute
   access (`x.key`) — the v0.5.1 class of regression where the
   `stats(action="health")` tool path called `s.get("active_memories")`
   on a `_Stats` instance that wasn't a dict.

3. Any RemoteUnsupportedError-raising stub doesn't accept `*args, **kw`
   defensively — the v0.5.1 class of regression where
   `db.procedural_stats(namespace=...)` from the tool layer raised
   `TypeError` before the friendly `RemoteUnsupportedError` could fire.

This test is fully static — it does NOT require a running yantrikdb
server, network, or LLM. Just AST inspection + class introspection.
Runs in <1s and protects every PR.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

# ── locate sources without requiring an install ─────────────────────────
SRC = Path(__file__).parent.parent / "src" / "yantrikdb_mcp"
TOOLS_PATH = SRC / "tools.py"
HTTP_BACKEND_PATH = SRC / "http_backend.py"


def _import_http_backend():
    """Import the package directly from src/ so this test runs both
    against `pip install -e .` and against a fresh checkout."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "yantrikdb_mcp.http_backend", HTTP_BACKEND_PATH
    )
    module = importlib.util.module_from_spec(spec)
    # http_backend uses `import requests` — we only need it as a module
    # object so name resolution works; we don't actually call HTTP code.
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        pytest.skip(f"http_backend.py has unimport-able dep: {e}")
    return module


# ─────────────────────────────────────────────────────────────────────
# 1. Method-existence: every db.X() call in tools.py must exist on
#    HttpBackend (real impl OR explicit RemoteUnsupportedError stub).
# ─────────────────────────────────────────────────────────────────────


def _extract_db_method_calls(tools_src: str) -> set[str]:
    """Walk tools.py AST and collect every `db.<name>(...)` call name."""
    tree = ast.parse(tools_src)
    methods: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            f = node.func
            # Match `db.<attr>(...)` — `db` may be any local name, but the
            # convention in tools.py is `db = _get_db(ctx)` so we look
            # specifically at attribute access on a Name='db'.
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "db"
            ):
                methods.add(f.attr)
            self.generic_visit(node)

    V().visit(tree)
    return methods


def test_http_backend_covers_every_tools_method() -> None:
    """v0.5.0 regression class — `tools.py` called `db.get_conflict()`
    but HttpBackend didn't define it, so users hit raw AttributeError."""
    hb = _import_http_backend()
    HttpBackend = hb.HttpBackend  # noqa: N806

    expected = _extract_db_method_calls(TOOLS_PATH.read_text(encoding="utf-8"))
    actual = {
        n for n in dir(HttpBackend)
        if not n.startswith("_") and callable(getattr(HttpBackend, n))
    }

    missing = expected - actual
    assert not missing, (
        f"HttpBackend is missing {len(missing)} methods that tools.py "
        f"calls on the db object — users in remote-cluster mode would "
        f"hit AttributeError. Add real impls or RemoteUnsupportedError "
        f"stubs for: {sorted(missing)}"
    )


# ─────────────────────────────────────────────────────────────────────
# 2. Wrapper-class dict-compat: every adapter class returned to tools.py
#    must support BOTH dict-style and attribute-style access.
# ─────────────────────────────────────────────────────────────────────


WRAPPER_CLASS_NAMES = (
    "_Stats", "_Conflict", "_ThinkResult", "_Edge", "_ResolveResult",
)


def test_wrapper_classes_are_dict_and_attribute_compatible() -> None:
    """v0.5.1 regression class — `_Stats` was attribute-only, but
    `tools.py` called `s.get("active_memories", 0)` and
    `result["procedural"] = ...`. Both ops must work on every wrapper."""
    hb = _import_http_backend()
    failures: list[str] = []

    for cls_name in WRAPPER_CLASS_NAMES:
        cls = getattr(hb, cls_name, None)
        if cls is None:
            failures.append(f"{cls_name}: not defined")
            continue

        # Construct an instance with an empty dict (every wrapper takes
        # a `d: dict` arg per the existing pattern).
        try:
            inst = cls({})
        except TypeError:
            failures.append(
                f"{cls_name}: constructor doesn't accept a dict — "
                "wrappers must take a single dict argument"
            )
            continue

        # Subscript access (read) — must NOT raise TypeError.
        try:
            _ = inst["nonexistent_key_for_test"]
        except KeyError:
            pass  # KeyError on missing key is fine; that's dict semantics
        except TypeError as e:
            failures.append(
                f"{cls_name}: subscript-read failed with TypeError "
                f"({e}) — class must inherit from dict or define "
                "__getitem__"
            )

        # Subscript-set — required by tools.py's stats handler:
        #     result = db.stats(...); result["procedural"] = ...
        try:
            inst["test_subscript_set"] = "ok"
        except TypeError as e:
            failures.append(
                f"{cls_name}: subscript-set failed with TypeError "
                f"({e}) — class must inherit from dict"
            )

        # `.get(...)` — required by tools.py's stats(action="health"):
        #     s.get("active_memories", 0)
        if not hasattr(inst, "get"):
            failures.append(
                f"{cls_name}: missing .get() method — class must "
                "inherit from dict"
            )

        # Attribute access for known fields — backwards compat with
        # legacy callers using `s.active_memories` style.
        # We don't assert specific field names (those vary per wrapper);
        # we only assert that __getattr__ falls through to the dict
        # for arbitrary keys we set above.
        try:
            _ = inst.test_subscript_set
        except AttributeError:
            failures.append(
                f"{cls_name}: attribute access doesn't fall through "
                "to dict — define __getattr__ to read self[name]"
            )

    assert not failures, (
        "Wrapper-class contract violations:\n  - "
        + "\n  - ".join(failures)
    )


# ─────────────────────────────────────────────────────────────────────
# 3. Stub kwarg-defensiveness: every method that raises
#    RemoteUnsupportedError must accept arbitrary kwargs so the tool
#    layer's kwarg injection doesn't TypeError before the clean error.
# ─────────────────────────────────────────────────────────────────────


def test_unsupported_stubs_accept_arbitrary_kwargs() -> None:
    """v0.5.1 regression class — `procedural_stats(self)` had no kwargs
    so `db.procedural_stats(namespace="x")` raised TypeError instead of
    RemoteUnsupportedError. Every stub now uses *args, **kw."""
    hb = _import_http_backend()
    HttpBackend = hb.HttpBackend  # noqa: N806
    RemoteUnsupportedError = hb.RemoteUnsupportedError  # noqa: N806

    failures: list[str] = []

    for name in dir(HttpBackend):
        if name.startswith("_"):
            continue
        method = getattr(HttpBackend, name)
        if not callable(method):
            continue

        # Determine whether this method is a stub (raises
        # RemoteUnsupportedError unconditionally). Heuristic: read the
        # source and look for `raise self._not_supported(...)` as the
        # first executable statement.
        try:
            src = inspect.getsource(method)
        except (OSError, TypeError):
            continue
        if "raise self._not_supported" not in src:
            continue
        if "if " in src.split("raise self._not_supported")[0]:
            # Conditional stub — exclude (skip the check, it's a real impl)
            continue

        # This IS a stub. Verify its signature accepts **kw.
        sig = inspect.signature(method)
        accepts_var_kwarg = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
        if not accepts_var_kwarg:
            failures.append(
                f"HttpBackend.{name}{sig}: RemoteUnsupportedError stub "
                "must accept **kw so tool-layer kwarg injection "
                "(e.g. namespace=) doesn't TypeError before the "
                "clean RemoteUnsupportedError fires"
            )

    assert not failures, (
        "Stub signature contract violations:\n  - "
        + "\n  - ".join(failures)
    )
