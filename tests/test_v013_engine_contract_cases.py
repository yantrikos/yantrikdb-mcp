"""v0.13 engine behavioral-contract cases.

The 0.12 -> 0.13 delta removed no methods, but it contains the most dangerous
kind of change this project has hit: a DEFAULT THAT FLIPPED.

    0.12.0:  recall(..., expand_entities=True,  ...)
    0.13.4:  recall(..., expand_entities=False, ...)

Nothing raises. Signatures still accept the same arguments. Every existing
test still passes. Recall simply starts returning different results for any
caller that relied on the default — silently, and in a way an assertion-based
suite will not notice unless it is looking for exactly this.

This project is insulated because `recall()` in tools.py declares its own
default and passes the value explicitly, so agents keep graph boosting. That
is luck reinforced by discipline, not a guarantee, so it is pinned here: if a
future refactor drops the explicit pass-through, this file fails.

Also covers the additive surface: recall(snippets=, min_score_ratio=) and
recall_text(explain=).
"""
from __future__ import annotations

import inspect
import os
import tempfile

import pytest

from yantrikdb import YantrikDB


def _default_of(method, name):
    try:
        p = inspect.signature(method).parameters.get(name)
    except (ValueError, TypeError):
        return None
    return None if p is None else p.default


HAS_MIN_SCORE_RATIO = "min_score_ratio" in (
    inspect.signature(YantrikDB.recall).parameters
    if hasattr(YantrikDB, "recall") else {}
)


@pytest.fixture(scope="module")
def db():
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    d = load_engine(os.path.join(tempfile.mkdtemp(), "v013.db"), model_name="bundled")
    for t in ("Deploy port is 8425", "Deploy timeout is 3s",
              "Deploy leader election", "Coffee machine broken",
              "Cat photos folder"):
        d.record(t, namespace="v13")
    yield d
    try:
        d.close()
    except AttributeError:
        pass


# ── the default flip ─────────────────────────────────────────────────


def test_tools_recall_does_not_rely_on_the_engine_default(db):
    """THE regression guard for the 0.13 flip.

    Whatever the engine's default for expand_entities happens to be, the MCP
    tool must declare and pass its own — otherwise an engine bump silently
    changes what agents recall, with the whole suite still green.
    """
    from yantrikdb_mcp import tools

    tool_default = _default_of(tools.recall, "expand_entities")
    assert tool_default is True, (
        "recall() must declare its OWN expand_entities default (True), not "
        "inherit whatever the installed engine ships"
    )
    src = inspect.getsource(tools.recall)
    assert "expand_entities=expand_entities" in src, (
        "recall() must PASS expand_entities through explicitly. Engine 0.13 "
        "flipped this default True->False; relying on it means an engine "
        "upgrade silently narrows every agent's recall while tests stay green."
    )


def test_engine_default_flip_is_documented_not_forgotten():
    """A tripwire on the engine itself. If a later engine flips this BACK,
    this fails and the comment above gets revisited rather than rotting into
    folklore that no longer matches reality."""
    d = _default_of(YantrikDB.recall, "expand_entities")
    assert d in (True, False), "expand_entities should still be a bool default"
    if d is True:
        pytest.fail(
            "engine recall(expand_entities=) is True again — it was False as of "
            "0.13.4. Re-read the flip notes in this file and in tools.recall(); "
            "the insulation may no longer be describing reality."
        )


# ── additive surface ─────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_MIN_SCORE_RATIO, reason="engine lacks min_score_ratio — pre-0.13")
def test_engine_min_score_ratio_matches_a_client_side_filter(db):
    """The MCP layer implements min_score_ratio CLIENT-SIDE, because the engine
    kwarg exists only on row-level recall() and not on the
    recall_with_response() path the tool actually uses.

    That substitution is only legitimate if the two are equivalent — i.e. the
    engine filters after ranking rather than backfilling to top_k. Pin it, so
    the day the engine starts backfilling we find out here instead of through
    a quietly different result set.
    """
    for ratio in (0.5, 0.8, 0.95):
        engine_rows = db.recall(query="deploy configuration", namespace="v13",
                                top_k=5, min_score_ratio=ratio)
        base = db.recall(query="deploy configuration", namespace="v13", top_k=5)
        top = max((r.get("score", 0.0) for r in base), default=0.0)
        client_rows = [r for r in base if r.get("score", 0.0) >= ratio * top]
        assert [r.get("rid") for r in engine_rows] == [r.get("rid") for r in client_rows], (
            f"engine min_score_ratio={ratio} diverged from a client-side ratio "
            f"filter — the tool layer's client-side implementation is no longer "
            f"equivalent and must be revisited"
        )


@pytest.mark.skipif(
    "snippets" not in inspect.signature(YantrikDB.recall).parameters,
    reason="engine lacks snippets — pre-0.13",
)
def test_snippets_is_row_level_only_which_is_why_it_is_not_exposed(db):
    """Documents a deliberate NON-exposure: `snippets` exists on row-level
    recall but NOT on recall_with_response, so surfacing it as a tool param
    would give agents a flag that silently does nothing on the default path.
    If it ever lands on recall_with_response, this fails and we should expose
    it."""
    assert "snippets" not in inspect.signature(YantrikDB.recall_with_response).parameters, (
        "recall_with_response now supports `snippets` — the reason for not "
        "exposing it as a recall() tool param is gone; wire it up."
    )
