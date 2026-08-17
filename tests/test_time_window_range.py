"""Time-window retrieval: recall(since/until) + temporal(action="range").

The failure these surfaces fix, reproduced live on the production store
2026-08-16: recall("in what order did tonight's releases ship") returned
April/May records keyword-matched on "tonight"/"order"/"release" while six
release memories written HOURS earlier retrieved ZERO — and passing
since/until to the old schema was SILENTLY DROPPED by validation. Sequence
and period questions are SET queries over a time window; similarity search
structurally cannot answer them because the question's words describe the
frame, not the content.

The engine has carried `time_window` since before the rename; these tests
pin the MCP layer actually forwarding it (recall) and the new chronological
window scan (range). Every filtering test is fail-on-old: on v0.18.0 the
params did not exist, so the calls TypeError.
"""
from __future__ import annotations

import json
import time

import pytest

from yantrikdb_mcp._compat import ToolError
from yantrikdb_mcp.tools import recall, temporal


class _Ctx:
    def __init__(self, db):
        self.request_context = type("R", (), {"lifespan_context": {"lazy": type("L", (), {"db": db})()}})()


@pytest.fixture
def ctx(tmp_path):
    import os

    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    db = load_engine(str(tmp_path / "window.db"), model_name="bundled")
    yield _Ctx(db)
    try:
        db.close()
    except AttributeError:
        pass


def _seed_ledger(ctx):
    """Four in-window release events (1h apart) + one 300-day-old decoy that
    similarity LOVES for release-shaped queries — the live failure shape."""
    db = ctx.request_context.lifespan_context["lazy"].db
    now = time.time()
    events = [
        ("engine 0.15.0 shipped to PyPI and crates.io", now - 4 * 3600),
        ("CT128 deployed engine 0.15.0 from the published wheel", now - 3 * 3600),
        ("mcp 0.18.0 released and dogfooded on CT128", now - 2 * 3600),
        ("server 0.16.0 released, train complete", now - 1 * 3600),
    ]
    for text, at in events:
        db.record(text, memory_type="episodic", namespace="tw", created_at=at)
    db.record(
        "ancient release ledger: every deploy shipped in order tonight after tonight",
        memory_type="episodic", namespace="tw", created_at=now - 300 * 86400,
    )
    return now


# ── recall(since/until) ──────────────────────────────────────────────


def test_recall_window_excludes_out_of_window_rows(ctx):
    now = _seed_ledger(ctx)
    out = json.loads(recall("releases shipped", namespace="tw",
                            since="6h", ctx=ctx))
    blob = json.dumps(out["results"])
    assert "0.15.0" in blob, "in-window rows must be retrievable"
    assert "ancient release ledger" not in blob, (
        "a row from 300 days ago must not survive since='6h' — filtering "
        "after ranking (or not at all) is the failure this param fixes"
    )
    # The response must SAY the window was applied; a silently-honored filter
    # is indistinguishable from a silently-dropped one on a dense corpus.
    assert "time_window" in out
    assert out["time_window"]["since"] < out["time_window"]["until"]
    del now


def test_recall_until_alone_means_everything_up_to_then(ctx):
    _seed_ledger(ctx)
    out = json.loads(recall("release ledger deploys", namespace="tw",
                            until="30d", ctx=ctx))
    blob = json.dumps(out["results"])
    assert "ancient release ledger" in blob
    assert "0.15.0" not in blob, "rows newer than `until` must be excluded"


def test_recall_rejects_empty_window(ctx):
    with pytest.raises(ToolError, match="empty time window"):
        recall("anything", since="1h", until="2h", ctx=ctx)


def test_recall_rejects_window_with_refine(ctx):
    with pytest.raises(ToolError, match="refine"):
        recall("refined", refine_from="original", since="6h", ctx=ctx)


def test_recall_window_survives_include_superseded_path(ctx):
    """The archaeology path goes through row-level db.recall — the window
    must ride BOTH paths, not just recall_with_response."""
    _seed_ledger(ctx)
    out = json.loads(recall("releases shipped", namespace="tw", since="6h",
                            include_superseded=True, ctx=ctx))
    assert "ancient release ledger" not in json.dumps(out["results"])


# ── temporal(action="range") ─────────────────────────────────────────


def test_range_requires_since(ctx):
    with pytest.raises(ToolError, match="since"):
        temporal(action="range", ctx=ctx)


def test_range_rejects_empty_window(ctx):
    with pytest.raises(ToolError, match="empty time window"):
        temporal(action="range", since="1h", until="2h", ctx=ctx)


def test_range_scan_returns_window_chronologically(ctx):
    """The dogfood gate: 'what happened tonight, in order' answered from the
    window alone — no query, complete window, oldest first."""
    _seed_ledger(ctx)
    out = json.loads(temporal(action="range", since="6h", namespace="tw", ctx=ctx))
    texts = [r["text"] for r in out["results"]]
    assert len(texts) == 4, f"expected the complete 4-event window, got {texts}"
    assert "ancient" not in json.dumps(texts)
    order = ["engine 0.15.0", "CT128 deployed", "mcp 0.18.0", "server 0.16.0"]
    for expected, got in zip(order, texts):
        assert expected in got, (
            f"chronological order broken: wanted {order}, got {texts}"
        )
    assert out["order"].startswith("chronological")
    assert out["selection"] == "complete window"
    # created_at emitted in both readable and comparable forms.
    assert out["results"][0]["created_at"].endswith("Z")
    assert out["results"][0]["created_at_unix"] < out["results"][1]["created_at_unix"]


def test_range_with_query_selects_by_relevance_inside_window(ctx):
    _seed_ledger(ctx)
    out = json.loads(temporal(action="range", since="6h", query="CT128 deploy",
                              namespace="tw", ctx=ctx))
    blob = json.dumps(out["results"])
    assert "CT128" in blob
    assert "ancient" not in blob
    assert out["selection"] == "relevance-selected within the window"
    # Presentation stays chronological even under relevance selection.
    stamps = [r["created_at_unix"] for r in out["results"]]
    assert stamps == sorted(stamps)


def test_range_truncation_is_named_not_silent(ctx):
    """A window larger than `limit` must SAY it was truncated — 'covered
    everything' when it did not is the lie the selection field prevents."""
    _seed_ledger(ctx)
    out = json.loads(temporal(action="range", since="6h", limit=2,
                              namespace="tw", ctx=ctx))
    assert out["count"] == 2
    assert "newest 2" in out["selection"]
    # Newest two of the window, still presented oldest-first.
    texts = [r["text"] for r in out["results"]]
    assert "mcp 0.18.0" in texts[0] and "server 0.16.0" in texts[1]
