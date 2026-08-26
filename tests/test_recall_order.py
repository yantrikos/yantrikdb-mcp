"""recall(order=...) — the sequence the caller asked for.

The failure this fixes, found 2026-08-26 by testing the live production
store: `recall(query="CT128 deploy", top_k=4, order="recency")` and the
same call with no order returned BYTE-IDENTICAL results — same rids, same
sequence, same scores — and that sequence was not recency (2026-07-17,
08-20, 08-20, 06-23). `order` was not a parameter of this tool at all, so
passing it was silently ignored rather than rejected.

The engine has honoured `order` since #46: verified directly on 0.18.0
with rows aged 300/200/100/1 days, order="recency" returns 1/100/200/300
and order="first_mention" returns 300/200/100/1. Only the MCP layer was
missing.

This is the second time this exact shape has bitten this tool —
test_time_window_range.py exists because `since`/`until` were silently
dropped — so these tests assert the thing that catches it: that two
different orders produce DIFFERENT sequences. A test that merely checked
`order="recency"` returns rows would have passed against the broken
build.
"""
from __future__ import annotations

import json
import time

import pytest

from yantrikdb_mcp._compat import ToolError
from yantrikdb_mcp.tools import recall


class _Ctx:
    def __init__(self, db):
        self.request_context = type(
            "R", (), {"lifespan_context": {"lazy": type("L", (), {"db": db})()}}
        )()


@pytest.fixture
def ctx(tmp_path):
    import os

    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    db = load_engine(str(tmp_path / "order.db"), model_name="bundled")
    yield _Ctx(db)
    try:
        db.close()
    except AttributeError:
        pass


def _seed_ages(ctx):
    """Four equally-relevant rows at known, well-separated ages."""
    db = ctx.request_context.lifespan_context["lazy"].db
    now = time.time()
    for text, age_days in [
        ("deploy runbook alpha", 300),
        ("deploy runbook beta", 200),
        ("deploy runbook gamma", 100),
        ("deploy runbook delta", 1),
    ]:
        db.record(
            text, memory_type="episodic", namespace="ord",
            created_at=now - age_days * 86400,
        )
    return now


def _labels(payload):
    return [r["text"].split()[-1] for r in json.loads(payload)["results"]]


def test_recency_puts_the_newest_first(ctx):
    _seed_ages(ctx)
    labels = _labels(recall("deploy runbook", top_k=4, order="recency", ctx=ctx))
    assert labels == ["delta", "gamma", "beta", "alpha"], labels


def test_first_mention_puts_the_oldest_first(ctx):
    _seed_ages(ctx)
    labels = _labels(recall("deploy runbook", top_k=4, order="first_mention", ctx=ctx))
    assert labels == ["alpha", "beta", "gamma", "delta"], labels


def test_chronological_is_an_alias_for_first_mention(ctx):
    _seed_ages(ctx)
    a = _labels(recall("deploy runbook", top_k=4, order="chronological", ctx=ctx))
    b = _labels(recall("deploy runbook", top_k=4, order="first_mention", ctx=ctx))
    assert a == b, (a, b)


def test_two_orders_do_not_produce_the_same_sequence(ctx):
    """The assertion that would have caught the original defect.

    When `order` was ignored, every order returned the identical sequence.
    Checking that a given order 'returns rows' passes against that bug;
    checking that two orders DIFFER does not.
    """
    _seed_ages(ctx)
    newest = _labels(recall("deploy runbook", top_k=4, order="recency", ctx=ctx))
    oldest = _labels(recall("deploy runbook", top_k=4, order="first_mention", ctx=ctx))
    assert newest != oldest, f"order had no effect: {newest}"
    assert newest == list(reversed(oldest)), (newest, oldest)


def test_an_unknown_order_is_refused_not_ignored(ctx):
    _seed_ages(ctx)
    with pytest.raises(ToolError, match="order must be one of"):
        recall("deploy runbook", top_k=4, order="sideways", ctx=ctx)


def test_omitting_order_still_ranks_by_relevance(ctx):
    """The default path must be untouched: no order means the response
    envelope (confidence + hints) that callers already depend on."""
    _seed_ages(ctx)
    payload = json.loads(recall("deploy runbook", top_k=4, ctx=ctx))
    assert len(payload["results"]) == 4
    assert "confidence" in payload
    assert "hints" in payload
