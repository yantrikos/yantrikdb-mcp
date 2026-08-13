"""`remember(created_at=...)` — backdating, and why it matters.

v0.13.0 shipped time-travel recall (`temporal(action="as_of")`). Backdating is
the other half: without it, every imported memory stamps "now", so a backfilled
chat log or migration produces a substrate whose history is uniformly today.
`as_of` over that data then reports a past that never happened — confidently.

So this is not a convenience flag. It is what keeps the v0.13 feature honest on
imported data, which is exactly the data most likely to be queried historically.
"""
from __future__ import annotations

import inspect
import json
import os
import tempfile
import time

import pytest

import yantrikdb
from yantrikdb_mcp._compat import ToolError
from yantrikdb_mcp.tools import remember, temporal

ENGINE_SUPPORTS = "created_at" in inspect.signature(yantrikdb.YantrikDB.record).parameters
DAY = 86400


class _Ctx:
    def __init__(self, db):
        self.request_context = type(
            "R", (), {"lifespan_context": {"lazy": type("L", (), {"db": db})()}}
        )()


@pytest.fixture
def ctx():
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    db = load_engine(os.path.join(tempfile.mkdtemp(), "ca.db"), model_name="bundled")
    yield _Ctx(db)
    try:
        db.close()
    except AttributeError:
        pass


@pytest.mark.skipif(not ENGINE_SUPPORTS, reason="engine lacks created_at — pre-v0.14")
def test_single_memory_is_backdated(ctx):
    past = time.time() - 30 * DAY
    rid = json.loads(remember(text="Backdated fact", namespace="ca",
                              created_at="30d", ctx=ctx))["rid"]
    stored = ctx.request_context.lifespan_context["lazy"].db.get(rid)
    assert abs(stored["created_at"] - past) < 120, (
        f"created_at not honoured: stored {stored['created_at']}, wanted ~{past}"
    )


@pytest.mark.skipif(not ENGINE_SUPPORTS, reason="engine lacks created_at — pre-v0.14")
def test_batch_supports_per_item_and_call_level_dates(ctx):
    """A single import carries memories from different moments, so per-item
    dates must win over the call-level default rather than being ignored."""
    out = json.loads(remember(namespace="ca", created_at="10d", memories=[
        {"text": "Batch item using the call-level date"},
        {"text": "Batch item with its own date", "created_at": "60d"},
    ], ctx=ctx))
    db = ctx.request_context.lifespan_context["lazy"].db
    dates = [db.get(r)["created_at"] for r in out["rids"]]
    now = time.time()
    assert abs(dates[0] - (now - 10 * DAY)) < 120, "call-level date must apply"
    assert abs(dates[1] - (now - 60 * DAY)) < 120, "per-item date must override"


@pytest.mark.skipif(not ENGINE_SUPPORTS, reason="engine lacks created_at — pre-v0.14")
def test_backdating_makes_as_of_tell_the_truth(ctx):
    """The point of the feature, end to end: a memory backdated to 60 days ago
    must be INVISIBLE to an as_of cutoff of 30 days ago, and visible today.
    Without backdating it would appear at import time and `as_of` would report
    a history that never happened."""
    remember(text="Historical: the old port was 8080", namespace="hist",
             created_at="60d", ctx=ctx)
    remember(text="Current: the port is 8425", namespace="hist", ctx=ctx)

    old_view = json.loads(temporal(action="as_of", query="the port",
                                   as_of="90d", namespace="hist", ctx=ctx))
    mid_view = json.loads(temporal(action="as_of", query="the port",
                                   as_of="30d", namespace="hist", ctx=ctx))
    now_view = json.loads(temporal(action="as_of", query="the port",
                                   as_of=str(time.time()), namespace="hist", ctx=ctx))

    assert old_view["count"] == 0, "nothing existed 90 days ago"
    mid = json.dumps(mid_view["results"])
    assert "8080" in mid and "8425" not in mid, (
        "at the 30-day cutoff only the backdated memory should exist"
    )
    assert "8425" in json.dumps(now_view["results"]), "today must see both"


def test_bad_date_is_rejected_with_the_grammar(ctx):
    with pytest.raises(ToolError) as e:
        remember(text="x", created_at="whenever", ctx=ctx)
    assert "created_at" in str(e.value), "the error must name the offending field"


@pytest.mark.skipif(ENGINE_SUPPORTS, reason="only meaningful on a pre-v0.14 engine")
def test_refuses_rather_than_silently_recording_now(ctx):
    """On an engine that cannot backdate, recording at 'now' anyway would
    corrupt the very thing the caller asked for. Refuse and name the fix."""
    out = json.loads(remember(text="x", created_at="30d", ctx=ctx))
    assert "error" in out and "0.14" in json.dumps(out)
