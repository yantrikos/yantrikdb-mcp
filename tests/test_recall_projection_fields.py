"""The recall projection must not discard what makes a ranking auditable.

Reported by yantrikdb-core (2026-08-12, dogfooding). The engine emits ~20
fields per hit including `created_at` and a full `scores` breakdown; this
layer's compact projection forwarded 7 and dropped the rest — including every
timestamp.

WHY THAT WAS SERIOUS, in core's own measured case:

    recall("current published yantrikdb engine version", top_k=4)
    ground truth: 0.14.0, published that same day

    rank 1  score 0.9292  -> a March-2026 blob whose only version claim
                             is "yantrikdb-mcp v0.1.0"
    rank 2  score 0.8449  -> the correct record, written that day

recall ranks by SIMILARITY, not time — and this same layer emits a hint
saying so ("for the exact latest entry use chain_head") while withholding the
timestamps a caller would need to act on it. The ranking was both wrong and
unauditable. With dates visible, a caller sees "rank 1 = March, rank 2 =
today" and answers correctly DESPITE the imperfect ranking: one field turns an
unrecoverable failure into a recoverable one.

These tests pin the fields, not the wording, so a future re-shaping of the
response can't quietly drop them again.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time

import pytest

from yantrikdb_mcp.tools import recall

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class _Ctx:
    def __init__(self, db):
        self.request_context = type(
            "R", (), {"lifespan_context": {"lazy": type("L", (), {"db": db})()}}
        )()


@pytest.fixture(scope="module")
def ctx():
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    db = load_engine(os.path.join(tempfile.mkdtemp(), "proj.db"), model_name="bundled")
    db.record("Projection probe: the engine version is 0.14.0", namespace="pj")
    db.record("Projection probe: an older note about versions", namespace="pj")
    yield _Ctx(db)
    try:
        db.close()
    except AttributeError:
        pass


def test_every_hit_carries_a_readable_created_at(ctx):
    out = json.loads(recall(query="engine version", namespace="pj", top_k=3, ctx=ctx))
    assert out["results"], "probe should return hits"
    for hit in out["results"]:
        assert "created_at" in hit, (
            "recall hits must carry created_at — without it a caller cannot "
            "tell a stale top-ranked hit from a current lower-ranked one"
        )
        assert ISO_RE.match(hit["created_at"]), (
            f"created_at should be ISO-8601 UTC for at-a-glance comparison, "
            f"got {hit['created_at']!r}"
        )
        assert isinstance(hit["created_at_unix"], (int, float)), (
            "raw epoch must remain available for callers doing real comparisons"
        )


def test_similarity_is_exposed_not_just_the_blended_score(ctx):
    """`score` is a blend (similarity x decay x recency x importance ...), so
    two hits can share a score for entirely different reasons. similarity is
    what actually explains the ranking."""
    out = json.loads(recall(query="engine version", namespace="pj", top_k=3, ctx=ctx))
    for hit in out["results"]:
        assert "similarity" in hit, "similarity must be exposed alongside score"
        assert 0.0 <= hit["similarity"] <= 1.0


def test_prose_and_numbers_coexist(ctx):
    """why_retrieved reports recency as an adjective ("recent"). It stays —
    but as a companion to the numbers, never a substitute for them."""
    out = json.loads(recall(query="engine version", namespace="pj", top_k=2, ctx=ctx))
    hit = out["results"][0]
    assert "why_retrieved" in hit
    assert {"created_at", "similarity"} <= set(hit)


def test_timestamps_survive_the_superseded_archaeology_path(ctx):
    """include_superseded routes through db.recall instead of
    recall_with_response — a different code path that must not lose the
    fields the default path now carries."""
    out = json.loads(recall(query="engine version", namespace="pj", top_k=3,
                            include_superseded=True, ctx=ctx))
    for hit in out["results"]:
        assert "created_at" in hit, "archaeology path dropped created_at"


def test_dates_make_a_stale_top_hit_recoverable(ctx):
    """Core's scenario in miniature: an older record that OUT-RANKS a newer,
    correct one is still answerable, because the caller can see which is
    which."""
    db = ctx.request_context.lifespan_context["lazy"].db
    old = time.time() - 200 * 86400
    try:
        db.record("Recoverable probe: the version is 0.1.0", namespace="rec",
                  importance=1.0, created_at=old)
    except TypeError:
        pytest.skip("engine lacks created_at backdating — pre-v0.14")
    db.record("Recoverable probe: the version is 0.14.0", namespace="rec",
              importance=0.5)

    out = json.loads(recall(query="the version", namespace="rec", top_k=2, ctx=ctx))
    dates = [h["created_at"] for h in out["results"]]
    assert len(set(dates)) == len(dates), (
        "the two hits must be distinguishable by date — that distinguishability "
        "is the whole fix"
    )
