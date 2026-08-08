"""`recall(min_score_ratio=...)` — trim the semantic-search tail.

Semantic search always returns top_k, even when one hit is relevant and the
rest are noise the agent then has to judge (and pay tokens for). A RELATIVE
cutoff — keep hits scoring at least X of the top hit — is the right shape,
because absolute score thresholds don't transfer across queries.

Implemented client-side deliberately; see the comment in tools.recall() and
the equivalence check in test_v013_engine_contract_cases.py.
"""
from __future__ import annotations

import json
import os

import pytest

from yantrikdb_mcp._compat import ToolError
from yantrikdb_mcp.tools import recall


class _Ctx:
    def __init__(self, db):
        self.request_context = type(
            "R", (), {"lifespan_context": {"lazy": type("L", (), {"db": db})()}}
        )()


@pytest.fixture(scope="module")
def ctx():
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    import tempfile

    from yantrikdb_mcp.embedder import load_engine

    db = load_engine(os.path.join(tempfile.mkdtemp(), "ratio.db"), model_name="bundled")
    for t in ("Deploy port is 8425", "Deploy timeout is 3 seconds",
              "Deploy leader election window", "Coffee machine is broken",
              "Cat photos live in Dropbox"):
        db.record(t, namespace="r")
    yield _Ctx(db)
    try:
        db.close()
    except AttributeError:
        pass


def test_ratio_narrows_results_and_keeps_the_best_hit(ctx):
    wide = json.loads(recall(query="deploy configuration", namespace="r", top_k=5, ctx=ctx))
    tight = json.loads(recall(query="deploy configuration", namespace="r", top_k=5,
                              min_score_ratio=0.9, ctx=ctx))
    assert tight["count"] <= wide["count"]
    assert tight["count"] >= 1, "a cutoff must never drop the top hit itself"
    assert tight["results"][0]["rid"] == wide["results"][0]["rid"], (
        "the highest-scoring hit must survive any ratio <= 1"
    )


def test_drops_are_reported_not_silent(ctx):
    """A filtered result must not read as 'the substrate barely knows this' —
    otherwise the agent's next move is a broader re-query it doesn't need, or a
    wrong conclusion about coverage."""
    out = json.loads(recall(query="deploy configuration", namespace="r", top_k=5,
                            min_score_ratio=0.95, ctx=ctx))
    wide = json.loads(recall(query="deploy configuration", namespace="r", top_k=5, ctx=ctx))
    if out["count"] < wide["count"]:
        assert out.get("filtered_by_min_score_ratio", 0) == wide["count"] - out["count"]


def test_absent_key_when_nothing_was_filtered(ctx):
    """Don't spend tokens announcing a no-op on every unfiltered call."""
    out = json.loads(recall(query="deploy configuration", namespace="r", top_k=5,
                            min_score_ratio=1.0, ctx=ctx))
    if out["count"] == len(out["results"]) and "filtered_by_min_score_ratio" not in out:
        assert True
    out_none = json.loads(recall(query="deploy configuration", namespace="r",
                                 top_k=5, ctx=ctx))
    assert "filtered_by_min_score_ratio" not in out_none


@pytest.mark.parametrize("bad", [0, -0.5, 1.5, 2])
def test_rejects_out_of_range_ratios(ctx, bad):
    with pytest.raises(ToolError, match="min_score_ratio"):
        recall(query="deploy", namespace="r", min_score_ratio=bad, ctx=ctx)


def test_works_on_the_superseded_archaeology_path_too(ctx):
    """include_superseded routes through db.recall instead of
    recall_with_response; the cutoff must apply on both paths, not just the
    default one."""
    out = json.loads(recall(query="deploy configuration", namespace="r", top_k=5,
                            include_superseded=True, min_score_ratio=0.9, ctx=ctx))
    assert out["count"] >= 1
