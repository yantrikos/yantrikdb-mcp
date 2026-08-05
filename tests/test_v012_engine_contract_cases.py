"""v0.12 engine behavioral-contract cases — time-travel recall.

Same GATING RULE as the v0.10/v0.11 suites: every case gates on a FEATURE
PROBE (hasattr / signature), never a version parse. On a pre-v0.12 engine
these SKIP naming the missing surface; on v0.12 they RUN and become the wall
that lets pyproject widen the pin to <0.13.0.

The v0.12 delta over 0.11.3 is small and purely additive:
  1. recall_as_of(as_of, query=..., ...) — recall the past AS IT WAS KNOWN
  2. seal_pack(..., recommended_top_k=, recommended_min_similarity=) — the
     retrieval settings a publisher signs into a pack manifest

Behaviours pinned here were learned by probing 0.12.0, not assumed — including
the sharp edge that `as_of` accepts ONLY a real number, which is why the MCP
layer parses agent-friendly forms before calling through.
"""
from __future__ import annotations

import inspect
import os
import tempfile
import time

import pytest

from yantrikdb import YantrikDB


def _param(method, name: str) -> bool:
    try:
        return name in inspect.signature(method).parameters
    except (ValueError, TypeError):
        return False


HAS_AS_OF = hasattr(YantrikDB, "recall_as_of")
HAS_PACK_RECOMMENDATIONS = _param(getattr(YantrikDB, "seal_pack", None), "recommended_top_k") \
    if hasattr(YantrikDB, "seal_pack") else False


@pytest.fixture(scope="module")
def db():
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    d = load_engine(os.path.join(tempfile.mkdtemp(), "v012.db"), model_name="bundled")
    yield d
    try:
        d.close()
    except AttributeError:
        pass


# ── 1. time-travel recall ────────────────────────────────────────────


@pytest.mark.skipif(not HAS_AS_OF, reason="engine lacks recall_as_of — pre-v0.12 surface")
def test_recall_as_of_excludes_later_writes(db):
    """The core contract: a memory recorded AFTER the cutoff must not appear.
    This is what makes the result 'the belief held then' rather than 'today's
    belief filtered by date'."""
    db.record("Time-travel probe: the limit is 100 rps", namespace="asof")
    time.sleep(1.1)
    cut = time.time()
    time.sleep(1.1)
    db.record("Time-travel probe: the limit is 500 rps", namespace="asof")

    past = db.recall_as_of(cut, query="time-travel probe limit rps", namespace="asof", top_k=10)
    now = db.recall_as_of(time.time(), query="time-travel probe limit rps", namespace="asof", top_k=10)

    past_text = " ".join((r.get("text") or "") for r in (past or []) if isinstance(r, dict))
    now_text = " ".join((r.get("text") or "") for r in (now or []) if isinstance(r, dict))

    assert "100 rps" in past_text, "the value known at the cutoff must be returned"
    assert "500 rps" not in past_text, (
        "a memory recorded AFTER as_of leaked into the past view — time-travel "
        "recall would then answer with knowledge the agent did not have"
    )
    assert "500 rps" in now_text, "as_of=now must include the later write"


@pytest.mark.skipif(not HAS_AS_OF, reason="engine lacks recall_as_of — pre-v0.12 surface")
def test_recall_as_of_requires_a_real_number(db):
    """The sharp edge the MCP layer exists to absorb: an ISO string raises
    TypeError. If this ever starts accepting strings, `_parse_as_of` can be
    simplified — so pin it as a tripwire rather than leaving it folklore."""
    with pytest.raises(TypeError):
        db.recall_as_of("2026-08-01T00:00:00Z", query="anything", namespace="asof")


# ── 2. pack retrieval recommendations ────────────────────────────────


@pytest.mark.skipif(
    not HAS_PACK_RECOMMENDATIONS,
    reason="engine seal_pack lacks recommended_* — pre-v0.12 surface",
)
def test_seal_pack_carries_recommended_retrieval_settings(db):
    """A publisher can sign retrieval hints INTO the manifest, so a consumer
    doesn't have to guess top_k / similarity for someone else's corpus."""
    db.record("Recommendation probe row", namespace="recns")
    dest = os.path.join(tempfile.mkdtemp(), "rec.ypack")
    ident = db.embedder_identity() or {}
    db.seal_pack(
        dest, "rec-pack", "1.0.0", "rec-origin", namespace="recns",
        embedder_name=ident.get("name"), embedder_digest=ident.get("digest"),
        embedder_dim=ident.get("dim"),
        recommended_top_k=7, recommended_min_similarity=0.42,
    )
    man = db.read_pack_manifest(dest)
    blob = str(man)
    assert "7" in blob and "0.42" in blob, (
        f"manifest must carry the recommended retrieval settings; got keys "
        f"{sorted(man.keys())}"
    )
