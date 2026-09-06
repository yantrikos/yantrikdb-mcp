"""remember(claims=[...]) — the writer states the facts, the engine grounds them.

Engine 0.19 extracts only a few relation shapes on its own; anything else
the caller wants tracked (contradiction, succession, entity threads,
multi-hop recall) has to be STATED. The tool passes the triples to
`attach_claims`, which grounds each one mechanically and reports back what
it refused — the caller acts on the rejected list, so it is returned whole.

The refusal test matters most: a backend without `attach_claims` must not
record the text and drop the claims, because "recorded" would then be a
lie about the facts the caller asked to have tracked.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import yantrikdb
from yantrikdb_mcp.tools import remember

ENGINE_SUPPORTS = hasattr(yantrikdb.YantrikDB, "attach_claims")


class _Ctx:
    def __init__(self, db):
        self.request_context = type(
            "R", (), {"lifespan_context": {"lazy": type("L", (), {"db": db})()}}
        )()


@pytest.fixture
def ctx():
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    db = load_engine(os.path.join(tempfile.mkdtemp(), "cl.db"), model_name="bundled")
    yield _Ctx(db)
    try:
        db.close()
    except AttributeError:
        pass


def _db(ctx):
    return ctx.request_context.lifespan_context["lazy"].db


@pytest.mark.skipif(not ENGINE_SUPPORTS, reason="engine lacks attach_claims — pre-v0.19")
def test_grounded_claim_is_stored_and_ungrounded_is_reported(ctx):
    out = json.loads(remember(
        text="Alice Moreau works at Fennwick Labs.",
        namespace="cl",
        claims=[
            {"subject": "Alice Moreau", "relation": "works_at", "object": "Fennwick Labs"},
            # Globex is not in the text: must come back rejected, never stored.
            {"subject": "Alice Moreau", "relation": "works_at", "object": "Globex"},
        ],
        ctx=ctx,
    ))
    assert out["status"] == "recorded"
    assert out["claims"]["accepted"] == 1
    assert len(out["claims"]["rejected"]) == 1
    rejected = out["claims"]["rejected"][0]
    assert "Globex" in json.dumps(rejected)
    # The accepted claim is queryable through the engine's graph surface.
    edges = _db(ctx).claims_for_entity("Alice Moreau") if hasattr(_db(ctx), "claims_for_entity") else None
    if edges is not None:
        assert any(e.get("dst") == "Fennwick Labs" or e.get("object") == "Fennwick Labs" for e in edges)


@pytest.mark.skipif(not ENGINE_SUPPORTS, reason="engine lacks attach_claims — pre-v0.19")
def test_no_claims_means_no_claims_key(ctx):
    out = json.loads(remember(text="Plain fact without stated claims.", namespace="cl", ctx=ctx))
    assert out["status"] == "recorded"
    assert "claims" not in out


@pytest.mark.skipif(not ENGINE_SUPPORTS, reason="engine lacks attach_claims — pre-v0.19")
def test_batch_items_carry_their_own_claims(ctx):
    out = json.loads(remember(namespace="cl", memories=[
        {"text": "Bob Lin leads the Atlas team.",
         "claims": [{"subject": "Bob Lin", "relation": "leads", "object": "Atlas"}]},
        {"text": "An item that states nothing."},
        {"text": "Cara Ito lives in Lisbon.",
         "claims": [{"subject": "Cara Ito", "relation": "lives_in", "object": "Lisbon"},
                    {"subject": "Cara Ito", "relation": "lives_in", "object": "Porto"}]},
    ], ctx=ctx))
    assert out["count"] == 3
    reports = out["claims"]
    assert set(reports) == {"0", "2"}, "only items that stated claims get a report"
    assert reports["0"]["accepted"] == 1 and reports["0"]["rejected"] == []
    assert reports["2"]["accepted"] == 1 and len(reports["2"]["rejected"]) == 1


def test_engine_without_attach_claims_refuses_before_writing():
    """An older engine: nothing is recorded, and the error names the fix."""
    class _OldDb:
        def __init__(self):
            self.records = 0

        def record(self, *a, **k):
            self.records += 1
            return "rid"

    db = _OldDb()
    out = json.loads(remember(
        text="Alice works at Fennwick.",
        claims=[{"subject": "Alice", "relation": "works_at", "object": "Fennwick"}],
        ctx=_Ctx(db),
    ))
    assert out.get("error") or out.get("status") != "recorded"
    assert "0.19" in json.dumps(out)
    assert db.records == 0, "the text must not be recorded when its claims cannot be"

    # Batch form is refused the same way.
    out = json.loads(remember(memories=[
        {"text": "x", "claims": [{"subject": "x", "relation": "is", "object": "x"}]},
    ], ctx=_Ctx(db)))
    assert db.records == 0
    assert "0.19" in json.dumps(out)
