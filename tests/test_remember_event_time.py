"""remember(event_time=...) and claim windows — the temporal tag at ingestion.

A memory can say when it is ABOUT, not only when it was written. The tag
lands as metadata event_time_min/max (which the engine persists as columns
and reads for as_of recall) and every claim on the memory inherits it as
valid_from unless the claim carries its own window. That is what makes a
2024 fact the predecessor of the same 2026 fact rather than its
contradiction.

Engine < 0.20 (no StatedClaim windows) still records the tag on the memory;
only the claim-window checks are skipped there.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

import yantrikdb
from yantrikdb_mcp.tools import _claims_with_windows, _with_event_time, remember

ENGINE_CLAIMS = hasattr(yantrikdb.YantrikDB, "attach_claims")
DAY = 86_400.0
T_2024 = 1_704_067_200.0  # 2024-01-01T00:00:00Z


class _Ctx:
    def __init__(self, db):
        self.request_context = type(
            "R", (), {"lifespan_context": {"lazy": type("L", (), {"db": db})()}}
        )()


@pytest.fixture
def ctx():
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    db = load_engine(os.path.join(tempfile.mkdtemp(), "et.db"), model_name="bundled")
    yield _Ctx(db)
    try:
        db.close()
    except AttributeError:
        pass


def _db(ctx):
    return ctx.request_context.lifespan_context["lazy"].db


def test_event_time_parses_the_same_forms_as_created_at():
    meta = _with_event_time({"topic": "x"}, "2024-01-01")
    assert meta["topic"] == "x"
    assert abs(meta["event_time_min"] - T_2024) < DAY and meta["event_time_min"] == meta["event_time_max"]
    rel = _with_event_time(None, "7d")
    assert abs(rel["event_time_min"] - (time.time() - 7 * DAY)) < 120
    assert _with_event_time({"a": 1}, None) == {"a": 1}
    # An explicit window in metadata is not overwritten by the shorthand.
    kept = _with_event_time({"event_time_min": 1.0, "event_time_max": 2.0}, "2024-01-01")
    assert (kept["event_time_min"], kept["event_time_max"]) == (1.0, 2.0)


def test_claim_windows_are_parsed_and_other_keys_pass_through():
    out = _claims_with_windows([
        {"subject": "A", "relation": "r", "object": "B", "valid_from": "2024-01-01", "valid_to": None},
        {"subject": "A", "relation": "r", "object": "C"},
    ])
    assert abs(out[0]["valid_from"] - T_2024) < DAY and out[0]["valid_to"] is None
    assert "valid_from" not in out[1]
    assert _claims_with_windows(None) is None


def test_event_time_lands_on_the_record(ctx):
    rid = json.loads(remember(text="Alice Moreau works at Fennwick Labs.", namespace="et",
                              event_time="2024-01-01", ctx=ctx))["rid"]
    stored = _db(ctx).get(rid)
    meta = stored.get("metadata") or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert abs(meta["event_time_min"] - T_2024) < DAY, stored


@pytest.mark.skipif(not ENGINE_CLAIMS, reason="engine lacks attach_claims — pre-v0.19")
def test_claims_inherit_event_time_and_an_explicit_window_wins(ctx):
    out = json.loads(remember(
        text="Alice Moreau works at Fennwick Labs, and Alice Moreau lives in Berlin.",
        namespace="et", event_time="2024-01-01",
        claims=[
            {"subject": "Alice Moreau", "relation": "based_in", "object": "Berlin"},
            {"subject": "Alice Moreau", "relation": "visited", "object": "Berlin",
             "valid_from": "2025-01-01", "valid_to": "2025-01-02"},
        ],
        ctx=ctx,
    ))
    assert out["claims"]["accepted"] == 2, out
    db = _db(ctx)
    if not hasattr(db, "get_claims"):
        pytest.skip("engine lacks get_claims (claim windows) — pre-v0.20")
    claims = {(c["src"], c["rel_type"], c["dst"]): c for c in db.get_claims("Alice Moreau")}
    based = claims[("Alice Moreau", "based_in", "Berlin")]
    visited = claims[("Alice Moreau", "visited", "Berlin")]
    assert abs(based["valid_from"] - T_2024) < DAY, based
    assert visited["valid_from"] > based["valid_from"] and visited["valid_to"] > visited["valid_from"], visited
