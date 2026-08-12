"""v0.14 engine behavioral-contract cases.

Core shipped 0.14.0 with a breaking-change ledger whose first row reads:

    surface              compile break   ...
    RecordInput (Rust)   YES

A naive reading says this package is affected — `record_batch()` is exactly a
list of RecordInputs. It is NOT, and the reason is worth pinning rather than
re-deriving under time pressure on some future upgrade:

    The break is a RUST struct-literal break. Adding a field to a
    non-#[non_exhaustive] struct breaks Rust callers that construct it
    positionally / exhaustively — i.e. yantrikdb-server. Python crosses the
    pyo3 boundary with DICTS, and an extra optional field is simply a key we
    don't set.

"Compile break" in a ledger is therefore a per-LANGUAGE claim, not a
per-consumer one. The test below asserts the thing that actually matters to us:
the exact dict shape tools.py builds still round-trips.

Python-side delta is additive: record(created_at=), record_text(created_at=),
and get_memory() as an alias of get().
"""
from __future__ import annotations

import inspect
import os
import tempfile
import time

import pytest

from yantrikdb import YantrikDB

HAS_CREATED_AT = "created_at" in inspect.signature(YantrikDB.record).parameters
HAS_GET_MEMORY = hasattr(YantrikDB, "get_memory")


@pytest.fixture(scope="module")
def db():
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    d = load_engine(os.path.join(tempfile.mkdtemp(), "v014.db"), model_name="bundled")
    yield d
    try:
        d.close()
    except AttributeError:
        pass


def test_record_batch_still_accepts_the_dict_shape_tools_builds(db):
    """THE test that answers the ledger's RecordInput row for this consumer.

    Mirrors tools.remember()'s batch construction exactly. If a future engine
    adds a REQUIRED field to RecordInput, this fails — which is the real
    breaking change for Python, as opposed to the Rust-only compile break.
    """
    inputs = [{
        "text": "Contract batch item", "memory_type": "semantic",
        "importance": 0.5, "valence": 0.0, "metadata": {},
        "namespace": "v14", "certainty": 0.8, "domain": "general",
        "source": "user", "emotional_state": None,
    }]
    rids = db.record_batch(inputs)
    assert isinstance(rids, list) and len(rids) == 1
    assert db.get(rids[0]) is not None


@pytest.mark.skipif(not HAS_CREATED_AT, reason="engine lacks created_at — pre-v0.14")
def test_created_at_is_honoured_on_single_and_batch(db):
    past = time.time() - 45 * 86400
    rid = db.record("Backdated single", namespace="v14", created_at=past)
    assert abs(db.get(rid)["created_at"] - past) < 5

    rids = db.record_batch([{
        "text": "Backdated batch", "memory_type": "semantic", "importance": 0.5,
        "valence": 0.0, "metadata": {}, "namespace": "v14", "certainty": 0.8,
        "domain": "general", "source": "user", "emotional_state": None,
        "created_at": past,
    }])
    assert abs(db.get(rids[0])["created_at"] - past) < 5, (
        "record_batch must honour a per-item created_at — the batch path is "
        "the one that matters for backfill, which is why backdating exists"
    )


@pytest.mark.skipif(not HAS_GET_MEMORY, reason="engine lacks get_memory — pre-v0.14")
def test_get_memory_is_an_alias_of_get(db):
    rid = db.record("Alias probe", namespace="v14")
    assert db.get_memory(rid)["rid"] == db.get(rid)["rid"], (
        "get_memory diverged from get — tools.py uses get() everywhere and "
        "would need auditing if these are no longer interchangeable"
    )
