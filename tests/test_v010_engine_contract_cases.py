"""v0.10 engine behavioral-contract cases — the six post-tag assertions.

THE GATING RULE (the release train's thesis): every case gates on a FEATURE
PROBE (hasattr / signature / raised-type), never on a version parse. Two
builds both self-reporting "0.9.4" behaved oppositely on correct(new_text=)
— the version string is a backstop, the probe is the gate. On the released
v0.9.x engine these cases SKIP with a reason naming the missing surface; on
the v0.10 engine they RUN and become the regression wall.

Cases contracted with yantrikdb-core (recorded 2026-07-17):
  1. superseded-exclusion    — recall hides superseded rows by default
  2. supersede-vs-dispute    — supersede HIDES loser; dispute keeps BOTH (T05)
  3. ns-normalization        — blank namespace stored as literal "default"
  4. idempotency             — same key+payload → same rid; same key+diff → conflict
  5. T07 zero-writes         — duplicate keyed write leaves store untouched
  6. typed-error branching   — IdempotencyConflict catchable as its own type
"""
from __future__ import annotations

import inspect
import os
import tempfile

import pytest

import yantrikdb


# ── feature probes (not version parses) ──────────────────────────────


def _engine_param(method, name: str) -> bool:
    """True when the pyo3 method's signature exposes `name`."""
    try:
        return name in inspect.signature(method).parameters
    except (ValueError, TypeError):
        return False


HAS_TYPED_ERRORS = hasattr(yantrikdb, "IdempotencyConflict")


@pytest.fixture(scope="module")
def db():
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    d = load_engine(os.path.join(tempfile.mkdtemp(), "contract.db"), model_name="bundled")
    yield d
    try:
        d.close()
    except AttributeError:
        pass


def _has_recall_superseded(d) -> bool:
    return _engine_param(d.recall_with_response, "include_superseded") or _engine_param(
        getattr(d, "recall", d.recall_with_response), "include_superseded"
    )


def _has_record_idempotency(d) -> bool:
    return _engine_param(d.record, "idempotency_key")


# ── 1 + 2. superseded-exclusion / supersede-vs-dispute ───────────────
# Core's recipe (swarm 0fdf5d94, 2026-07-17): supersede = link(NEW, OLD,
# "supersedes") — SOURCE supersedes TARGET, the new record is always the
# source. No implicit supersede: recording into a namespace never creates
# the link; chain_head WALKS explicit links, it doesn't create them.
# Supersedes does not tombstone — the loser stays reachable by rid.


def _rids(rows):
    return [r.get("rid") for r in (rows or []) if isinstance(r, dict)]


def test_recall_excludes_superseded_by_default(db):
    if not _has_recall_superseded(db):
        pytest.skip("engine lacks include_superseded — pre-v0.10 surface")
    old = db.record("Contract target is Python 3.11", namespace="sup_case")
    new = db.record("Contract target is Python 3.12", namespace="sup_case")
    db.link(new, old, "supersedes")

    default = db.recall(query="contract target python", namespace="sup_case", top_k=10)
    archaeo = db.recall(query="contract target python", namespace="sup_case",
                        top_k=10, include_superseded=True)

    assert old not in _rids(default), "superseded row must be EXCLUDED by default"
    assert new in _rids(default), "the superseding row must remain visible"
    assert old in _rids(archaeo) and new in _rids(archaeo), (
        "include_superseded=True must serve BOTH revisions"
    )
    # Non-tombstone semantics: the loser is demoted from recall, not deleted.
    assert db.get(old) is not None, "supersedes must NOT tombstone the target"


def test_supersede_hides_loser_but_dispute_keeps_both(db):
    """T05 non-conflation: supersede HIDES the loser; dispute (contradicts)
    keeps BOTH visible. The two link types must never behave alike."""
    if not _has_recall_superseded(db):
        pytest.skip("engine lacks include_superseded — pre-v0.10 surface")
    s_old = db.record("Deploy window opens at 02:00 UTC", namespace="t05_sup")
    s_new = db.record("Deploy window opens at 04:00 UTC", namespace="t05_sup")
    db.link(s_new, s_old, "supersedes")

    d_a = db.record("The rate limit is 100 rps", namespace="t05_disp")
    d_b = db.record("The rate limit is 500 rps", namespace="t05_disp")
    db.link(d_b, d_a, "contradicts")

    sup = _rids(db.recall(query="deploy window opens", namespace="t05_sup", top_k=10))
    disp = _rids(db.recall(query="rate limit rps", namespace="t05_disp", top_k=10))

    assert s_old not in sup and s_new in sup, "supersede must hide the loser"
    assert d_a in disp and d_b in disp, "dispute must keep BOTH visible"


@pytest.mark.xfail(
    strict=True,
    reason="yantrikdb#110 — two mechanisms own one concept: disputed_with "
           "populates from OPEN rows in the CONFLICTS table only; a "
           "'contradicts' record-link is stored (bidirectional in "
           "linked_records) but creates no conflicts row, so recall never "
           "surfaces it. The python surface has no public conflict-CREATE, "
           "so the populating kind of dispute can't be constructed manually. "
           "strict=True — alerts when #110 lands.",
)
def test_dispute_populates_disputed_with(db):
    d_a = db.record("The cache TTL is 60 seconds", namespace="t05_dw")
    d_b = db.record("The cache TTL is 300 seconds", namespace="t05_dw")
    db.link(d_b, d_a, "contradicts")
    rows = db.recall(query="cache TTL seconds", namespace="t05_dw", top_k=10)
    by_rid = {r["rid"]: r for r in rows if isinstance(r, dict)}
    assert d_a in by_rid and d_b in by_rid
    assert d_a in (by_rid[d_b].get("disputed_with") or []), (
        "recall result must carry the disputing rid in disputed_with"
    )
    assert d_b in (by_rid[d_a].get("disputed_with") or []), (
        "contradicts is quasi-symmetric — both sides must cross-reference"
    )


# ── 3. namespace normalization ───────────────────────────────────────


def test_blank_namespace_stored_as_literal_default(db):
    """v0.10 item 3: record with blank/whitespace namespace is STORED under
    the literal namespace "default" — verified via get(), not recall, so a
    recall-side normalization can't mask a storage-side miss."""
    try:
        rid = db.record("ns-normalization probe fact", namespace="   ")
    except (ValueError, RuntimeError):
        pytest.skip("engine rejects blank namespace outright — pre-v0.10 behavior")
    mem = db.get(rid)
    assert mem is not None, "recorded memory must be retrievable by rid"
    ns = mem.get("namespace") if isinstance(mem, dict) else getattr(mem, "namespace", None)
    assert ns == "default", (
        f"blank namespace must normalize to literal 'default' at STORAGE, got {ns!r}"
    )


# ── 4 + 5. idempotency + T07 zero-writes ─────────────────────────────


def test_same_key_same_payload_returns_same_rid(db):
    if not _has_record_idempotency(db):
        pytest.skip("engine lacks record(idempotency_key=) — pre-v0.10 surface")
    kw = dict(namespace="idem", idempotency_key="contract-key-1")
    rid1 = db.record("idempotent fact alpha", **kw)
    rid2 = db.record("idempotent fact alpha", **kw)
    assert rid1 == rid2, "same key + same payload must dedupe to the SAME rid"


def test_same_key_different_payload_conflicts(db):
    if not _has_record_idempotency(db):
        pytest.skip("engine lacks record(idempotency_key=) — pre-v0.10 surface")
    db.record("conflict probe original", namespace="idem", idempotency_key="contract-key-2")
    expected = (
        (yantrikdb.IdempotencyConflict,) if HAS_TYPED_ERRORS else (RuntimeError,)
    )
    with pytest.raises(expected):
        db.record("conflict probe DIFFERENT", namespace="idem",
                  idempotency_key="contract-key-2")


def test_t07_duplicate_keyed_write_is_zero_writes(db):
    """T07: repetition is not corroboration — the dedupe hit must leave the
    store untouched (same active count), not silently re-record."""
    if not _has_record_idempotency(db):
        pytest.skip("engine lacks record(idempotency_key=) — pre-v0.10 surface")
    db.record("zero-writes probe", namespace="idem", idempotency_key="contract-key-3")
    before = db.stats().get("active_memories")
    db.record("zero-writes probe", namespace="idem", idempotency_key="contract-key-3")
    after = db.stats().get("active_memories")
    assert after == before, (
        f"duplicate keyed write changed active_memories {before}→{after} — "
        "a dedupe hit must be a ZERO-write"
    )


# ── 6. typed-error branching ─────────────────────────────────────────


def test_typed_errors_branch_without_regex():
    """PR #107 surface: all 7 classes on the package root, all RuntimeError
    subclasses (old handlers keep working), catchable as their own type."""
    if not HAS_TYPED_ERRORS:
        pytest.skip("engine lacks typed exceptions — pre-#107 surface")
    names = (
        "Backpressure", "CorrectionDeferredDuringReembed",
        "BatchDeferredDuringReembed", "IdempotencyConflict",
        "InvalidIdempotencyKey", "ProvenanceInconsistent", "RecallContended",
    )
    for n in names:
        cls = getattr(yantrikdb, n, None)
        assert cls is not None, f"yantrikdb.{n} missing from package root"
        assert issubclass(cls, RuntimeError), (
            f"{n} must subclass RuntimeError so legacy handlers keep working"
        )
