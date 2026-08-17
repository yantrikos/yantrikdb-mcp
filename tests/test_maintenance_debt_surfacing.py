"""Maintenance-debt surfacing — the debt rides IN tool responses.

In a reactive deployment the calling LLM is the only scheduler: nothing else
will ever run think(). So write debt is surfaced inside remember/recall/think
responses — flat data plus a suggestion, no urgency prose — against the
engine contract `db.maintenance_debt()` -> {writes_since_think, last_think_at,
open_conflicts, pending_triggers}. On an engine WITHOUT the method (the
published 0.15.x line, and the HTTP backend) every part of this silently
no-ops; that degradation is itself a contract and is tested here.
"""
from __future__ import annotations

import json

import pytest

from yantrikdb_mcp import tools
from yantrikdb_mcp.tools import recall, remember, think


class _Ctx:
    def __init__(self, db):
        self.request_context = type(
            "R", (), {"lifespan_context": {"lazy": type("L", (), {"db": db})()}}
        )()


class LegacyEngine:
    """Minimal engine exposing exactly what the surfacing paths touch —
    WITHOUT maintenance_debt (the published 0.15.x shape). This is the base
    on purpose: the degradation contract is 'the method is truly absent',
    not 'the method is overridden'."""

    def __init__(self, writes=0, last_think_at=None, open_conflicts=0,
                 pending_triggers=0):
        self.debt = {
            "writes_since_think": writes,
            "last_think_at": last_think_at,
            "open_conflicts": open_conflicts,
            "pending_triggers": pending_triggers,
        }
        self.recorded = 0
        self.debt_probes = 0

    def record(self, text, **kw):
        self.recorded += 1
        return f"rid-{self.recorded}"

    def recall_with_response(self, **kw):
        return {"results": [], "confidence": 0.0, "hints": []}

    def think(self, config):
        # A think pass clears write debt on the real engine; mirror that so
        # the post-run payload shows the debt actually cleared.
        self.debt["writes_since_think"] = 0
        return {"triggers": [], "consolidation_count": 0, "conflicts_found": 0,
                "patterns_new": 0, "patterns_updated": 0, "expired_triggers": 0,
                "duration_ms": 1.0}

    def get_patterns(self, **kw):
        return []


class FakeEngine(LegacyEngine):
    """LegacyEngine + the v0.16 maintenance_debt contract."""

    def maintenance_debt(self):
        self.debt_probes += 1
        return dict(self.debt)


class BrokenDebtEngine(FakeEngine):
    """Has the method, but it blows up — the probe must never fail a call."""

    def maintenance_debt(self):
        raise RuntimeError("debt counters unavailable")


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Per-test in-process state + deterministic env."""
    monkeypatch.setattr(tools, "_NUDGE_STATE",
                        {"armed": True, "since_last": 0, "conflicts_surfaced": 0})
    monkeypatch.delenv("YANTRIKDB_THINK_NUDGE_THRESHOLD", raising=False)
    monkeypatch.delenv("YANTRIKDB_THINK_NUDGE_EVERY", raising=False)


def _remember(db, text="fact"):
    return json.loads(remember(text=text, ctx=_Ctx(db)))


def _recall(db):
    return json.loads(recall(query="anything at all", ctx=_Ctx(db)))


# ── remember(): threshold crossing ───────────────────────────────────


def test_crossing_threshold_attaches_maintenance_object():
    db = FakeEngine(writes=50, last_think_at=1755000000.0, open_conflicts=2)
    out = _remember(db)
    assert out["status"] == "recorded", "surfacing must not disturb the write receipt"
    m = out["maintenance"]
    assert m == {"writes_since_think": 50, "last_think_at": 1755000000.0,
                 "open_conflicts": 2, "suggest": "think()"}, (
        "flat facts + suggestion only — no prose fields"
    )


def test_below_threshold_attaches_nothing():
    db = FakeEngine(writes=49)
    out = _remember(db)
    assert "maintenance" not in out


def test_threshold_env_is_honored(monkeypatch):
    monkeypatch.setenv("YANTRIKDB_THINK_NUDGE_THRESHOLD", "5")
    assert "maintenance" in _remember(FakeEngine(writes=5))
    tools._NUDGE_STATE.update(armed=True, since_last=0)
    assert "maintenance" not in _remember(FakeEngine(writes=4))


def test_threshold_zero_disables_and_skips_the_probe(monkeypatch):
    monkeypatch.setenv("YANTRIKDB_THINK_NUDGE_THRESHOLD", "0")
    db = FakeEngine(writes=10_000)
    out = _remember(db)
    assert "maintenance" not in out
    assert db.debt_probes == 0, "0 disables the feature, not just the payload"


def test_batch_write_surfaces_too():
    db = FakeEngine(writes=50)
    out = json.loads(remember(memories=[{"text": "a"}, {"text": "b"}], ctx=_Ctx(db)))
    assert out["count"] == 2
    assert out["maintenance"]["suggest"] == "think()"


# ── remember(): in-process rate limit ────────────────────────────────


def test_rate_limit_suppresses_then_refires_at_the_nth(monkeypatch):
    monkeypatch.setenv("YANTRIKDB_THINK_NUDGE_EVERY", "3")
    db = FakeEngine(writes=50)
    assert "maintenance" in _remember(db), "first crossing surfaces"
    db.debt["writes_since_think"] = 51
    assert "maintenance" not in _remember(db), "suppressed (1 of 3)"
    db.debt["writes_since_think"] = 52
    assert "maintenance" not in _remember(db), "suppressed (2 of 3)"
    db.debt["writes_since_think"] = 53
    out = _remember(db)
    assert out["maintenance"]["writes_since_think"] == 53, "re-fires on the 3rd"
    db.debt["writes_since_think"] = 54
    assert "maintenance" not in _remember(db), "counter reset after re-fire"


def test_dropping_below_threshold_rearms_the_first_crossing():
    db = FakeEngine(writes=50)
    assert "maintenance" in _remember(db)
    db.debt["writes_since_think"] = 3          # debt cleared out-of-band
    assert "maintenance" not in _remember(db)
    db.debt["writes_since_think"] = 50         # crosses again
    assert "maintenance" in _remember(db), (
        "a fresh crossing is a FIRST crossing — no 10-call wait"
    )


# ── degradation contract ─────────────────────────────────────────────


def test_engine_without_maintenance_debt_no_object_no_error():
    db = LegacyEngine(writes=999)
    out = _remember(db)
    assert out == {"rid": "rid-1", "status": "recorded"}
    r = _recall(db)
    assert "open_conflicts" not in r
    t = json.loads(think(ctx=_Ctx(db)))
    assert "maintenance_debt" not in t


def test_probe_failure_never_fails_the_write():
    out = _remember(BrokenDebtEngine(writes=999))
    assert out["status"] == "recorded"
    assert "maintenance" not in out


# ── recall(): open_conflicts changed-since-last-surfaced ─────────────


def test_recall_surfaces_open_conflicts_only_on_change():
    db = FakeEngine(open_conflicts=3)
    assert _recall(db)["open_conflicts"] == 3, "changed from initial 0 — surface"
    assert "open_conflicts" not in _recall(db), "unchanged — silent"
    db.debt["open_conflicts"] = 0
    assert _recall(db)["open_conflicts"] == 0, "transition TO 0 must surface"
    assert "open_conflicts" not in _recall(db), "steady 0 — silent again"


def test_recall_on_a_quiet_store_stays_quiet():
    db = FakeEngine(open_conflicts=0)
    assert "open_conflicts" not in _recall(db), (
        "0 at process start was never 'surfaced' — nothing changed"
    )


# ── think(): post-run debt + rearm ───────────────────────────────────


def test_think_reports_post_run_debt():
    db = FakeEngine(writes=60, open_conflicts=1, pending_triggers=4)
    out = json.loads(think(ctx=_Ctx(db)))
    assert out["maintenance_debt"] == {
        "writes_since_think": 0,       # POST-run: the debt visibly cleared
        "last_think_at": None,
        "open_conflicts": 1,
        "pending_triggers": 4,
    }


def test_think_rearms_the_nudge():
    db = FakeEngine(writes=50)
    assert "maintenance" in _remember(db)      # nudge consumed
    db.debt["writes_since_think"] = 51
    assert "maintenance" not in _remember(db)  # rate-limited
    json.loads(think(ctx=_Ctx(db)))            # pass completes, debt clears
    db.debt["writes_since_think"] = 50         # ...and builds up again
    assert "maintenance" in _remember(db), (
        "post-think crossing must surface immediately, not after N calls"
    )


def test_think_counts_as_surfacing_open_conflicts():
    db = FakeEngine(open_conflicts=2)
    json.loads(think(ctx=_Ctx(db)))            # response carried the count
    assert "open_conflicts" not in _recall(db), (
        "recall must not re-surface a count think() already reported"
    )
