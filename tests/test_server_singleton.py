"""Tests for the v0.8.3 process-singleton `_LazyDB` + atexit close
(yantrikos/yantrikdb-mcp#11).

Background: under SSE transport the lifespan ran once per client
session, each constructing its own `_LazyDB` with its own SQLite
handle. Multiple instances racing to checkpoint+close the same WAL on
process exit corrupted the on-disk DB. The fix pins one `_LazyDB`
per-process, registers a single atexit close, and de-duplicates the
startup safety-warning emit so reconnects don't spam the journal.
"""

from __future__ import annotations

import atexit
import importlib
import threading


def _reload_server():
    """Fresh import so module-level singletons reset between tests."""
    from yantrikdb_mcp import server
    return importlib.reload(server)


def test_singleton_is_same_across_calls():
    server = _reload_server()
    a = server._get_lazy_singleton()
    b = server._get_lazy_singleton()
    assert a is b, "every caller must see the same _LazyDB instance"


def test_singleton_initialised_only_once_under_threads():
    server = _reload_server()
    results = []
    barrier = threading.Barrier(8)

    def race():
        barrier.wait()
        results.append(server._get_lazy_singleton())

    threads = [threading.Thread(target=race) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len({id(r) for r in results}) == 1, (
        "double-checked locking must yield a single instance under contention"
    )


def test_atexit_close_is_registered(monkeypatch):
    """The first `_get_lazy_singleton()` call must register the close
    callback exactly once. We can't assert on Python's atexit list
    directly (private), so we monkeypatch atexit.register and verify."""
    server = _reload_server()
    registered = []
    monkeypatch.setattr(atexit, "register", lambda fn: registered.append(fn) or fn)

    s1 = server._get_lazy_singleton()
    s2 = server._get_lazy_singleton()
    assert s1 is s2
    assert len(registered) == 1, "atexit.register should fire exactly once"
    # And the callable should be the singleton's close method
    assert registered[0] == s1.close


def test_close_is_idempotent():
    server = _reload_server()
    s = server._get_lazy_singleton()

    # Stub the inner _db so close() actually has something to close
    closes = []
    class _FakeDB:
        def close(self): closes.append("called")
    s._db = _FakeDB()

    s.close()
    s.close()  # second call must be a no-op, no AttributeError
    s.close()
    assert closes == ["called"], "engine close should fire exactly once"
    assert s._db is None, "_db must be nulled after close"


def test_close_tolerates_engine_without_close_method():
    """HTTP backend or older engines may not expose close(); the
    wrapper should swallow AttributeError so atexit can run cleanly
    in cluster mode."""
    server = _reload_server()
    s = server._get_lazy_singleton()
    s._db = object()  # bare object — no .close() attribute
    s.close()  # must not raise
    assert s._db is None


def test_safety_warnings_emitted_once_per_process(monkeypatch):
    """Under SSE the lifespan re-enters per session — startup safety
    warnings must NOT re-fire on every reconnect."""
    server = _reload_server()
    calls = []
    monkeypatch.setattr(
        "yantrikdb_mcp.skill_security.startup_safety_checks",
        lambda **kw: (calls.append(kw) or ["[F.test] dummy warning"]),
    )
    monkeypatch.setattr(
        "yantrikdb_mcp.skill_security.audit_event",
        lambda evt: calls.append(evt),
    )

    # Simulate multiple lifespan entries (what SSE does on reconnect)
    server._emit_skill_safety_warnings()
    server._emit_skill_safety_warnings()
    server._emit_skill_safety_warnings()

    # Exactly one safety-check probe + one audit event from the first call;
    # subsequent calls short-circuit on the de-dupe flag.
    probe_calls = [c for c in calls if isinstance(c, dict) and "is_cluster_mode" in c]
    audit_calls = [c for c in calls if isinstance(c, dict) and c.get("event") == "startup_safety_warnings"]
    assert len(probe_calls) == 1, f"safety check should fire once, fired {len(probe_calls)}"
    assert len(audit_calls) == 1, f"audit emit should fire once, fired {len(audit_calls)}"
