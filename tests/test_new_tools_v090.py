"""Integration tests for v0.9.0 new MCP tools: gaps, conversation, task,
session(action=digest), and the Tier-2 action additions.

Each test spawns the real MCP entrypoint via stdio JSON-RPC against a
fresh bundled-embedder DB, exercising the new engine surface end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────
# JSON-RPC stdio helpers (mirror test_e2e_mcp's framing)
# ─────────────────────────────────────────────────────────────────────


def _read_message(proc: subprocess.Popen, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else b""
                raise RuntimeError(
                    f"server exited (code={proc.returncode}) before sending a message.\n"
                    f"stderr:\n{stderr.decode('utf-8', errors='replace')}"
                )
            time.sleep(0.05)
            continue
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    raise TimeoutError(f"no MCP message in {timeout}s")


def _send(proc, msg):
    proc.stdin.write((json.dumps(msg) + "\n").encode("utf-8"))
    proc.stdin.flush()


def _rpc(proc, method, params, msg_id):
    _send(proc, {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
    while True:
        msg = _read_message(proc)
        if msg.get("id") == msg_id:
            return msg


# ─────────────────────────────────────────────────────────────────────
# Fixture: live MCP process bound to a fresh per-test DB
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def mcp_proc(tmp_path):
    db_path = tmp_path / "v090.db"
    env = {
        **os.environ,
        "YANTRIKDB_DB_PATH": str(db_path),
        "YANTRIKDB_EMBEDDER": "bundled",
        "YANTRIKDB_SKILLS_WRITE_ENABLED": "true",
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "yantrikdb_mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )
    # Handshake
    init = _rpc(proc, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "v090-test", "version": "0.0"},
    }, msg_id=1)
    assert init.get("result"), f"initialize failed: {init}"
    _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

    yield proc

    try:
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/cancelled"})
    except Exception:
        pass
    proc.stdin.close()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# Each test gets its own msg_id counter
def _call_tool(proc, name, mid, **args):
    return _rpc(proc, "tools/call", {"name": name, "arguments": args}, mid)


def _unwrap(resp):
    """Extract the parsed JSON body from a tools/call response."""
    text = resp["result"]["content"][0]["text"]
    try:
        return resp["result"].get("isError", False), json.loads(text)
    except json.JSONDecodeError:
        return resp["result"].get("isError", False), text


# ─────────────────────────────────────────────────────────────────────
# tools/list — confirm new tools are registered
# ─────────────────────────────────────────────────────────────────────


def test_v090_tools_registered(mcp_proc):
    resp = _rpc(mcp_proc, "tools/list", {}, msg_id=10)
    names = {t["name"] for t in resp["result"]["tools"]}
    # Tier 1 new tools
    assert "gaps" in names, "gaps tool not registered"
    assert "conversation" in names, "conversation tool not registered"
    assert "task" in names, "task tool not registered"
    # Pre-existing tools still here
    assert "session" in names
    assert "remember" in names
    assert "skill" in names
    # Total should be 19 (16 in v0.8.x + gaps + conversation + task)
    assert len(names) == 19, f"expected 19 tools, got {len(names)}: {sorted(names)}"


# ─────────────────────────────────────────────────────────────────────
# gaps tool
# ─────────────────────────────────────────────────────────────────────


def test_gaps_returns_empty_on_fresh_db(mcp_proc):
    is_err, body = _unwrap(_call_tool(mcp_proc, "gaps", 20))
    assert not is_err
    assert body["count"] == 0
    assert body["gaps"] == []


# ─────────────────────────────────────────────────────────────────────
# conversation tool
# ─────────────────────────────────────────────────────────────────────


def test_conversation_record_recent_clear_cycle(mcp_proc):
    # Record three turns
    for i, (role, content) in enumerate([
        ("user", "What's the weather?"),
        ("assistant", "Sunny and 72."),
        ("user", "Thanks."),
    ]):
        is_err, body = _unwrap(_call_tool(
            mcp_proc, "conversation", 30 + i,
            action="record", namespace="weather_chat",
            role=role, content=content,
        ))
        assert not is_err, f"record failed: {body}"
        assert body["recorded"] is True

    # Recent: should return all 3
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "conversation", 40,
        action="recent", namespace="weather_chat", limit=10,
    ))
    assert not is_err
    assert body["count"] == 3
    assert len(body["turns"]) == 3

    # Clear
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "conversation", 41,
        action="clear", namespace="weather_chat",
    ))
    assert not is_err
    assert body["removed"] >= 3

    # Recent after clear: empty
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "conversation", 42,
        action="recent", namespace="weather_chat",
    ))
    assert not is_err
    assert body["count"] == 0


def test_conversation_record_requires_role_and_content(mcp_proc):
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "conversation", 50,
        action="record", namespace="x",
    ))
    assert is_err
    assert "role" in str(body).lower() and "content" in str(body).lower()


def test_conversation_namespace_isolation(mcp_proc):
    """Two separate namespaces must NOT share their ring buffers."""
    _call_tool(mcp_proc, "conversation", 60, action="record",
                namespace="ns_a", role="user", content="from A")
    _call_tool(mcp_proc, "conversation", 61, action="record",
                namespace="ns_b", role="user", content="from B")
    _, body_a = _unwrap(_call_tool(mcp_proc, "conversation", 62,
                                     action="recent", namespace="ns_a"))
    _, body_b = _unwrap(_call_tool(mcp_proc, "conversation", 63,
                                     action="recent", namespace="ns_b"))
    assert body_a["count"] == 1
    assert body_b["count"] == 1
    assert body_a["turns"][0]["content"] == "from A"
    assert body_b["turns"][0]["content"] == "from B"


# ─────────────────────────────────────────────────────────────────────
# task tool
# ─────────────────────────────────────────────────────────────────────


def test_task_full_crud_cycle(mcp_proc):
    # Add
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "task", 70,
        action="add", namespace="release_v090",
        title="Ship yantrikdb-mcp v0.9.0", priority="high",
    ))
    assert not is_err, f"add failed: {body}"
    task_id = body["task_id"]
    assert task_id
    assert body["priority"] == "high"

    # Get
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "task", 71, action="get", task_id=task_id,
    ))
    assert not is_err
    assert body.get("id") == task_id or body.get("task_id") == task_id

    # List
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "task", 72, action="list", namespace="release_v090",
    ))
    assert not is_err
    assert body["count"] >= 1

    # Update to done
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "task", 73, action="update", task_id=task_id, status="done",
    ))
    assert not is_err
    assert body["updated"] is True

    # Filter list by status
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "task", 74, action="list",
        namespace="release_v090", status="done",
    ))
    assert not is_err
    assert body["count"] >= 1

    # Delete
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "task", 75, action="delete", task_id=task_id,
    ))
    assert not is_err
    assert body["removed"] is True


def test_task_add_requires_title(mcp_proc):
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "task", 80, action="add", namespace="x",
    ))
    assert is_err
    assert "title" in str(body).lower()


def test_task_get_unknown_returns_error_envelope(mcp_proc):
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "task", 81, action="get", task_id="does-not-exist",
    ))
    # Soft error — not a ToolError, just {"error": "...not found"}
    assert "error" in body
    assert "not found" in body["error"].lower()


# ─────────────────────────────────────────────────────────────────────
# session(action="digest")
# ─────────────────────────────────────────────────────────────────────


def test_session_digest_returns_envelope_on_fresh_db(mcp_proc):
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "session", 90, action="digest",
    ))
    assert not is_err
    # Engine returns a dict with the digest fields; we just confirm shape
    assert isinstance(body, dict)


# ─────────────────────────────────────────────────────────────────────
# Tier-2: think(maintenance_cycle=True) + think(last_cycle_only=True)
# ─────────────────────────────────────────────────────────────────────


def test_think_maintenance_cycle_dry_run(mcp_proc):
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "think", 100, maintenance_cycle=True, dry_run=True,
    ))
    assert not is_err
    assert "maintenance_cycle" in body


def test_think_last_cycle_read_only(mcp_proc):
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "think", 101, last_cycle_only=True,
    ))
    assert not is_err
    assert "last_maintenance_cycle" in body


# ─────────────────────────────────────────────────────────────────────
# Tier-2: stats(action="audit_leak") + stats(action="skill_outcomes")
# ─────────────────────────────────────────────────────────────────────


def test_stats_audit_leak(mcp_proc):
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "stats", 110, action="audit_leak", max_rids=10,
    ))
    assert not is_err
    assert isinstance(body, dict)


def test_stats_skill_outcomes_zero_on_fresh_db(mcp_proc):
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "stats", 111, action="skill_outcomes",
    ))
    assert not is_err
    assert body["skill_outcomes_total"] == 0


# ─────────────────────────────────────────────────────────────────────
# Tier-2: remember(summary=...) — draft mode
# ─────────────────────────────────────────────────────────────────────


def test_remember_draft_from_summary(mcp_proc):
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "remember", 120,
        summary=(
            "End of session. We decided to ship v0.9.0 with the new task, "
            "conversation, and gaps tools. Pranab approved the full scope."
        ),
        namespace="release_log",
    ))
    assert not is_err
    # Body shape varies by engine; just confirm non-empty dict
    assert isinstance(body, dict)


# ─────────────────────────────────────────────────────────────────────
# Regression: correct tool signature (v0.9.1 — engine v0.7.20+ change)
# ─────────────────────────────────────────────────────────────────────


def test_correct_tool_accepts_reason_and_works_e2e(mcp_proc):
    """v0.9.1 regression pin — engine v0.7.20 (Issue #47) made `reason`
    required and removed `correction_note` on `db.correct()`. Prior to
    v0.9.1 the MCP `correct` tool passed `correction_note=` and blew up
    with `unexpected keyword argument 'correction_note'` for every user
    on engine >=0.7.20. This test would have caught it: create a
    memory, then correct it via the tool, then assert the result shape.

    NOTE: this asserts the SIGNATURE contract (reason required, no
    correction_note, in-place corrected_rid) via an importance correction,
    which is engine-stable across every version we pin. It deliberately
    does NOT correct via `new_text`, because the engine's new_text policy
    is version-dependent — released 0.9.4 REFUSES a text change (it would
    leave the memory retrieved under its old vector), while the v0.10
    engine re-embeds in place and accepts it. That 422→200 transition is a
    behavioral-contract case gated on the v0.10 engine pin (see the v0.10
    contract suite), not a signature regression, so coupling this test to
    it would make a signature guard fail on a text-mutation policy change."""
    # 1) Record a memory to correct
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "remember", 200,
        text="Python 3.11 is the target for the build.",
        domain="architecture",
    ))
    assert not is_err, f"seed remember failed: {body}"
    seed_rid = body["rid"]

    # 2) Correct with the new-signature args (reason required). Use an
    #    importance correction — exercises the exact v0.9.1 forwarding fix
    #    (reason=, no correction_note=) without depending on new_text policy.
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "correct", 201,
        rid=seed_rid,
        reason="Raised importance after 3.12 adoption became load-bearing",
        new_importance=0.9,
    ))
    assert not is_err, f"correct failed: {body}"
    # Engine v0.7.20+ mutates in-place; the response includes corrected_rid
    assert body.get("corrected_rid") is not None
    assert body.get("reason", "").startswith("Raised importance")


def test_correct_tool_reason_is_required(mcp_proc):
    """Missing reason must fail loud (schema-level rejection) before the
    engine sees the call — matches v0.7.20's contract."""
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "remember", 210,
        text="Any old memory",
    ))
    assert not is_err
    rid = body["rid"]

    is_err, body = _unwrap(_call_tool(
        mcp_proc, "correct", 211,
        rid=rid, reason="",  # empty — must reject
        new_text="Doesn't matter",
    ))
    assert is_err
    assert "reason" in str(body).lower()


def test_memory_update_importance_uses_new_correct_signature(mcp_proc):
    """The `memory(action="update_importance")` action calls `db.correct()`
    internally. Before v0.9.1 it passed `correction_note=`, which would
    have exploded on any engine >=0.7.20. This regression pins the
    action against the new signature."""
    is_err, body = _unwrap(_call_tool(
        mcp_proc, "remember", 220,
        text="A minor detail worth remembering.",
        importance=0.4,
    ))
    assert not is_err
    rid = body["rid"]

    is_err, body = _unwrap(_call_tool(
        mcp_proc, "memory", 221,
        action="update_importance", rid=rid, importance=0.9,
    ))
    assert not is_err, f"update_importance failed: {body}"
    assert body["new_importance"] == 0.9
    assert body["status"] == "updated"
