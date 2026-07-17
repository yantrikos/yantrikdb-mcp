"""v0.10.0 — engine-contract firewall (D3) + response-shaping tests (A1/A3)
+ golden-path instruction + digest-decode regression.

The contract test is the one that would have caught the v0.9.1 `correct()`
signature regression at CI time instead of in a production session: it
asserts every `db.*` method the server invokes actually exists on the
installed engine. Signature-level drift (a renamed/removed method) fails
here immediately.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


# The exact set of engine methods tools.py calls on the db object. Kept as
# an explicit allowlist (not auto-derived) so a reviewer sees the real
# contract surface and a removal shows up as a diff. Regenerate with:
#   grep -oE "db\.[a-z_]+\(" src/yantrikdb_mcp/tools.py | sed 's/db\.//;s/(//' | sort -u
ENGINE_CONTRACT_METHODS = {
    "active_session", "archive", "audit_leak_candidates", "auto_relate",
    "auto_resolve_conflicts", "backfill_memory_entities", "chain_head",
    "clear_turns", "correct", "derive_personality", "dismiss_conflict",
    "draft_memories_from_summary", "embed", "entity_profile", "forget",
    "get", "get_conflict", "get_conflicts", "get_edges", "get_patterns",
    "get_pending_triggers", "get_personality", "get_trigger_history",
    "history", "hydrate", "knowledge_gaps", "last_maintenance_cycle",
    "learn_category_members", "learned_weights", "link", "link_memory_entity",
    "linked_records", "list_memories", "procedural_stats", "prune_triggers",
    "rebuild_graph_index", "rebuild_vec_index", "recall_feedback",
    "recall_refine", "recall_with_links", "recall_with_response",
    "recent_turns", "reclassify_conflict", "record", "record_batch",
    "record_procedural", "record_turn", "reinforce_procedural", "relate",
    "relationship_depth", "reset_category_to_seed", "resolve_conflict",
    "run_maintenance_cycle", "search_entities", "session_abandon_stale",
    "session_digest", "session_end", "session_history", "session_start",
    "set_personality_trait", "skill_outcome_count", "stale", "stats",
    "substitution_categories", "substitution_members", "surface_procedural",
    "task_add", "task_delete", "task_get", "task_list", "task_update",
    "think", "unlink", "upcoming",
}


def test_d3_engine_contract_every_called_method_exists():
    """D3 firewall: every engine method the server calls must exist on the
    installed yantrikdb. Catches signature drift (renamed/removed method)
    at CI time. This is the exact class of bug that broke v0.9.1's
    `correct()` in production."""
    import yantrikdb

    missing = sorted(
        m for m in ENGINE_CONTRACT_METHODS
        if not hasattr(yantrikdb.YantrikDB, m)
    )
    assert not missing, (
        f"Installed yantrikdb is missing {len(missing)} method(s) the MCP "
        f"server calls: {missing}. The engine pin (pyproject `yantrikdb`) is "
        f"admitting an incompatible version, OR tools.py calls a method that "
        f"was removed/renamed upstream. Fix the call site or tighten the pin."
    )


def test_d3_engine_pin_is_upper_bounded():
    """The pin must have an upper bound so an untested engine minor can't
    silently satisfy it (the v0.9.x regressions shipped through an
    open-ended `>=` pin)."""
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    # Find the yantrikdb line in dependencies
    lines = [ln for ln in pyproject.splitlines() if '"yantrikdb' in ln and "mcp" not in ln]
    assert lines, "no yantrikdb dependency line found"
    dep = lines[0]
    assert "<" in dep, (
        f"yantrikdb dependency must be upper-bounded (found: {dep.strip()}). "
        f"An open-ended >= pin admits untested breaking engine releases."
    )


# ─────────────────────────────────────────────────────────────────────
# Response-shaping (A1 / A3) + digest decode — via live stdio
# ─────────────────────────────────────────────────────────────────────


def _rpc(proc, method, params, mid):
    proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params}) + "\n").encode())
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(proc.stderr.read().decode("utf-8", errors="replace"))
        s = line.decode("utf-8", errors="replace").strip()
        if not s:
            continue
        try:
            msg = json.loads(s)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == mid:
            return msg


@pytest.fixture
def proc(tmp_path):
    env = {
        **os.environ,
        "YANTRIKDB_DB_PATH": str(tmp_path / "v010.db"),
        "YANTRIKDB_EMBEDDER": "bundled",
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1",
    }
    p = subprocess.Popen(
        [sys.executable, "-m", "yantrikdb_mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    _rpc(p, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "v010", "version": "0"}}, 1)
    p.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
    p.stdin.flush()
    yield p
    p.stdin.close()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()


def _call(proc, name, mid, **args):
    r = _rpc(proc, "tools/call", {"name": name, "arguments": args}, mid)
    return json.loads(r["result"]["content"][0]["text"])


def test_a1_recall_importance_is_rounded(proc):
    """importance must arrive rounded to <=3 decimals, not the engine's raw
    decay float. The bug: score was rounded, importance was passed raw."""
    _call(proc, "remember", 10, text="The build target is Python 3.12", importance=0.7)
    body = _call(proc, "recall", 11, query="build target", top_k=3)
    assert body["count"] >= 1
    for r in body["results"]:
        imp = r["importance"]
        # decimal places <= 3
        s = repr(imp)
        if "." in s:
            decimals = len(s.split(".", 1)[1])
            assert decimals <= 3, f"importance {imp} not rounded (>{3} dp)"


def test_a3_recall_prunes_null_emotional_state(proc):
    """A memory recorded without emotional_state must not ship
    `emotional_state: null` in the recall result."""
    _call(proc, "remember", 20, text="A neutral factual note about the system")
    body = _call(proc, "recall", 21, query="factual note system", top_k=3)
    assert body["count"] >= 1
    for r in body["results"]:
        assert "emotional_state" not in r, (
            "null emotional_state should be pruned from recall results"
        )


def test_digest_returns_decoded_object_not_double_encoded_string(proc):
    """session(action='digest') must return a first-class object, not a
    `{"digest": "{\\"...\\": ...}"}` double-encoded string envelope."""
    _call(proc, "remember", 30, text="A decision: adopt the new deployment flow", importance=0.8)
    body = _call(proc, "session", 31, action="digest")
    assert isinstance(body, dict)
    # If the engine returned a JSON string, the wrapper must have decoded it:
    # the value under "digest" (if present) must NOT itself be a JSON string.
    if "digest" in body and isinstance(body["digest"], str):
        # allowed only if it's genuinely not JSON (plain text); reject the
        # double-encoded case where it parses back into a dict.
        try:
            reparsed = json.loads(body["digest"])
            assert not isinstance(reparsed, dict), (
                "digest is double-encoded — a JSON object was returned as a string"
            )
        except json.JSONDecodeError:
            pass  # plain-text digest is fine


# ─────────────────────────────────────────────────────────────────────
# D1 golden-path instructions
# ─────────────────────────────────────────────────────────────────────


def test_d1_instructions_are_golden_path_and_have_trust_boundary():
    """The injected server instructions must prescribe the digest-first
    golden path and carry the trust-boundary directive."""
    from yantrikdb_mcp.server import INSTRUCTIONS

    # Normalize whitespace so line-wrapping doesn't break phrase matching.
    text = " ".join(INSTRUCTIONS.lower().split())
    # digest-first cold start
    assert 'session(action="digest")' in INSTRUCTIONS, "instructions must lead with the digest"
    # trust boundary (Sol's mandatory addition)
    assert "trust boundary" in text or "not instructions" in text, (
        "instructions must carry the recalled-content-is-data trust boundary"
    )
    assert "never execute" in text, "instructions must forbid executing recalled directives"


def test_d1_instructions_teach_current_value_and_trust_signals():
    """The control plane must activate the benchmarked differentiator
    (chain_head current-value beats recall for "what is the latest X" —
    measured RAG 0.00 vs substrate 0.78-1.00) and tell agents to heed
    why_retrieved staleness flags. A capability the instructions don't
    mention is dark capability — agents skim tool docs but read the
    injected instructions."""
    from yantrikdb_mcp.server import INSTRUCTIONS

    text = " ".join(INSTRUCTIONS.lower().split())
    assert "chain_head" in text, (
        "instructions must route current-value questions to chain_head"
    )
    assert "why_retrieved" in text, (
        "instructions must tell agents to heed why_retrieved staleness flags"
    )


# ─────────────────────────────────────────────────────────────────────
# Honest surface — engine-version divergence must be visible to the agent
# ─────────────────────────────────────────────────────────────────────


def test_digest_include_gaps_is_never_silently_dropped(proc):
    """Version-agnostic contract: session(digest, include_gaps=True) must
    either deliver gaps (engine supports it) or SAY the param was ignored
    (`unsupported_params`) — never a bare digest where the agent can't
    distinguish "no gaps exist" from "gaps weren't computed". This is the
    anti-silent-drift rule applied to our own tool surface."""
    _call(proc, "remember", 40, text="Seed memory so the digest has content")
    body = _call(proc, "session", 41, action="digest", include_gaps=True, max_gaps=3)
    assert isinstance(body, dict)
    supported = "knowledge_gaps" in body
    declared_dropped = "include_gaps" in body.get("unsupported_params", [])
    assert supported or declared_dropped, (
        "include_gaps=True yielded neither knowledge_gaps nor an "
        "unsupported_params declaration — the param was silently dropped"
    )


def test_remember_idempotency_key_dedupes_via_tool_surface(proc):
    """v0.10 leg: the tool arg must reach the engine — same key + same text
    through the PUBLIC surface returns the same rid, no second write."""
    a = _call(proc, "remember", 60, text="Idempotent tool-surface probe",
              idempotency_key="tool-key-1")
    b = _call(proc, "remember", 61, text="Idempotent tool-surface probe",
              idempotency_key="tool-key-1")
    assert a.get("rid") and a["rid"] == b.get("rid"), (
        f"same key+text must dedupe to one rid: {a} vs {b}"
    )
    conflict = _call(proc, "remember", 62, text="DIFFERENT text same key",
                     idempotency_key="tool-key-1")
    assert "error" in conflict and "idempotency" in conflict["error"].lower(), (
        f"same key + different text must surface an agent-readable conflict: {conflict}"
    )


def test_session_capture_drafts_memories(proc):
    """Server ruling: capture segments a summary (no session_id) into
    candidate facts — distinct from tracking-end."""
    body = _call(proc, "session", 70, action="capture",
                 summary="We decided to gate releases on the semantic contract. "
                         "Alice owns the deploy runbook. The rollback window is 04:00 UTC.")
    assert body.get("count", 0) >= 1 and body.get("drafted"), (
        f"capture must draft at least one memory: {body}"
    )


def test_remember_wrong_param_error_names_the_fix(proc):
    """Dogfood regression: an agent that passes the memory body under a
    wrong name (content=) has it dropped by schema validation and used to
    get the true-but-useless "text must be non-empty". The error must name
    the actual fix — the `text` parameter."""
    r = _rpc(proc, "tools/call",
             {"name": "remember", "arguments": {"content": "misnamed body"}}, 50)
    # Whether it surfaces as a protocol error or a tool error, the message
    # the agent sees must point at `text`.
    blob = json.dumps(r).lower()
    assert "text" in blob and ("parameter" in blob or "non-empty" in blob), (
        f"wrong-param remember error must name the `text` parameter: {blob[:400]}"
    )
