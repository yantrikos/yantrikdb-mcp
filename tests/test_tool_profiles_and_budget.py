"""Tool profiles (core/full) + the schema budget gate.

BUDGET GATE (sol-converged): tools/list schema is a fixed per-session cost
paid by every client before any call. This test freezes it — growth fails CI
until the budget literal below is raised in the same diff, which IS the
review. On failure it prints per-tool attribution so the offender is obvious.

PROFILE GATE: `core` must advertise exactly the 10 golden-path tools the
INSTRUCTIONS choreography uses — intact tools, no aliases. `full` (default)
must advertise all 19 so no existing deployment silently loses surface.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

CORE_TOOLS = {
    "remember", "recall", "correct", "forget", "session",
    "memory", "think", "graph", "conflict", "procedure",
}
FULL_TOOLS = CORE_TOOLS | {
    "temporal", "category", "personality", "trigger", "stats",
    "conversation", "task", "gaps", "skill",
}

# ── budget ───────────────────────────────────────────────────────────
# Raising this number is allowed ONLY as a reviewed, explained diff.
# Baseline set 2026-07-17 on the v0.10.0 branch after the capability-
# activation guidance (chain_head / why_retrieved / include_gaps) and the
# v0.10 args (idempotency_key, include_superseded, capture) landed.
SCHEMA_BUDGET_CHARS = 43_000


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


def _list_tools(profile: str | None, tmp_path):
    env = {
        **os.environ,
        "YANTRIKDB_DB_PATH": str(tmp_path / "prof.db"),
        "YANTRIKDB_EMBEDDER": "bundled",
        "PYTHONIOENCODING": "utf-8",
    }
    env.pop("YANTRIKDB_TOOL_PROFILE", None)
    if profile is not None:
        env["YANTRIKDB_TOOL_PROFILE"] = profile
    p = subprocess.Popen(
        [sys.executable, "-m", "yantrikdb_mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    try:
        _rpc(p, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "profiles", "version": "0"}}, 1)
        p.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        p.stdin.flush()
        r = _rpc(p, "tools/list", {}, 2)
        return r["result"]["tools"]
    finally:
        p.stdin.close()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def test_full_profile_is_default_and_complete(tmp_path):
    names = {t["name"] for t in _list_tools(None, tmp_path)}
    assert names == FULL_TOOLS, (
        f"default (full) profile drifted: missing={FULL_TOOLS - names}, "
        f"unexpected={names - FULL_TOOLS}"
    )


def test_core_profile_is_exactly_the_golden_path(tmp_path):
    names = {t["name"] for t in _list_tools("core", tmp_path)}
    assert names == CORE_TOOLS, (
        f"core profile drifted: missing={CORE_TOOLS - names}, "
        f"unexpected={names - CORE_TOOLS}"
    )


def test_unknown_profile_falls_back_to_full(tmp_path):
    names = {t["name"] for t in _list_tools("experimental-nonsense", tmp_path)}
    assert names == FULL_TOOLS, "unknown profile must fail OPEN to full, with a warning"


def test_schema_budget_is_frozen(tmp_path):
    tools = _list_tools(None, tmp_path)
    sizes = []
    total = 0
    for t in tools:
        blob = json.dumps({"name": t["name"], "description": t.get("description"),
                           "inputSchema": t.get("inputSchema")})
        sizes.append((len(blob), t["name"]))
        total += len(blob)
    attribution = "\n".join(f"  {n:>7,}  {name}" for n, name in sorted(sizes, reverse=True))
    assert total <= SCHEMA_BUDGET_CHARS, (
        f"tools/list schema is {total:,} chars — over the {SCHEMA_BUDGET_CHARS:,} "
        f"budget. Every char here is paid per-session by EVERY client. Either "
        f"trim the growth or raise SCHEMA_BUDGET_CHARS in this diff with a "
        f"one-line justification.\nPer-tool attribution:\n{attribution}"
    )
