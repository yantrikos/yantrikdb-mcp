"""End-to-end MCP-over-stdio smoke test.

Spawns the real `yantrikdb-mcp` console entrypoint, completes the JSON-RPC
handshake, calls `tools/list` + a record→recall round-trip, and verifies
the protocol surface works against a fresh database with the bundled
embedder. CI runs this on every push so we never publish a broken wheel.

Test matrix (parametrized):
- Bundled embedder (default, no extras)
- ONNX embedder (skipped if ONNX deps not installed)
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
# JSON-RPC stdio framing helpers
# ─────────────────────────────────────────────────────────────────────


def _read_message(proc: subprocess.Popen, timeout: float = 30.0) -> dict:
    """Read one MCP message from the server's stdout (newline-delimited JSON)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(
                    f"server exited (code={proc.returncode}) before sending a message.\nstderr:\n{stderr}"
                )
            time.sleep(0.05)
            continue
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Probably a stderr-on-stdout warning; ignore.
            continue
    raise TimeoutError(f"no MCP message in {timeout}s")


def _send(proc: subprocess.Popen, msg: dict) -> None:
    encoded = (json.dumps(msg) + "\n").encode("utf-8")
    proc.stdin.write(encoded)
    proc.stdin.flush()


def _rpc(proc: subprocess.Popen, method: str, params: dict, msg_id: int) -> dict:
    _send(proc, {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params})
    while True:
        msg = _read_message(proc)
        # Skip notifications / progress messages without an id match
        if msg.get("id") == msg_id:
            return msg


# ─────────────────────────────────────────────────────────────────────
# Backend matrix
# ─────────────────────────────────────────────────────────────────────


def _onnx_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
        return True
    except ImportError:
        return False


def _multilingual_supported() -> bool:
    """The engine's set_embedder_named registry lives in compiled Rust;
    just probe whether the API itself exists. We don't try to predict
    whether the network fetch will succeed — CI either has connectivity
    or it doesn't, and the test will fail loudly if extraction breaks.
    """
    try:
        from yantrikdb import YantrikDB
        return hasattr(YantrikDB, "with_default") and hasattr(YantrikDB, "set_embedder_named")
    except ImportError:
        return False


BACKENDS = ["bundled"]
if _onnx_available():
    BACKENDS.append("onnx")
# Multilingual is gated separately because the engine downloads a ~460 MB
# model on first use. CI must opt in via YANTRIKDB_TEST_MULTILINGUAL=1 so
# bog-standard PR runs don't pull half a GB.
if _multilingual_supported() and os.environ.get("YANTRIKDB_TEST_MULTILINGUAL") == "1":
    BACKENDS.append("multilingual")


# ─────────────────────────────────────────────────────────────────────
# E2E test
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_mcp_stdio_record_recall_roundtrip(tmp_path: Path, backend: str):
    """Spawn the real entrypoint, do a record → recall round-trip via JSON-RPC."""
    db_path = tmp_path / f"e2e_{backend}.db"

    env = {
        **os.environ,
        "YANTRIKDB_DB_PATH": str(db_path),
        "YANTRIKDB_EMBEDDER": backend,
        "PYTHONIOENCODING": "utf-8",
        # Stop huggingface-hub from contacting the network on cold start.
        "HF_HUB_OFFLINE": "1",
    }

    # Use the module entrypoint so we don't depend on the .exe being on PATH.
    proc = subprocess.Popen(
        [sys.executable, "-m", "yantrikdb_mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    try:
        # 1) initialize handshake
        init = _rpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e2e-test", "version": "0.0.0"},
        }, msg_id=1)
        assert init.get("result"), f"initialize failed: {init}"
        assert init["result"]["serverInfo"]["name"] == "yantrikdb"

        # initialized notification (no id, no response)
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 2) tools/list — must include remember + recall
        tools_resp = _rpc(proc, "tools/list", {}, msg_id=2)
        assert "result" in tools_resp, f"tools/list failed: {tools_resp}"
        tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
        assert "remember" in tool_names
        assert "recall" in tool_names

        # 3) remember — write a memory
        remember_resp = _rpc(proc, "tools/call", {
            "name": "remember",
            "arguments": {
                "text": "E2E test sentinel: the leopard prowls the savanna",
                "importance": 0.9,
                "domain": "general",
            },
        }, msg_id=3)
        assert "result" in remember_resp, f"remember failed: {remember_resp}"
        rcontent = remember_resp["result"]["content"][0]["text"]
        assert "rid" in rcontent, f"remember response missing rid: {rcontent}"

        # 4) recall — read it back
        recall_resp = _rpc(proc, "tools/call", {
            "name": "recall",
            "arguments": {"query": "leopard savanna prowling", "top_k": 5},
        }, msg_id=4)
        assert "result" in recall_resp, f"recall failed: {recall_resp}"
        recall_text = recall_resp["result"]["content"][0]["text"]
        recall_obj = json.loads(recall_text)
        # Both bundled (64-dim) and ONNX (384-dim) should hit on lexical/semantic overlap
        assert recall_obj.get("count", 0) >= 1, (
            f"recall returned no hits for backend={backend}: {recall_obj}"
        )
        # Sanity: the sentinel phrase appears in at least one result
        results = recall_obj.get("results", [])
        any_hit = any("leopard" in r.get("text", "").lower() for r in results)
        assert any_hit, f"recall didn't surface the sentinel for backend={backend}: {results}"

    finally:
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


def test_e2e_backends_include_bundled():
    """Sanity: bundled is always tested (no extras required)."""
    assert "bundled" in BACKENDS


def test_skill_define_surface_outcome_roundtrip(tmp_path: Path):
    """Spawn the real entrypoint with the bundled backend, drive the
    `skill` tool through a full define → surface → outcome → get → list
    JSON-RPC round-trip. Bundled-only to keep CI cheap; the validator
    layer is exercised across all backends by unit tests.

    Sets `YANTRIKDB_SKILLS_WRITE_ENABLED=true` because skill writes are
    off by default — see `test_skill_writes_disabled_by_default` below
    for the gate behavior.
    """
    db_path = tmp_path / "e2e_skills.db"

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

    try:
        init = _rpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "skill-e2e", "version": "0.0.0"},
        }, msg_id=1)
        assert init.get("result"), f"initialize failed: {init}"
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        tools_resp = _rpc(proc, "tools/list", {}, msg_id=2)
        tool_names = {t["name"] for t in tools_resp["result"]["tools"]}
        assert "skill" in tool_names, "skill tool not registered"

        # 1) define — happy path
        define_resp = _rpc(proc, "tools/call", {
            "name": "skill",
            "arguments": {
                "action": "define",
                "skill_id": "workflow.git.commit_clean",
                "body": (
                    "Before commit: run pytest, run lint, write a clear "
                    "subject and body. Never include co-authored-by unless asked."
                ),
                "skill_type": "procedure",
                "applies_to": ["git", "release"],
            },
        }, msg_id=3)
        assert "result" in define_resp, f"define failed: {define_resp}"
        define_obj = json.loads(define_resp["result"]["content"][0]["text"])
        assert define_obj.get("stored") is True
        assert define_obj["skill_id"] == "workflow.git.commit_clean"

        # 2) define — invalid skill_id (hyphen) must fail loudly
        bad_resp = _rpc(proc, "tools/call", {
            "name": "skill",
            "arguments": {
                "action": "define",
                "skill_id": "workflow-git.bad",
                "body": "x" * 60,
                "skill_type": "procedure",
                "applies_to": ["git"],
            },
        }, msg_id=4)
        # ToolError surfaces as isError=True on the result envelope
        assert bad_resp.get("result", {}).get("isError") is True \
            or "error" in bad_resp, f"hyphen skill_id should fail: {bad_resp}"

        # 3) surface — should find the skill we just defined
        surface_resp = _rpc(proc, "tools/call", {
            "name": "skill",
            "arguments": {
                "action": "surface",
                "query": "how to commit code cleanly",
                "top_k": 5,
            },
        }, msg_id=5)
        assert "result" in surface_resp, f"surface failed: {surface_resp}"
        surface_obj = json.loads(surface_resp["result"]["content"][0]["text"])
        assert surface_obj.get("count", 0) >= 1, f"surface returned nothing: {surface_obj}"
        skill_ids_found = [r["skill_id"] for r in surface_obj["results"]]
        assert "workflow.git.commit_clean" in skill_ids_found, (
            f"surfaced skills missing the defined one: {skill_ids_found}"
        )

        # 4) outcome — append a success event
        outcome_resp = _rpc(proc, "tools/call", {
            "name": "skill",
            "arguments": {
                "action": "outcome",
                "skill_id": "workflow.git.commit_clean",
                "succeeded": True,
                "note": "caught a flake8 issue pre-push",
            },
        }, msg_id=6)
        assert "result" in outcome_resp, f"outcome failed: {outcome_resp}"
        outcome_obj = json.loads(outcome_resp["result"]["content"][0]["text"])
        assert outcome_obj.get("recorded") is True

        # 5) get — fetch by id
        get_resp = _rpc(proc, "tools/call", {
            "name": "skill",
            "arguments": {"action": "get", "skill_id": "workflow.git.commit_clean"},
        }, msg_id=7)
        get_obj = json.loads(get_resp["result"]["content"][0]["text"])
        assert get_obj.get("skill_id") == "workflow.git.commit_clean"
        assert "applies_to" in get_obj

        # 6) list — catalog browse
        list_resp = _rpc(proc, "tools/call", {
            "name": "skill",
            "arguments": {"action": "list", "limit": 10},
        }, msg_id=8)
        list_obj = json.loads(list_resp["result"]["content"][0]["text"])
        assert list_obj.get("count", 0) >= 1
        assert any(
            r["skill_id"] == "workflow.git.commit_clean" for r in list_obj["results"]
        )

    finally:
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


def test_skill_writes_disabled_by_default(tmp_path: Path):
    """Without `YANTRIKDB_SKILLS_WRITE_ENABLED`, the `define` and `outcome`
    actions must be refused with a clear error. Reads must still work."""
    db_path = tmp_path / "e2e_skills_gated.db"

    # Strip the gate from the parent env so the subprocess gets default-off
    # even if our test runner happens to have it set.
    env = {
        k: v for k, v in os.environ.items()
        if k != "YANTRIKDB_SKILLS_WRITE_ENABLED"
    }
    env.update({
        "YANTRIKDB_DB_PATH": str(db_path),
        "YANTRIKDB_EMBEDDER": "bundled",
        "PYTHONIOENCODING": "utf-8",
        "HF_HUB_OFFLINE": "1",
    })

    proc = subprocess.Popen(
        [sys.executable, "-m", "yantrikdb_mcp"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )

    try:
        _rpc(proc, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "skill-gate-e2e", "version": "0.0.0"},
        }, msg_id=1)
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        # define is gated
        define_resp = _rpc(proc, "tools/call", {
            "name": "skill",
            "arguments": {
                "action": "define",
                "skill_id": "workflow.git.commit_clean",
                "body": "x" * 60,
                "skill_type": "procedure",
                "applies_to": ["git"],
            },
        }, msg_id=2)
        is_error = define_resp.get("result", {}).get("isError")
        body = json.dumps(define_resp)
        assert is_error is True or "YANTRIKDB_SKILLS_WRITE_ENABLED" in body, (
            f"define should be gated by default: {define_resp}"
        )

        # outcome is also gated
        outcome_resp = _rpc(proc, "tools/call", {
            "name": "skill",
            "arguments": {
                "action": "outcome",
                "skill_id": "workflow.git.commit_clean",
                "succeeded": True,
            },
        }, msg_id=3)
        is_error = outcome_resp.get("result", {}).get("isError")
        body = json.dumps(outcome_resp)
        assert is_error is True or "YANTRIKDB_SKILLS_WRITE_ENABLED" in body, (
            f"outcome should be gated by default: {outcome_resp}"
        )

        # Reads ARE allowed even when the gate is off — they only return
        # skills someone authorized into the substrate. List should return
        # an empty result, not an error.
        list_resp = _rpc(proc, "tools/call", {
            "name": "skill",
            "arguments": {"action": "list", "limit": 10},
        }, msg_id=4)
        assert "result" in list_resp, f"list failed when gate off: {list_resp}"
        list_obj = json.loads(list_resp["result"]["content"][0]["text"])
        assert list_obj.get("count", -1) == 0, (
            f"list with gate off and no skills should be empty: {list_obj}"
        )

    finally:
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
