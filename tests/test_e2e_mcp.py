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
