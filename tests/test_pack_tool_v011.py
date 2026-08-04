"""v0.11.0 `pack` tool — MCP-surface contract.

Exercises the tool over the real stdio JSON-RPC boundary (not by importing the
function), because the properties that matter are surface properties: is the
tool advertised at all, does the operator gate actually refuse writes, and does
a read action work without the gate.

Gated on a FEATURE PROBE of the engine, same rule as the contract suites: on a
pre-v0.11 engine `pack` is deliberately absent and this whole module skips.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


def _engine_has_packs() -> bool:
    try:
        from yantrikdb import YantrikDB
    except ImportError:
        return False
    return hasattr(YantrikDB, "install_pack") and hasattr(YantrikDB, "mounted_packs")


pytestmark = pytest.mark.skipif(
    not _engine_has_packs(), reason="engine lacks the pack substrate — pre-v0.11"
)


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


class _Server:
    """A live stdio MCP server with configurable pack-write gating."""

    def __init__(self, tmp_path, pack_writes: bool):
        env = {
            **os.environ,
            "YANTRIKDB_DB_PATH": str(tmp_path / "pack.db"),
            "YANTRIKDB_EMBEDDER": "bundled",
            "PYTHONIOENCODING": "utf-8",
        }
        env.pop("YANTRIKDB_TOOL_PROFILE", None)
        if pack_writes:
            env["YANTRIKDB_ENABLE_PACK_WRITES"] = "1"
        else:
            env.pop("YANTRIKDB_ENABLE_PACK_WRITES", None)
        self.p = subprocess.Popen(
            [sys.executable, "-m", "yantrikdb_mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        _rpc(self.p, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                    "clientInfo": {"name": "packtest", "version": "0"}}, 1)
        self.p.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        self.p.stdin.flush()
        self._id = 1

    def call(self, name, args):
        self._id += 1
        r = _rpc(self.p, "tools/call", {"name": name, "arguments": args}, self._id)
        return r

    def tools(self):
        self._id += 1
        return _rpc(self.p, "tools/list", {}, self._id)["result"]["tools"]

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


@pytest.fixture
def server(tmp_path):
    s = _Server(tmp_path, pack_writes=False)
    yield s
    s.close()


@pytest.fixture
def writable_server(tmp_path):
    s = _Server(tmp_path, pack_writes=True)
    yield s
    s.close()


def _text(resp):
    return resp["result"]["content"][0]["text"]


# ── advertisement ────────────────────────────────────────────────────


def test_pack_tool_is_advertised_on_v011_engine(server):
    assert "pack" in {t["name"] for t in server.tools()}


# ── read actions work WITHOUT the operator gate ──────────────────────


def test_list_works_without_write_gate(server):
    body = json.loads(_text(server.call("pack", {"action": "list"})))
    assert "installed" in body and "mounted" in body
    assert body["installed"] == [] and body["mounted"] == []


def test_publishers_works_without_write_gate(server):
    body = json.loads(_text(server.call("pack", {"action": "publishers"})))
    assert body["trusted_publishers"] == []


def test_embedder_identity_is_honest_when_unadopted(server):
    """A virgin db has no embedding fingerprint yet. The tool must SAY that
    rather than returning a bare null the agent would misread as an error."""
    body = json.loads(_text(server.call("pack", {"action": "embedder_identity"})))
    if body.get("embedder_identity") is None:
        assert "note" in body and "adopt" in body["note"].lower()
    else:
        assert set(("name", "digest", "dim")).issubset(body["embedder_identity"])


# ── the operator gate ────────────────────────────────────────────────


@pytest.mark.parametrize("action,args", [
    ("install", {"path": "/tmp/x.ypack"}),
    ("uninstall", {"pack_id": "acme@1.0.0"}),
    ("trust", {"pubkey": "deadbeef"}),
    ("untrust", {"pubkey": "deadbeef"}),
    ("unmount_all", {}),
])
def test_write_actions_refused_without_operator_gate(server, action, args):
    """Importing a third party's memories is an operator decision. The refusal
    must NAME the env var — an error the agent can act on beats a bare denial."""
    resp = server.call("pack", {"action": action, **args})
    blob = json.dumps(resp)
    assert "YANTRIKDB_ENABLE_PACK_WRITES" in blob, (
        f"refusal for {action!r} must name the enabling env var; got: {blob[:300]}"
    )


def test_write_actions_reachable_when_gate_enabled(writable_server):
    """With the gate on, a write action gets past the gate and reaches the
    engine (here: a real trust round-trip)."""
    body = json.loads(_text(writable_server.call(
        "pack", {"action": "trust", "pubkey": "ab" * 32, "label": "acme"})))
    assert body["trusted"] == "ab" * 32
    assert any(t["pubkey"] == "ab" * 32 for t in body["trusted_publishers"])


# ── validation + error translation ───────────────────────────────────


def test_unknown_action_is_rejected(server):
    assert "action must be one of" in json.dumps(server.call("pack", {"action": "frobnicate"}))


def test_inspect_requires_path(server):
    assert "path required" in json.dumps(server.call("pack", {"action": "inspect"}))


def test_inspect_missing_file_does_not_crash(server):
    """A bad path is a soft error the agent can recover from, not a traceback."""
    blob = json.dumps(server.call("pack", {"action": "inspect", "path": "/nonexistent/x.ypack"}))
    assert "error" in blob.lower()
