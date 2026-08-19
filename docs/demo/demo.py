#!/usr/bin/env python3
"""Demo driver for docs/images/demo.gif — drives the REAL yantrikdb-mcp server.

Everything printed under a prompt is the server's own answer. This script
speaks MCP (JSON-RPC over stdio) to `python -m yantrikdb_mcp`, exactly as
Claude Code or Cursor does, and prints what comes back. Nothing is typed to
look like output; the only additions are the `#` comments and the colours.

Run it:     pip install 'yantrikdb-mcp[onnx]' && python docs/demo/demo.py
Record it:  vhs docs/demo/demo.tape
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from importlib.metadata import version

DB = os.environ.get("DEMO_DB", "/tmp/yantrikdb-demo.db")
PY = os.environ.get("DEMO_PY", sys.executable)
EMBEDDER = os.environ.get("YANTRIKDB_EMBEDDER", "onnx")
PACE = float(os.environ.get("DEMO_PACE", "1.0"))

DIM = "\033[38;5;245m"
PROMPT = "\033[38;5;170m"
OK = "\033[38;5;114m"
HOT = "\033[38;5;209m"
BOLD = "\033[1m"
OFF = "\033[0m"

FACTS = [
    "The payments service on-call lead is Alice Chen.",
    "The staging cluster is rebuilt every Sunday night.",
    "The database runs Postgres 16 on a single primary.",
]
QUERY = "who should I contact about a billing outage?"


class Session:
    """One MCP client connection over stdio — i.e. one agent session."""

    def __init__(self, db: str) -> None:
        env = dict(os.environ, YANTRIKDB_DB_PATH=db, YANTRIKDB_PATH=db,
                   YANTRIKDB_EMBEDDER=EMBEDDER)
        self.p = subprocess.Popen(
            [PY, "-m", "yantrikdb_mcp"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env)
        self.n = 0
        self._rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                 "clientInfo": {"name": "demo", "version": "1"}})
        self._rpc("notifications/initialized", notify=True)

    def _rpc(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        if not notify:
            self.n += 1
            msg["id"] = self.n
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        if notify:
            return None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed the connection")
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("id") == self.n:
                return m

    def call(self, tool, args=None):
        r = self._rpc("tools/call", {"name": tool, "arguments": args or {}})
        return json.loads(r["result"]["content"][0]["text"])

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=10)
        except Exception:
            pass


def say(line: str = "") -> None:
    print(line, flush=True)


def beat(mult: float = 1.0) -> None:
    time.sleep(PACE * mult)


def prompt(call_text: str, note: str = "") -> None:
    tail = f"  {DIM}{note}{OFF}" if note else ""
    say(f"{PROMPT}agent>{OFF} {call_text}{tail}")


def main() -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(DB + suffix)
        except OSError:
            pass

    label = {"onnx": "onnx 384d embedder", "bundled": "bundled 64d embedder",
             "multilingual": "multilingual 256d embedder"}.get(EMBEDDER, EMBEDDER)
    say(f"{DIM}yantrikdb-mcp {version('yantrikdb-mcp')} · engine {version('yantrikdb')} · "
        f"{label} · real MCP over stdio{OFF}")
    say()
    beat(0.8)

    s = Session(DB)

    # 1. store three facts
    for text in FACTS:
        rid = s.call("remember", {"text": text, "importance": 0.9})["rid"]
        prompt(f'remember("{text}")')
        say(f"  {OK}stored{OFF} {DIM}{rid[:18]}…{OFF}")
        beat(0.3)
    beat(1.4)
    say()

    # 2. recall by meaning — the query shares no word with the memory
    hits = s.call("recall", {"query": QUERY, "top_k": 2})
    prompt(f'recall("{QUERY}")', "# no shared words")
    say(f"  {DIM} #   similarity   memory{OFF}")
    for i, h in enumerate(hits["results"], 1):
        colour = OK if i == 1 else DIM
        say(f"  {colour} {i}      {h['similarity']:.2f}       {h['text']}{OFF}")
    beat(2.8)
    say()

    # 3. a second, incompatible belief lands in the graph
    s.call("graph", {"action": "relate", "entity": "payments service",
                     "target": "Alice Chen", "relationship": "is"})
    s.call("graph", {"action": "relate", "entity": "payments service",
                     "target": "David Kim", "relationship": "is"})
    prompt('graph(relate, "payments service" is "Alice Chen")')
    prompt('graph(relate, "payments service" is "David Kim")', "# learned weeks later")
    beat(1.6)
    say()

    # 4. think() catches the contradiction
    t = s.call("think", {})
    prompt("think()")
    say(f"  {HOT}{BOLD}conflicts_found: {t['conflicts_found']}{OFF} {DIM}({t['duration_ms']} ms){OFF}")
    c = s.call("conflict", {"action": "list"})["conflicts"][0]
    reason = c["detection_reason"].split(". Reasons:")[0].strip()
    say(f"  {HOT}{c['conflict_type']}{OFF}")
    for line in textwrap.wrap(reason, width=74):
        say(f"  {HOT}{line}{OFF}")
    beat(3.0)
    s.close()
    say()

    # 5. next session: recall hands back the same fact, now flagged
    say(f"{DIM}— new session, same database —{OFF}")
    s2 = Session(DB)
    hits = s2.call("recall", {"query": QUERY, "top_k": 1})
    top = hits["results"][0]
    prompt(f'recall("{QUERY}")')
    say(f"  {DIM} 1      {top['similarity']:.2f}       {top['text']}{OFF}")
    say(f"  {HOT}{BOLD}open_conflicts: {hits['open_conflicts']}{OFF}  "
        f"{DIM}# same answer, now flagged as disputed{OFF}")
    s2.close()
    beat(3.5)


if __name__ == "__main__":
    main()
