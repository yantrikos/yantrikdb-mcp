# YantrikDB + Prime Agent — persistent semantic memory in the kernel

[Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) persists its
own harness state — system prompts, learned skills, sub-agent definitions —
and that is exactly what it should persist. What it does not give you is a
queryable memory: facts recalled by meaning rather than phrasing, beliefs
revised without contradicting the old copy, memory shared with your *other*
agents. YantrikDB adds that layer: semantic recall, typed memories, belief
revision with history, contradiction detection, procedural learning, and
time-travel recall — in a local SQLite file. No telemetry, no external
services.

Because Prime Agent runs tools as Python in a persistent IPython kernel,
the integration is idiomatic there: `import yantrikdb; await
yantrikdb.recall(...)`.

## 1. Install and start the server

Prime Agent's MCP bridge is HTTP-only (stdio servers are not wired through
to the kernel yet), so run yantrikdb-mcp on the streamable-http transport:

```bash
pip install yantrikdb-mcp

export YANTRIKDB_API_KEY=$(openssl rand -hex 16)
yantrikdb-mcp --transport streamable-http --host 127.0.0.1 --port 8420
```

~10 MB, bundled Rust embedder, no native ML dependencies. The bearer token
is required for network transports — the server refuses to run open.

## 2. Declare the server

Add to `~/.prime/agent/settings.json` (or project `.prime/agent/settings.json`):

```json
{
  "mcpServers": {
    "yantrikdb": {
      "type": "http",
      "url": "http://127.0.0.1:8420/mcp",
      "bearerTokenEnvVar": "YANTRIKDB_API_KEY"
    }
  }
}
```

## 3. Install the skill

Copy [`integrations/prime-agent/yantrikdb/`](../integrations/prime-agent/yantrikdb/)
into `~/.prime/agent/skills/`:

```bash
git clone --depth 1 https://github.com/yantrikos/yantrikdb-mcp
cp -r yantrikdb-mcp/integrations/prime-agent/yantrikdb ~/.prime/agent/skills/
```

Launch Prime Agent with `YANTRIKDB_API_KEY` set in the same shell.

## 4. Verify

Ask Prime Agent to run, in its kernel:

```python
import yantrikdb
r1 = await yantrikdb.remember(text="deploy window is Friday 6am", importance=0.8)
r2 = await yantrikdb.recall(query="when do we deploy")
print(r1, r2)
```

The `rid` returned by `remember` should appear in the `recall` results. If
`import yantrikdb` raises `NotEnabled`, the `YANTRIKDB_API_KEY` variable is
not visible to the Prime Agent process.

## What we measured before publishing this

Verified 2026-08-08 with prime-agent 0.7.1 (WSL Ubuntu), yantrikdb-mcp
0.14.0, qwen3.8-max as the driving model, on a fresh database:

- All 20 tools discovered via `await yantrikdb.list_tools()`.
- `remember` returned `{"rid": "019fe43b-ee52-70ce-af71-df900b0024d8", "status": "recorded"}`.
- `recall` on a paraphrased query returned that same rid, semantic score
  0.72, with the marker text intact.
- After packaging the `bearer_token_env` fix, the round-trip passed on the
  first call of a fresh session, with no setup beyond the env var.

## When you don't need this

If one Prime Agent install on one machine is your whole setup and its own
persistent state files cover what you want carried forward, they are the
simpler tool — no server process to run. YantrikDB earns its keep when
recall must survive paraphrase, when facts change and history matters, or
when several harnesses (Prime Agent, Claude Code, Cursor, Hermes) should
share one memory.

## Notes

- The skill module is named `yantrikdb`, the same name as the YantrikDB
  Python package. Prime Agent's managed kernel venv does not include the
  engine package, so there is no collision in a default setup; if you add
  the engine to that venv yourself, rename one of them.
- The database defaults to `~/.yantrikdb/memory.db`; point the server at a
  different file with `YANTRIKDB_DB_PATH`.
