# Installing yantrikdb-mcp (for install agents)

This file is for an agent (Cline, Claude Code, Cursor, …) setting this server
up on a user's machine. Follow it top to bottom; do not improvise.

## What the user gets

A local memory store the agent can write to and read back in later sessions:
`remember` / `recall`, contradiction detection, a knowledge graph, and
consolidation. Everything is stored in one SQLite file on the user's machine.
No account, no API key, no external service, no telemetry.

## Requirements

- Python 3.10 or newer (`python --version`)
- `pip`
- Nothing else. No compiler, no model download, no Docker.

## Step 1 — install the package

```bash
pip install yantrikdb-mcp
```

Verify the entry point resolves:

```bash
yantrikdb-mcp --help
```

If `yantrikdb-mcp` is not on PATH, use `python -m yantrikdb_mcp` everywhere
below instead of the bare command, or install with `pipx install yantrikdb-mcp`
and use the pipx shim.

## Step 2 — add the server to the MCP config

Use the default (local) mode. It needs no environment variables:

```json
{
  "mcpServers": {
    "yantrikdb": {
      "command": "yantrikdb-mcp"
    }
  }
}
```

Write that into the client's MCP settings file. For Cline that is
`cline_mcp_settings.json`; for Claude Code, `.mcp.json` in the project or the
user-level config.

Do **not** add `YANTRIKDB_SERVER_URL` or `YANTRIKDB_TOKEN` unless the user has
told you they run a YantrikDB HTTP cluster. Those switch the server into
cluster-client mode and it will then fail without a reachable cluster.

## Step 3 — verify

Restart the client and confirm the server connects and lists its tools
(`remember`, `recall`, `think`, `graph`, `session`, … — 20 tools on v0.19.x).
Then call `remember` once with a short fact and `recall` with a short query
describing it; the fact should come back.

If tools do not appear, check the client's MCP log for the launch command —
the usual cause is `yantrikdb-mcp` not being on the PATH the client uses. Fix
it by pointing `command` at the absolute path of the installed script.

## Optional settings (only if the user asks)

| Variable | Default | What it changes |
|---|---|---|
| `YANTRIKDB_DB_PATH` | `~/.yantrikdb/memory.db` | Where the SQLite database lives |
| `YANTRIKDB_EMBEDDER` | `auto` | `bundled` (64-dim, default), `onnx` (384-dim, needs `pip install 'yantrikdb-mcp[onnx]'`), `multilingual` (256-dim) |
| `YANTRIKDB_NAMESPACE` | `default` | Namespace records are written to |

The `[onnx]` extra adds roughly 150 MB of install. Do not install it by
default — the bundled embedder needs no native ML dependencies.

## Uninstall

```bash
pip uninstall yantrikdb-mcp
```

The database file is left in place. Delete `~/.yantrikdb/memory.db` (or the
path set in `YANTRIKDB_DB_PATH`) to remove the stored memories as well.

## Full documentation

<https://github.com/yantrikos/yantrikdb-mcp#readme> ·
<https://yantrikdb.com/guides/mcp/>
