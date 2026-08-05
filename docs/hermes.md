# YantrikDB + Hermes Agent — persistent semantic memory in five minutes

[Hermes Agent](https://hermes-agent.nousresearch.com/) ships with
agent-curated notes and FTS5 keyword recall. That is fine for finding what
you phrased the same way twice — it cannot find what you phrased
*differently*, revise a belief without contradicting the old copy, or tell
you what the agent knew at a point in time. YantrikDB adds that layer:
semantic recall, typed memories, belief revision with history,
contradiction detection, procedural learning, and time-travel recall — all
in a local SQLite file on your machine. No telemetry, no external services.

## 1. Install the server

```bash
pip install yantrikdb-mcp
```

~10 MB, bundled Rust embedder, no native ML dependencies. (Optional
higher-quality embedder: `pip install 'yantrikdb-mcp[onnx]'`.)

## 2. Register it with Hermes

Hermes has a built-in MCP client that discovers servers at startup. Add to
`~/.hermes/config.yaml`:

```yaml
mcp_servers:
  yantrikdb:
    command: "yantrikdb-mcp"
```

Restart Hermes. The tools register with the `mcp_yantrikdb_` prefix —
`mcp_yantrikdb_recall`, `mcp_yantrikdb_remember`, and 17 more.

Optional: pin the database location (default `~/.yantrikdb/memory.db`):

```yaml
mcp_servers:
  yantrikdb:
    command: "yantrikdb-mcp"
    env:
      YANTRIKDB_DB_PATH: "/home/you/agents/memory.db"
```

## 3. Verify

Ask Hermes:

> remember that my favorite deploy window is Friday 6am

then start a **new session** and ask:

> when do I like to deploy?

If the answer comes back with the fact (via `mcp_yantrikdb_recall`), memory
is live across sessions.

## 4. Install the skill (recommended)

The tools alone work, but the skill teaches the agent *when* to use them —
digest at session start, recall before acting, remember decisions, correct
instead of duplicating, consolidate at session end. It follows the open
[Agent Skills](https://agentskills.io) standard, so the same folder works in
Hermes, Claude Code, OpenCode, Goose, and any other compliant harness.

```bash
git clone https://github.com/yantrikos/yantrikdb-mcp
cp -r yantrikdb-mcp/skills/persistent-memory ~/.hermes/skills/
```

Restart Hermes; the skill activates automatically whenever a task touches
past context.

## Patterns for job agents

Running Hermes as a scraper, lead-gen pipeline, or benchmark runner rather
than a chat companion? Three habits pay for the whole setup:

- **One namespace per pipeline** (`namespace="lead-gen"`) so jobs never
  pollute each other's recall.
- **`idempotency_key` on writes** so a retried run cannot double-store a
  lead.
- **Recall before you scrape.** An item already stored with the same outcome
  is a skip — that is what makes run #50 cheaper than run #1.

The [skill](../skills/persistent-memory/SKILL.md) encodes all three.

## Multi-machine / fleet setups

Several Hermes instances sharing one memory: run YantrikDB as an HTTP
cluster and point each agent's MCP server at it — see the
[deployment modes](../README.md#configure) in the main README.

## Questions

Issues and discussion: [github.com/yantrikos/yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp).
