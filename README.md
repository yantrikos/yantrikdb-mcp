<!-- mcp-name: io.github.yantrikos/yantrikdb-mcp -->
# YantrikDB MCP Server

Cognitive memory for AI agents. Works with Claude Code, Cursor, Windsurf, and any MCP-compatible client.

**Website:** [yantrikdb.com](https://yantrikdb.com) · **Docs:** [yantrikdb.com/guides/mcp](https://yantrikdb.com/guides/mcp/) · **GitHub:** [yantrikos/yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp)

## Install

```bash
pip install yantrikdb-mcp
```

## Configure

Add to your MCP client config:

```json
{
  "mcpServers": {
    "yantrikdb": {
      "command": "yantrikdb-mcp"
    }
  }
}
```

That's it. The agent auto-recalls context, auto-remembers decisions, and auto-detects contradictions — no prompting needed.

### Remote Server Mode

Run as a shared server accessible from multiple machines:

```bash
# Generate a secure API key
export YANTRIKDB_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Start SSE server
yantrikdb-mcp --transport sse --port 8420
```

Connect clients to the remote server:

```json
{
  "mcpServers": {
    "yantrikdb": {
      "type": "sse",
      "url": "http://your-server:8420/sse",
      "headers": {
        "Authorization": "Bearer YOUR_API_KEY"
      }
    }
  }
}
```

Supports `sse` and `streamable-http` transports. Bearer token auth via `YANTRIKDB_API_KEY` env var.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `YANTRIKDB_DB_PATH` | `~/.yantrikdb/memory.db` | Database file path |
| `YANTRIKDB_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `YANTRIKDB_EMBEDDING_DIM` | `384` | Embedding dimension |
| `YANTRIKDB_API_KEY` | *(none)* | Bearer token for network transports |

## Why Not File-Based Memory?

File-based memory (CLAUDE.md, memory files) loads **everything** into context every conversation. YantrikDB recalls only what's relevant.

### Benchmark: 15 queries × 4 scales

| Memories | File-Based | YantrikDB | Savings | Precision |
|---|---|---|---|---|
| 100 | 1,770 tokens | 69 tokens | **96%** | 66% |
| 500 | 9,807 tokens | 72 tokens | **99.3%** | 77% |
| 1,000 | 19,988 tokens | 72 tokens | **99.6%** | 84% |
| 5,000 | 101,739 tokens | 53 tokens | **99.9%** | 88% |

**Selective recall is O(1). File-based memory is O(n).**

- At 500 memories, file-based exceeds 32K context windows
- At 5,000, it doesn't fit in *any* context window — not even 200K
- YantrikDB stays at ~70 tokens per query, under 60ms latency
- Precision *improves* with more data — the opposite of context stuffing

Run the benchmark yourself: `python benchmarks/bench_token_savings.py`

## Tools

15 tools, full engine coverage:

| Tool | Actions | Purpose |
|---|---|---|
| `remember` | single / batch | Store memories — decisions, preferences, facts, corrections |
| `recall` | search / refine / feedback | Semantic search, refinement, and retrieval feedback |
| `forget` | single / batch | Tombstone memories |
| `correct` | — | Fix incorrect memory (preserves history) |
| `think` | — | Consolidation + conflict detection + pattern mining |
| `memory` | get / list / search / update_importance / archive / hydrate | Manage individual memories + keyword search |
| `graph` | relate / edges / link / search / profile / depth | Knowledge graph operations |
| `conflict` | list / get / resolve / reclassify | Handle contradictions and teach substitution patterns |
| `trigger` | pending / history / acknowledge / deliver / act / dismiss | Proactive insights and warnings |
| `session` | start / end / history / active / abandon_stale | Session lifecycle management |
| `temporal` | stale / upcoming | Time-based memory queries |
| `procedure` | learn / surface / reinforce | Procedural memory — learn and reuse strategies |
| `category` | list / members / learn / reset | Substitution categories for conflict detection |
| `personality` | get / set | AI personality traits from memory patterns |
| `stats` | stats / health / weights / maintenance | Engine stats, health, weights, and index rebuilds |

See [yantrikdb.com/guides/mcp](https://yantrikdb.com/guides/mcp/) for full documentation.

## Examples

### 1. Auto-recall at conversation start

**User:** "What did we decide about the database migration?"

The agent automatically calls `recall("database migration decision")` and retrieves relevant memories before responding — no manual prompting needed.

### 2. Remember decisions + build knowledge graph

**User:** "We're going with PostgreSQL for the new service. Alice will own the migration."

The agent calls:
- `remember(text="Decided to use PostgreSQL for the new service", domain="architecture", importance=0.8)`
- `remember(text="Alice owns the PostgreSQL migration", domain="people", importance=0.7)`
- `graph(action="relate", entity="Alice", target="PostgreSQL Migration", relationship="owns")`

### 3. Contradiction detection

After storing "We use Python 3.11" and later "We upgraded to Python 3.12", calling `think()` detects the conflict. The agent surfaces it:

> "I found a contradiction: you previously said Python 3.11, but recently mentioned Python 3.12. Which is current?"

Then resolves with `conflict(action="resolve", conflict_id="...", strategy="keep_b")`.

## Privacy Policy

YantrikDB MCP Server stores all data **locally on your machine** (default: `~/.yantrikdb/memory.db`). No data is sent to external servers, no telemetry is collected, and no third-party services are contacted during operation.

- **Data collection:** Only what you explicitly store via the `remember` tool or what the AI agent stores on your behalf.
- **Data storage:** Local SQLite database on your filesystem. You control the path via `YANTRIKDB_DB_PATH`.
- **Third-party sharing:** None. Data never leaves your machine in local (stdio) mode.
- **Network mode:** When using SSE/HTTP transport, data travels between your client and your self-hosted server. No Anthropic or third-party servers are involved.
- **Embedding model:** Uses a local ONNX model (`all-MiniLM-L6-v2`). Model files are downloaded once from Hugging Face Hub on first use, then cached locally.
- **Retention:** Data persists until you delete it (`forget` tool) or delete the database file.
- **Contact:** developer@pranab.co.in

Full policy: [yantrikdb.com/privacy](https://yantrikdb.com/privacy/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for a venv setup, running `pytest`, and opening PRs.

## Support

- **Issues:** [github.com/yantrikos/yantrikdb-mcp/issues](https://github.com/yantrikos/yantrikdb-mcp/issues)
- **Email:** developer@pranab.co.in
- **Docs:** [yantrikdb.com/guides/mcp](https://yantrikdb.com/guides/mcp/)

## License

This MCP server is licensed under **MIT** — use it freely in any project.

Note: This package depends on [yantrikdb](https://github.com/yantrikos/yantrikdb) (the cognitive memory engine), which is licensed under **AGPL-3.0**. The AGPL applies to the engine itself — if you modify the engine and distribute it or provide it as a network service, those modifications must also be AGPL-3.0. Using the engine as-is via this MCP server does not trigger AGPL obligations on your code.
