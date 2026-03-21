# YantrikDB MCP Server

Cognitive memory for AI agents. Works with Claude Code, Cursor, Windsurf, and any MCP-compatible client.

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

## Why Not Just Use CLAUDE.md?

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

17 tools available to your agent:

| Tool | Purpose |
|---|---|
| `remember` | Store a new memory |
| `recall` | Search memories by semantic similarity |
| `recall_refine` | Refine a low-confidence recall |
| `bulk_remember` | Store multiple memories at once |
| `forget` | Tombstone a memory |
| `correct` | Fix an incorrect memory (preserves history) |
| `update_importance` | Adjust memory importance |
| `relate` | Create entity relationships |
| `entity_edges` | Get all relationships for an entity |
| `search_entities` | Find entities by name pattern |
| `think` | Run consolidation + conflict detection + pattern mining |
| `conflicts` | List detected contradictions |
| `conflict_resolve` | Resolve a contradiction |
| `recall_feedback` | Improve retrieval quality over time |
| `triggers` | Get proactive insights and warnings |
| `health_check` | Verify the server is operational |
| `stats` | Get memory engine statistics |

See [yantrikdb.com/guides/mcp](https://yantrikdb.com/guides/mcp/) for full documentation.

## License

This MCP server is licensed under **MIT** — use it freely in any project.

Note: This package depends on [yantrikdb](https://github.com/yantrikos/yantrikdb) (the cognitive memory engine), which is licensed under **AGPL-3.0**. The AGPL applies to the engine itself — if you modify the engine and distribute it or provide it as a network service, those modifications must also be AGPL-3.0. Using the engine as-is via this MCP server does not trigger AGPL obligations on your code.
