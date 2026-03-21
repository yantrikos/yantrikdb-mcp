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

| | CLAUDE.md | YantrikDB MCP |
|---|---|---|
| 106 memories | 2,550 tokens/conversation | ~450 tokens/task |
| 1,000 memories | 24,000+ tokens (won't fit) | ~450 tokens/task |
| Savings | — | **82–98%** fewer tokens |
| Scales | Linearly (breaks at ~500) | O(1) per query |
| Persists across sessions | Requires file management | Automatic |
| Detects contradictions | No | Yes |
| Knowledge graph | No | Yes |

Selective recall cost is constant — whether you have 100 or 100,000 memories, each query retrieves only the 3–5 most relevant ones.

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
