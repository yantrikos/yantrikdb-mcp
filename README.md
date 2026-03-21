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

43 tools available to your agent:

| Tool | Purpose |
|---|---|
| `remember` | Store a new memory |
| `recall` | Search memories by semantic similarity |
| `recall_refine` | Refine a low-confidence recall |
| `bulk_remember` | Store multiple memories at once |
| `bulk_forget` | Tombstone multiple memories at once |
| `forget` | Tombstone a memory |
| `get_memory` | Retrieve a single memory by ID |
| `list_memories` | List memories with filters |
| `correct` | Fix an incorrect memory (preserves history) |
| `update_importance` | Adjust memory importance |
| `relate` | Create entity relationships |
| `entity_edges` | Get all relationships for an entity |
| `link_memory_entity` | Manually link a memory to an entity |
| `search_entities` | Find entities by name pattern |
| `entity_profile` | Temporal profile with valence trends |
| `relationship_depth` | Measure depth of relationship knowledge |
| `think` | Run consolidation + conflict detection + pattern mining |
| `conflicts` | List or get specific contradictions |
| `conflict_resolve` | Resolve or dismiss a contradiction |
| `reclassify_conflict` | Reclassify and teach substitution patterns |
| `recall_feedback` | Improve retrieval quality over time |
| `triggers` | Get proactive insights, warnings, and history |
| `acknowledge_trigger` | Update trigger lifecycle (acknowledge/dismiss/act) |
| `personality` | Get emergent AI personality traits |
| `set_personality` | Manually tune a personality trait |
| `patterns` | View discovered cross-domain patterns |
| `archive` | Move memory to cold storage |
| `hydrate` | Restore archived memory to active |
| `learned_weights` | See adapted recall scoring weights |
| `substitution_categories` | Inspect or drill into conflict detection categories |
| `learn_category_members` | Teach new members to a category |
| `reset_category` | Reset a contaminated category to seed |
| `session_start` | Start an interaction session |
| `session_end` | End session or clean up stale sessions |
| `session_history` | View past sessions or check active session |
| `stale_memories` | Find important memories not accessed recently |
| `upcoming_memories` | Find memories with approaching deadlines |
| `learn_procedure` | Store a learned strategy/procedure |
| `surface_procedures` | Find relevant procedures |
| `reinforce_procedure` | Update procedure effectiveness |
| `maintenance` | Rebuild indexes and backfill entity links |
| `health_check` | Verify the server is operational |
| `stats` | Get memory engine statistics (with procedural) |

See [yantrikdb.com/guides/mcp](https://yantrikdb.com/guides/mcp/) for full documentation.

## License

This MCP server is licensed under **MIT** — use it freely in any project.

Note: This package depends on [yantrikdb](https://github.com/yantrikos/yantrikdb) (the cognitive memory engine), which is licensed under **AGPL-3.0**. The AGPL applies to the engine itself — if you modify the engine and distribute it or provide it as a network service, those modifications must also be AGPL-3.0. Using the engine as-is via this MCP server does not trigger AGPL obligations on your code.
