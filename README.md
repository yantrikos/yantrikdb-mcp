<!-- mcp-name: io.github.yantrikos/yantrikdb-mcp -->
# YantrikDB MCP Server

**YantrikDB — Cognitive memory for AI agents. Persistent semantic recall, knowledge graph, contradiction detection, and procedural learning. Ships as embeddable engine, network database, or MCP server.**

Works with Claude Code, Cursor, Windsurf, and any MCP-compatible client.

**Website:** [yantrikdb.com](https://yantrikdb.com) · **Docs:** [yantrikdb.com/guides/mcp](https://yantrikdb.com/guides/mcp/) · **GitHub:** [yantrikos/yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp) · **Paper:** [Skill as Memory, Not Document](https://doi.org/10.5281/zenodo.20128887)

## At a glance

| | |
|---|---|
| **What it is** | An MCP server that gives any MCP-compatible AI agent persistent, structured, queryable memory across sessions |
| **Install** | `pip install yantrikdb-mcp` |
| **Works with** | Claude Code, Cursor, Windsurf, Continue, Claude Desktop, any MCP client |
| **Storage** | Local SQLite at `~/.yantrikdb/memory.db` (or any path; or HTTP cluster) |
| **Embedder** | Bundled 64-dim Rust embedder (default), 384-dim ONNX MiniLM (`[onnx]` extra), 256-dim multilingual (101 languages) |
| **Tools** | 19 — remember, recall, forget, correct, think, memory, graph, conflict, trigger, session, temporal, procedure, category, personality, stats, skill, gaps, conversation, task |
| **License** | MIT (engine: AGPL-3.0) |
| **Privacy** | All data on your machine. No telemetry. No external services. |

## Install

```bash
# Default — uses the engine's bundled 64-dim embedder. ~10 MB install,
# ~80 ms cold start, no native ML deps.
pip install yantrikdb-mcp

# Optional: higher-quality 384-dim ONNX MiniLM-L6-v2 embedder (~150 MB install).
# Auto-used when an existing pre-v0.6 database is detected.
pip install 'yantrikdb-mcp[onnx]'
```

> **Upgrading from v0.5.x?** Your existing database stays at 384 dim — install
> the `[onnx]` extra to keep using it transparently. New installs default to
> the lean bundled embedder. v0.7.0+ pins the engine migration fix automatically.
> See [Embedder backends](#embedder-backends) below.

## Configure

The MCP server has three deployment modes. Pick the one that fits your setup.

### Mode 1 — Local (default, recommended for single user)

The MCP server runs the engine in-process with a local SQLite database. Fast, private, zero dependencies.

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

### Mode 2 — HTTP Cluster (recommended for shared/multi-machine setups)

Forward all tool calls to a [YantrikDB HTTP cluster](https://github.com/yantrikos/yantrikdb-server) instead of using an embedded engine. The MCP server is a thin stateless client — all memories live on the cluster, accessible from any machine.

Benefits: shared memory across machines, high availability, no local embedder download, no local database.

```json
{
  "mcpServers": {
    "yantrikdb": {
      "command": "yantrikdb-mcp",
      "env": {
        "YANTRIKDB_SERVER_URL": "http://node1:7438,http://node2:7438",
        "YANTRIKDB_TOKEN": "ydb_your_database_token"
      }
    }
  }
}
```

- Comma-separate multiple nodes for Raft cluster auto-discovery
- Automatic leader-following on failover
- 15s request timeout
- Get the token from the cluster: `yantrikdb token create --db your_database`

### Mode 3 — SSE Server (legacy, single remote instance)

Run the MCP server itself as a long-running SSE server with its own embedded database. Clients connect via HTTP streaming.

```bash
# Generate a secure API key
export YANTRIKDB_API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")

# Start SSE server
yantrikdb-mcp --transport sse --port 8420
```

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

Supports `sse` and `streamable-http` transports. Note: SSE connections can drop on idle — Mode 2 (HTTP Cluster) is more reliable for shared deployments.

### Environment Variables

| Variable | Used in Mode | Default | Description |
|---|---|---|---|
| `YANTRIKDB_SERVER_URL` | Cluster | *(unset → local mode)* | Comma-separated cluster node URLs |
| `YANTRIKDB_TOKEN` | Cluster | *(none)* | Bearer token for the cluster database |
| `YANTRIKDB_DB_PATH` | Local | `~/.yantrikdb/memory.db` | Database file path |
| `YANTRIKDB_EMBEDDER` | Local | `auto` | Backend selector: `auto` \| `bundled` \| `onnx` \| `multilingual` |
| `YANTRIKDB_EMBEDDING_MODEL` | Local | `all-MiniLM-L6-v2` | ONNX model name (only used when `YANTRIKDB_EMBEDDER=onnx`) |
| `YANTRIKDB_SKILLS_WRITE_ENABLED` | All | `false` | Set `true` to allow agents to author skills via `skill(action="define")` (see [Skill substrate](#skill-substrate-v070) below) |
| `YANTRIKDB_OUTCOMES_WRITE_ENABLED` | All | `true` | Outcome tracking via `skill(action="outcome")`. Defaults on so the feedback loop works out of the box; set `false` to lock the outcome substrate. Added in v0.8.1 per [#8](https://github.com/yantrikos/yantrikdb-mcp/issues/8) |
| `YANTRIKDB_API_KEY` | SSE server | *(none)* | Bearer token when serving SSE/HTTP |

### Embedder backends

Local mode ships three embedders. The MCP picks one automatically; override with `YANTRIKDB_EMBEDDER`.

| Backend | Dim | Cold start | Install size | Language coverage | When it's used |
|---|---|---|---|---|---|
| `bundled` (engine default) | 64 | ~80 ms | ~10 MB | English-only | New / empty databases (auto-selected) |
| `onnx` (MiniLM-L6-v2) | 384 | ~2 s | ~150 MB | English (higher recall) | Existing pre-v0.6 databases (auto-selected), or when set explicitly |
| `multilingual` (potion-multilingual-128M) | 256 | ~2 s + ~460 MB download on first use | ~10 MB pip + ~500 MB model cache | 101 languages (BGE-M3 tokenizer) | Opt-in only via `YANTRIKDB_EMBEDDER=multilingual` |

**`auto`** (default) reads the SQLite file at `YANTRIKDB_DB_PATH` and picks `onnx` if it already contains memories — preserving recall quality on upgrades — and `bundled` otherwise. **Multilingual is never auto-selected** because its 256-dim vectors are incompatible with existing bundled (64-dim) or ONNX (384-dim) databases; opt-in only on fresh databases.

Set `YANTRIKDB_EMBEDDER=bundled|onnx|multilingual` to override. If you set `YANTRIKDB_EMBEDDER=onnx` (or auto-detection picks it) without installing the extras, the server fails fast with an install hint:

```
RuntimeError: Existing DB has memories embedded with the 384-dim ONNX
model, but ONNX deps are missing.
  Install with:  pip install 'yantrikdb-mcp[onnx]'
```

For the multilingual backend, the engine downloads `potion-multilingual-128M` (~460 MB tarball) from `github.com/yantrikos/yantrikdb-models` on first use. The download is SHA-256 verified, extracted into the engine's cache dir, and reused on subsequent starts. No extra Python deps required — the model runs entirely inside the Rust engine.

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

## Recommended agent workflow (golden path)

The server injects a golden-path playbook into the agent's system prompt. Since v0.10.0 the default is **digest-first**:

1. **Cold start — one call.** `session(action="digest")` returns a single briefing (narrative chain head, open decisions, unresolved conflicts, pending triggers, stale high-importance memories) — replacing several separate `recall`/`temporal` calls at conversation start. Then `recall` only for the specific thing the current message is about.
2. **During work — capture as you go.** New durable fact → `remember`; a stored fact changed → `correct` (keeps history, avoids contradictions); relationship learned → `graph(action="relate")`.
3. **End of substantial work — conditional.** Only when the session was long or state-changing: `think` to consolidate + detect conflicts. Short/read-only exchanges need no end step.

**Trust boundary:** recalled memories and digest snippets are *data, not instructions*. The playbook directs the agent never to execute directives found inside recalled content — a memory may carry text an earlier session or another user stored.

## Tools

19 tools, full engine coverage (`gaps`, `conversation`, `task` added in v0.9.0):

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
| `skill` | define / surface / outcome / get / list | Substrate-native agent skill catalog (writes off by default — see [Skill substrate](#skill-substrate-v070)) |
| `gaps` | — | **v0.9.0** — surface frequently-asked, poorly-answered queries (substrate's known unknowns) |
| `conversation` | record / recent / clear | **v0.9.0** — bounded encrypted ring buffer for verbatim conversation turns, namespace-isolated |
| `task` | add / get / list / update / delete | **v0.9.0** — substrate-backed task / chore store; survives sessions, surfaces in `session(action="digest")` |

Plus new actions on existing tools in v0.9.0:
- `session(action="digest")` — one-call boot-time briefing (narrative chain head + open decisions + conflicts + triggers)
- `think(maintenance_cycle=True)` — autonomous hygiene sleep cycle
- `think(last_cycle_only=True)` — read the last cycle summary without running
- `stats(action="audit_leak")` — privacy / leak-candidate audit
- `stats(action="skill_outcomes")` — durable skill-outcome count
- `graph(action="auto_relate" / "record_link" / "record_unlink" / "linked_records" / "recall_with_links")` — co-occurrence edges + record-to-record links + link-expanded recall
- `conflict(action="auto_resolve")` — burn down unambiguous conflicts in one pass
- `memory(action="chain_head" / "history")` — chain-namespace head + revision history
- `trigger(action="prune")` — bound the pending-trigger backlog
- `remember(summary=...)` — draft mode: engine atomizes a long summary into linked semantic facts (end-of-session auto-capture)

See [yantrikdb.com/guides/mcp](https://yantrikdb.com/guides/mcp/) for full documentation.

## Skill substrate (v0.8.0+)

YantrikDB exposes a structured agent skill catalog — separate from loose `procedure` memories. Skills have schema (`skill_id`, `applies_to`, `triggers`, `body`, `type`) and are stored in the dedicated `skill_substrate` namespace so multiple consumers (this MCP, [yantrikdb-hermes-plugin](https://github.com/yantrikos/yantrikdb-hermes-plugin), Lane B SDK, WisePick, yantrikdb-server's `/v1/skills/*` endpoints) all read and write the same substrate. Background: [Sarkar 2026 — Skill as Memory, Not Document](https://doi.org/10.5281/zenodo.20128887).

### Security model

Skill writes shape future agent behavior across sessions, so the MCP server implements defense-in-depth. Every control has an env-var knob (locked once at startup — `C2`) and the full state is exposed via `stats(action="stats")` and the audit log.

**Layered controls** (each ships *on* by default unless noted):

| Layer | Control | Env var | Notes |
|---|---|---|---|
| **Schema** | `skill_id` regex, body 50–5000 chars, `applies_to` 1–10 entries, `skill_type` enum | (always on) | Same regex set as yantrikdb-server `/v1/skills/define` |
| **A1** Prompt-injection markers | Reject bodies containing role-confusion / "ignore previous instructions" patterns | `YANTRIKDB_SKILLS_DISABLE_SCANNERS=A1` to disable (audited) | OWASP LLM01 |
| **A2** Credential scanner | AWS/GitHub/Slack/Stripe/Google/Anthropic/OpenAI keys, SSH/PGP private keys, JWT, password assignments | `=A2` to disable | Subset of GitHub secret-scanning |
| **A3** URL/IP block | Reject http(s), ftp, IPv4 literals in body | `YANTRIKDB_SKILLS_ALLOW_URLS=true` to allow | Exfil path for downstream agents |
| **A4** Unicode evasion | Reject non-printing chars (Cf/Cs/Cn except whitelisted) | `=A4` to disable | Bidi override (U+202E), zero-width spaces |
| **A5** Encoded payload | Reject ≥200-char runs of base64/hex | `=A5` to disable | Heuristic — false-positive prone for large hashes |
| **B1** Namespace allowlist | `skill_id` first segment must be in operator list | `YANTRIKDB_SKILLS_ALLOWED_NAMESPACES=workflow,review` | Unset = all allowed |
| **B2** Author attribution | Records `session_id`, `os_user`, `hostname`, `wall_clock`, `audit_nonce` | (always on) | Forensic trail |
| **B3** Cross-origin replace | Refuse to overwrite a skill written by a different consumer | `YANTRIKDB_SKILLS_ALLOW_CROSS_ORIGIN_REPLACE=true` to allow | Defends against MCP↔hermes-plugin collision |
| **B4** Supersedes integrity | `supersedes` must reference an existing skill in the same namespace | (always on) | Blocks malicious retirement of legit skills |
| **C1** Time-bound gate | Gate auto-closes at the timestamp (applies to both define + outcome) | `YANTRIKDB_SKILLS_WRITE_EXPIRES_AT=2026-12-31T00:00:00Z` | Unset = no expiry |
| **C1.5** Split outcome gate | `outcome` action uses its own gate, default ON | `YANTRIKDB_OUTCOMES_WRITE_ENABLED=false` to lock outcomes too | v0.8.1+: `define` and `outcome` have different threat profiles — outcome can't introduce new instructions, only append `{succeeded, note≤500}` against an existing skill. Feedback loop works by default; lock explicitly if needed |
| **C2** Locked config | All `YANTRIKDB_SKILLS_*` / `YANTRIKDB_OUTCOMES_*` env vars read once at startup | (always on) | Mutating env in a sub-process can't bypass the gate |
| **D1** Audit log | JSONL append of every accept/reject/tamper event | `YANTRIKDB_SKILLS_AUDIT_LOG=/var/log/yantrikdb/skills.jsonl` | Unset = no auditing (warns at boot) |
| **D2** Rate limit | Per-session-id sliding-window write cap | `YANTRIKDB_SKILLS_WRITE_RATE=30` (default writes/min) | Defeats flood attacks |
| **D3** Outcome.note guards | Note ≤500 chars + scanned by A1/A2/A4 | (always on) | Closes the outcome side-channel |
| **D4** Counters in `stats` | Accept/reject counts by reason, surfaced in `stats(action="stats")["skill_substrate"]` | (always on) | Operator dashboards |
| **E1** Body SHA-256 | Stored at write time, re-verified on every read | (always on) | Detects out-of-band DB tampering — surface/get omit mismatches and log to audit |
| **E2** Author origin | `metadata.author_origin` tag — defaults to `yantrikdb-mcp` | `YANTRIKDB_SKILLS_AUTHOR_ORIGIN=...` to override | Tracks substrate provenance across consumers |
| **F** Startup safety | Boot-time warnings about dangerous configurations | (always on) | Logs `[F.1]`–`[F.5]` to stderr + audit |
| **G** Review queue for `rule` | `rule`-type skills route to `skill_pending_review` (not surfaced by `surface/get/list`) | `YANTRIKDB_SKILLS_RULE_REQUIRES_REVIEW=false` to disable (not recommended) | Rules influence agent policy — human approval required |
| **Multi-tenant guard** | `[F.1]` warning if DB shows multiple actor IDs without ack | `YANTRIKDB_SKILLS_MULTITENANT_ACK=true` | One DB = one tenant is the safe default |

**Enterprise checklist:**

```bash
# Minimum production config when you turn the gate ON:
YANTRIKDB_SKILLS_WRITE_ENABLED=true
YANTRIKDB_SKILLS_WRITE_EXPIRES_AT=2026-12-31T00:00:00Z
YANTRIKDB_SKILLS_ALLOWED_NAMESPACES=workflow,review,onboarding
YANTRIKDB_SKILLS_AUDIT_LOG=/var/log/yantrikdb/skills.audit.jsonl
YANTRIKDB_SKILLS_AUTHOR_ORIGIN=acme-corp-claude-prod
# Defaults are already correct: writes off, scanners on, rate-limit 30/min,
# rule-type routed to review, body-hash verified on read, locked at startup.
```

The audit log is the canonical record. Every accept, every reject (with the scanner that flagged), every tamper-detection on read, every gate-closed-due-to-expiry — all there in JSONL. Plug it into your SIEM.

### `stats(action="stats")` example output (skill_substrate slice)

```json
"skill_substrate": {
  "counters": {
    "skill_defines_accepted": 12,
    "skill_defines_rejected": {"content_scan:A2": 1, "namespace_not_allowed": 3},
    "skill_outcomes_recorded": 47,
    "skill_pending_review": 2
  },
  "config": {
    "writes_enabled": true,
    "write_expires_at": "2026-12-31T00:00:00+00:00",
    "allowed_namespaces": ["workflow", "review"],
    "audit_log_path": "/var/log/yantrikdb/skills.audit.jsonl",
    "rule_requires_review": true,
    "author_origin": "acme-corp-claude-prod"
  }
}
```

### Schema (validated at write time)

| Field | Constraint |
|---|---|
| `skill_id` | Lowercase dot-separated segments, length 4–200, e.g. `workflow.git.commit_clean` |
| `body` | 50–5000 chars |
| `applies_to` | 1–10 lowercase-underscore identifiers (**no hyphens** — load-bearing for substrate consistency) |
| `skill_type` | One of `procedure`, `reference`, `lesson`, `pattern`, `rule` |
| `on_conflict` | `reject` (default) or `replace` |

### Example session

```python
# Define (requires gate enabled)
skill(action="define",
      skill_id="workflow.git.commit_clean",
      body="Before commit: run pytest, run lint, write a clear subject + body.",
      skill_type="procedure",
      applies_to=["git", "release"])

# Surface relevant skills for the current task
skill(action="surface", query="how to commit cleanly", top_k=5)

# Record an outcome after using the skill (gated, append-only)
skill(action="outcome", skill_id="workflow.git.commit_clean",
      succeeded=True, note="caught a flake8 issue pre-push")
```

Outcomes are append-only events in the `outcome_substrate` namespace — no auto-rollup on the parent skill, matching yantrikdb-server's "schema not semantics" design rule. Agents (or the operator) can aggregate outcomes themselves to compute success rates.

## FAQ

### What is YantrikDB MCP?

YantrikDB MCP is a Model Context Protocol (MCP) server that gives AI agents persistent cognitive memory across sessions. It exposes 16 tools (remember, recall, forget, correct, think, graph, conflict, trigger, session, temporal, procedure, category, personality, stats, memory, skill) that any MCP-compatible client — Claude Code, Cursor, Windsurf, Continue, Claude Desktop — can call automatically without prompting.

### How is this different from file-based memory like CLAUDE.md?

File-based memory loads *everything* into context on every conversation, which scales O(n) in token cost. YantrikDB uses selective semantic recall — at 5,000 memories, file-based costs ~101K tokens per conversation while YantrikDB costs ~53 tokens. Precision *improves* with more data instead of degrading as the context window fills up. Benchmark script: `python benchmarks/bench_token_savings.py`.

### How does it compare to mem0 / Letta / Zep / native MCP memory?

See [comparison table](#comparison-with-other-agent-memory-systems) below. Short version: YantrikDB is the only one that ships as both an embeddable Rust engine *and* an MCP server *and* a network database with the same substrate semantics. It's the only one with first-class procedural memory + a skill substrate validated by schema at write time + autonomous consolidation/conflict detection. It's also the only one whose underlying engine is published as a peer-reviewed paper ([Sarkar 2026, Zenodo DOI 10.5281/zenodo.20128887](https://doi.org/10.5281/zenodo.20128887)).

### Can I self-host?

Yes — three ways. (1) Local: just `pip install yantrikdb-mcp` and point your MCP client at it. SQLite lives at `~/.yantrikdb/memory.db`. (2) Network: run [yantrikdb-server](https://github.com/yantrikos/yantrikdb-server) as a multi-tenant HTTP cluster, point the MCP at it via `YANTRIKDB_SERVER_URL`. (3) Hybrid: SSE server mode (`yantrikdb-mcp --transport sse`) for shared deployments.

### Is my data sent anywhere?

No. All data stays on your machine (or your cluster). No telemetry, no third-party services. The default embedder runs entirely in the Rust engine via static lookup — no model downloads or API calls. The optional `[onnx]` and multilingual embedders fetch model weights once from HuggingFace's CDN and run locally thereafter.

### What's the difference between `procedure` and `skill`?

`procedure` stores loose how-to memories (effectiveness-ranked, no schema). `skill` stores structured catalog entries (`skill_id`, `applies_to`, `triggers`, `body`, `type`) in a dedicated `skill_substrate` namespace shared with [yantrikdb-hermes-plugin](https://github.com/yantrikos/yantrikdb-hermes-plugin), Lane B SDK, WisePick, and the [yantrikdb-server `/v1/skills/*` endpoints](https://github.com/yantrikos/yantrikdb-server). Use `procedure` for personal how-to notes; use `skill` for structured agent capabilities that other consumers should be able to surface.

### Is skill authoring safe to enable?

Skill writes are off by default precisely because they can shape future agent behavior. When you turn the gate on, seven layers of defense-in-depth apply: prompt-injection scanner, credential scanner, URL block, unicode-evasion scanner, namespace allowlist, author attribution, audit log, rate limit, body-hash tamper detection, and a review queue for `rule`-type skills. See [Security model](#security-model) above.

### Does it work in production?

Yes — yantrikdb-mcp runs in production on the YantrikDB homelab cluster (1973+ memories, SSE transport, 2 weeks uptime per release cycle) and is the reference deployment behind the engine's release decisions. v0.8.x added the engine's same-day-patch cadence to the MCP server itself: external issues filed by community contributors land as released fixes within 2 hours.

### What's the engine written in?

The YantrikDB engine is Rust ([crates.io: yantrikdb](https://crates.io/crates/yantrikdb)) with pyo3 Python bindings ([PyPI: yantrikdb](https://pypi.org/project/yantrikdb/)). The MCP server itself is Python — a thin wrapper around the engine's Python bindings, plus stdio/SSE/HTTP transport plumbing.

## Comparison with other agent memory systems

| Capability | YantrikDB MCP | mem0 | Letta (MemGPT) | Zep | Native MCP filesystem memory |
|---|---|---|---|---|---|
| MCP-native | ✅ first-class | via custom integration | via custom integration | via custom integration | ✅ filesystem-shaped |
| Embeddable (no server) | ✅ Rust + Python | ❌ requires service | ❌ requires service | ❌ requires service | ✅ filesystem |
| Network database mode | ✅ Raft HA cluster | ✅ Pro / Enterprise | ✅ self-host | ✅ managed + self-host | ❌ |
| Semantic recall (vector) | ✅ HNSW | ✅ | ✅ | ✅ | ❌ (file grep only) |
| Knowledge graph | ✅ typed nodes + edges | ✅ (recent addition) | partial | ✅ | ❌ |
| Contradiction detection | ✅ autonomous | ❌ | ❌ | ❌ | ❌ |
| Procedural memory | ✅ effectiveness-ranked | ❌ | partial | ❌ | ❌ |
| Skill substrate (schema-validated) | ✅ with 7 defense layers | ❌ | ❌ | ❌ | ❌ |
| Autonomous consolidation (`think`) | ✅ | ❌ | partial | ✅ | ❌ |
| Temporal decay + half-life | ✅ biological model | ❌ | ❌ | ❌ | ❌ |
| Proactive triggers | ✅ | ❌ | ❌ | ❌ | ❌ |
| Personality traits derivation | ✅ from memory patterns | ❌ | ❌ | ❌ | ❌ |
| Storage | local SQLite + WAL | hosted | local | local + hosted | filesystem |
| License | MIT (engine AGPL-3.0) | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT |
| Peer-reviewed paper | ✅ [Zenodo](https://doi.org/10.5281/zenodo.20128887) | ❌ | ✅ MemGPT paper | ❌ | ❌ |
| Same-day patch cadence for issues | ✅ (avg <2h on v0.8.x) | varies | varies | varies | n/a |

Comparisons reflect public-facing capabilities as of May 2026. PRs welcome to correct any rows.

## Cite this work

If you use YantrikDB in academic or research context, please cite the substrate paper:

```bibtex
@misc{sarkar2026skill,
  author       = {Sarkar, Pranab},
  title        = {Skill as Memory, Not Document: A Database-Native Substrate for Agent Skill Catalogs},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20128887},
  url          = {https://doi.org/10.5281/zenodo.20128887},
  orcid        = {0009-0009-8683-1481}
}
```

Plain text citation:

> Sarkar, P. (2026). *Skill as Memory, Not Document: A Database-Native Substrate for Agent Skill Catalogs*. Zenodo. https://doi.org/10.5281/zenodo.20128887

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
