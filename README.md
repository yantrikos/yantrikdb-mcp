<!-- mcp-name: io.github.yantrikos/yantrikdb-mcp -->
# YantrikDB MCP Server

Cognitive memory for AI agents. Works with Claude Code, Cursor, Windsurf, and any MCP-compatible client.

**Website:** [yantrikdb.com](https://yantrikdb.com) · **Docs:** [yantrikdb.com/guides/mcp](https://yantrikdb.com/guides/mcp/) · **GitHub:** [yantrikos/yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp)

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
| `YANTRIKDB_SKILLS_WRITE_ENABLED` | All | `false` | Set `true` to allow agents to author skills (see [Skill substrate](#skill-substrate-v070) below) |
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

## Tools

16 tools, full engine coverage:

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
| **C1** Time-bound gate | Gate auto-closes at the timestamp | `YANTRIKDB_SKILLS_WRITE_EXPIRES_AT=2026-12-31T00:00:00Z` | Unset = no expiry |
| **C2** Locked config | All `YANTRIKDB_SKILLS_*` env vars read once at startup | (always on) | Mutating env in a sub-process can't bypass the gate |
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
