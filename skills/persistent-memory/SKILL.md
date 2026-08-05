---
name: persistent-memory
description: >-
  Give this agent durable memory across sessions using YantrikDB. Use at the
  start of every session to load prior context, whenever the user or a job
  references past decisions, people, preferences, or earlier runs, and
  whenever something worth keeping is learned (a decision, a fact, a working
  method, a qualified lead, a benchmark result). Requires the yantrikdb MCP
  server to be connected.
license: MIT
metadata:
  author: yantrikos
  homepage: https://yantrikdb.com
  source: https://github.com/yantrikos/yantrikdb-mcp
---

# Persistent memory (YantrikDB)

You have a persistent, semantic, typed memory that survives across sessions,
provided by the `yantrikdb` MCP server. It is not a notes file: it does
similarity recall, tracks belief revisions, detects contradictions, learns
procedures, and answers "what did I believe at time t". Treat it as your
long-term memory, not as optional tooling.

Tool names may carry a harness prefix (`mcp_yantrikdb_recall`,
`mcp__yantrikdb__recall`, or bare `recall`). The unprefixed names are used
below; map them to whatever your harness registered.

## Session start — one call

Call `session` with `action="digest"` once. It returns a single briefing:
narrative head, open decisions, unresolved conflicts, pending triggers, and
stale high-importance memories. Do this before answering anything
substantive; a session that starts blank when 500 memories exist is a bug,
and it is this one.

## Before acting — recall, the right way

- Query with **one short natural-language sentence** (5–10 words), not a
  keyword list. Separate focused calls beat one broad one.
- For "what is the CURRENT value of X" (latest config, current owner, present
  status), use `memory` with `action="chain_head"` — similarity search
  returns the most-similar *revision*, which for values that change over time
  is often stale. `chain_head` returns the actual current value.
- Trust signals: hits whose `why_retrieved` says "aged", "superseded", or
  "rarely confirmed" are weak evidence. Prefer fresher hits, and say so if
  you act on a flagged one.
- Before starting a task you may have done before, call `procedure` with
  `action="surface"` — a learned method beats re-deriving one.

## During work — capture as you go

- New durable fact → `remember`. Be specific and searchable ("Client Foo
  approved the March pricing tier", not "they said yes"). Set `importance`:
  0.8–1.0 decisions and hard commitments, 0.5–0.7 useful context, 0.3–0.5
  background. Set `domain` and `source`.
- A stored fact **changed** → `correct` with a `reason` — never a second
  `remember`. This preserves history and prevents the contradiction the next
  session would otherwise inherit.
- Entity relationship learned → `graph` with `action="relate"`.
- A working method discovered (a scraping recipe that got past a blocker, a
  benchmark invocation that finally worked) → `procedure` with
  `action="learn"`.
- Do NOT store: secrets or credentials, ephemeral task chatter, anything
  derivable from the repo or files you already have. Memory is for what
  would otherwise be lost when this session ends.

## Job and pipeline agents — the operator patterns

If you are a recurring job (scraper, lead-gen, benchmark, monitor) rather
than a chat companion:

- **Namespace per pipeline.** Pass `namespace="<pipeline-name>"` on every
  call so runs of different jobs never pollute each other's recall.
- **Exactly-once on retries.** Harness retries duplicate writes. Pass
  `idempotency_key` on `remember` — same key + same text returns the same
  record instead of writing twice.
- **Dedupe against prior runs.** Before processing an item (a lead, a URL, a
  test case), `recall` it. Already stored with the same outcome → skip. This
  is what makes run #50 cheaper than run #1 instead of identical to it.
- **Store outcomes, not transcripts.** "example.com qualified: 200-seat firm,
  hiring for data roles" is a memory. The raw scraped HTML is not.
- **Resurface on schedule.** Use `trigger` to have a memory come back at a
  time or on a condition ("re-check this lead in two weeks") instead of
  hoping a future run thinks of it.

## End of substantial work — conditional

Only when the session was long or changed real state: call `think` to
consolidate and surface contradictions. If it reports conflicts, resolve them
with `conflict` `action="resolve"` or ask the user — do not leave a
contradiction for the next session to trip over. Short read-only sessions
skip this.

## The habit that makes it work

Every substantive turn either **reads from** memory (recall) or **writes
to** it (remember). A turn that does neither leaves no trace, and the next
session starts closer to blank than it should. If you notice three
substantive turns without touching memory, pause and recall.
