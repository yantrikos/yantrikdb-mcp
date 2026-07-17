"""MCP tool implementations for YantrikDB cognitive memory engine."""

import json
import logging
import os
import time

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from .server import mcp

log = logging.getLogger("yantrikdb.mcp.tools")


# ── Tool profiles (v0.10.0) ──
# The full tools/list schema costs ~9.8k tokens per session on every client
# BEFORE any call is made. The `core` profile advertises only the 10
# golden-path tools the injected INSTRUCTIONS choreography uses (cold-start
# digest, recall/remember/correct capture, chain_head current-value, graph
# relate, procedure surface/learn, think + conflict wrap-up, forget) —
# a ~36% schema reduction. Design per the sol-converged verdict: profiles
# expose FEWER INTACT tools; no mergers, no renamed aliases (aliases erase
# the savings), no router meta-tool (destroys permission-UI honesty).
# Default is `full` so no existing deployment loses tools silently; `core`
# is opt-in via YANTRIKDB_TOOL_PROFILE=core.
_PROFILE = os.environ.get("YANTRIKDB_TOOL_PROFILE", "full").strip().lower()
if _PROFILE not in ("core", "full"):
    log.warning("Unknown YANTRIKDB_TOOL_PROFILE=%r — using 'full'", _PROFILE)
    _PROFILE = "full"


def _specialist_tool(**kw):
    """Decorator for tools advertised only in the `full` profile.

    In the `core` profile the function is left unregistered — the tool
    simply doesn't appear in tools/list, so agents can't half-see it.
    (Honest surface: absent beats advertised-but-degraded.)"""
    if _PROFILE == "core":
        return lambda fn: fn
    return mcp.tool(**kw)


def _get_db(ctx: Context):
    """Get the YantrikDB instance from the lifespan context."""
    if ctx is None:
        raise ToolError("Tool context not available — is the server running?")
    lc = ctx.request_context.lifespan_context
    lazy = lc["lazy"]
    return lazy.db


def _err(msg, **extra):
    """Soft error — valid call but nothing to return (not found, empty results)."""
    return json.dumps({"error": msg, **extra})


def _translate_idempotency_error(e: Exception, key: str) -> str | None:
    """Translate v0.10 engine idempotency failures into agent-actionable MCP
    errors. Branch on TYPE (PR #107), never message regex. Returns None when
    the exception isn't idempotency-related (caller re-raises).

    The (InvalidIdempotencyKey, ValueError) pairing is core's recorded
    decision: InvalidIdempotencyKey is an ENGINE verdict about the key;
    the bare ValueError is the WRAPPER's capability refusal on a
    python-fallback (ONNX/multilingual) embedder — a host-config error,
    typed deliberately differently. Both mean "this call, as configured,
    cannot be keyed" and both must name the fix, not just the symptom."""
    import yantrikdb

    conflict_cls = getattr(yantrikdb, "IdempotencyConflict", None)
    invalid_cls = getattr(yantrikdb, "InvalidIdempotencyKey", None)
    if conflict_cls is not None and isinstance(e, conflict_cls):
        # Claim-resolution territory — usually same key with DIFFERENT
        # payload. The message carries the existing rid; surface it.
        return _err(
            f"idempotency conflict on key {key!r}: {e}. Same key must carry "
            f"the same text — use a new key for new content, or drop the key "
            f"to record unconditionally.",
            idempotency_key=key,
        )
    if (invalid_cls is not None and isinstance(e, invalid_cls)) or isinstance(e, ValueError):
        backend = os.environ.get("YANTRIKDB_EMBEDDER", "auto")
        return _err(
            f"idempotency_key not usable here: {e}. Keys require the engine "
            f"embedder (bundled) backend; this server runs "
            f"YANTRIKDB_EMBEDDER={backend!r}. Drop the key, or run the "
            f"bundled backend for exactly-once writes.",
            idempotency_key=key,
        )
    return None


# ── v0.10.0 response-shaping helpers ──
# Token optimization: engine responses carry full-precision floats and,
# occasionally, null fields that cost tokens on every call without helping
# the agent. These helpers shape a response before serialization.
#
# DELIBERATELY NOT a recursive float-rounder: timestamps, thresholds,
# calibration weights, user-supplied values, and mutation receipts are
# contract data that must not be silently truncated (and in HTTP-cluster
# mode the raw value is NOT re-derivable client-side). Rounding is applied
# per-field, explicitly, only to display-oriented ranking values.

_FLOAT_ROUND_PLACES = 3


def _prune_nulls(d: dict, keys: tuple[str, ...]) -> dict:
    """Drop ONLY the named keys when their value is None. Scoped by an
    explicit allowlist — a field is prunable only when "absent" carries the
    same meaning as "null" for that field (e.g. `emotional_state`). Fields
    where null is semantically load-bearing (`active_session`,
    `narrative_head`, mutation receipts) must never be passed here."""
    return {k: v for k, v in d.items() if not (k in keys and v is None)}


# Recall result fields where null == absent (safe to prune).
_RECALL_PRUNABLE_NULLS = ("emotional_state",)


# ── 1. remember ──


@mcp.tool(annotations=ToolAnnotations(title="Remember", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def remember(
    text: str | None = None,
    memory_type: str = "semantic",
    importance: float = 0.5,
    domain: str = "general",
    source: str = "user",
    valence: float = 0.0,
    metadata: dict | None = None,
    namespace: str = "default",
    certainty: float = 0.8,
    emotional_state: str | None = None,
    memories: list[dict] | None = None,
    summary: str | None = None,
    idempotency_key: str | None = None,
    ctx: Context = None,
) -> str:
    """Store one or more memories in persistent cognitive memory.

    WHEN TO USE: Call proactively whenever the conversation reveals something
    worth remembering — decisions, preferences, facts about people, project context.
    Do NOT store ephemeral task details, code snippets, or git-derivable info.

    SINGLE:  remember(text="User prefers dark mode", domain="preference", importance=0.7)
    BATCH:   remember(memories=[{"text": "Alice is DevOps lead", "domain": "people"}, ...])
    DRAFT:   remember(summary="...long end-of-session summary...") — v0.8.0+ engine
             atomizes the summary into linked semantic facts; useful for the
             end-of-session auto-capture pattern.

    IMPORTANCE: 0.8-1.0 critical decisions | 0.5-0.7 useful context | 0.3-0.5 background

    Args:
        text: Memory text (for single memory). Be specific and searchable.
        memory_type: "semantic" (facts), "episodic" (events), "procedural" (how-to).
        importance: 0.0-1.0. Higher = remembered longer.
        domain: "work", "preference", "architecture", "people", "infrastructure", "health", "finance", "general".
        source: "user", "inference", "document", "system".
        valence: Emotional tone (-1.0 to 1.0). 0.0 neutral.
        metadata: Optional key-value pairs.
        namespace: For per-project isolation.
        certainty: Confidence 0.0-1.0.
        emotional_state: joy, frustration, excitement, concern, neutral.
        memories: List of memory dicts for batch.
        summary: For draft mode — long summary that the engine atomizes.
        idempotency_key: v0.10 engine — makes the write exactly-once: retrying
            with the same key + same text returns the SAME rid with no second
            write; same key + different text is an error. Engine-embedder
            (bundled) backend only. On batch, the key scopes per item as
            "{key}:{index}" if the atomic batch path is unavailable.
    """
    db = _get_db(ctx)

    # Draft mode (v0.8.0+) — engine atomizes a summary into linked facts
    if summary is not None:
        if not summary.strip():
            raise ToolError("summary must be non-empty when provided")
        result = db.draft_memories_from_summary(summary.strip(), namespace=namespace, domain=domain)
        return json.dumps(result if isinstance(result, dict) else {"drafted": result})

    # Batch mode — uses record_batch() (v0.9.0) under the hood for one-shot insert
    if memories:
        valid_types = ("semantic", "episodic", "procedural")
        inputs = []
        for i, mem in enumerate(memories):
            t = (mem.get("text") or "").strip()
            if not t:
                raise ToolError(f"memories[{i}].text must be non-empty")
            mt = mem.get("memory_type", "semantic")
            if mt not in valid_types:
                raise ToolError(f"memories[{i}].memory_type must be one of {valid_types}, got '{mt}'")
            inputs.append({
                "text": t,
                "memory_type": mt,
                "importance": max(0.0, min(1.0, mem.get("importance", 0.5))),
                "valence": max(-1.0, min(1.0, mem.get("valence", 0.0))),
                "metadata": mem.get("metadata", {}),
                "namespace": mem.get("namespace", namespace),
                "certainty": max(0.0, min(1.0, mem.get("certainty", 0.8))),
                "domain": mem.get("domain", "general"),
                "source": mem.get("source", "user"),
                "emotional_state": mem.get("emotional_state"),
            })

        # Prefer record_batch (v0.9.0+) for atomic batch insert; fall back to
        # loop on older engines OR HTTP backend that may not expose it.
        if hasattr(db, "record_batch"):
            try:
                results = db.record_batch(inputs)
                if isinstance(results, list):
                    return json.dumps({"rids": results, "count": len(results), "status": "recorded"})
            except Exception as e:
                # Fall through to the per-item loop — but never silently:
                # if record_batch partially wrote before failing, an unkeyed
                # loop re-records those items. With a caller-supplied
                # idempotency_key the loop below keys each item
                # "{key}:{index}" (core's contracted design), so the retry
                # dedupes instead of double-writing.
                log.warning(
                    "record_batch failed (%s: %s) — falling back to per-item "
                    "loop for %d memories%s",
                    type(e).__name__, e, len(inputs),
                    " (keyed per-item — retry-safe)" if idempotency_key
                    else "; duplicates possible if the batch partially committed",
                )

        results = []
        for i, mem in enumerate(inputs):
            item_kwargs = dict(
                memory_type=mem["memory_type"],
                importance=mem["importance"],
                valence=mem["valence"],
                metadata=mem["metadata"],
                namespace=mem["namespace"],
                certainty=mem["certainty"],
                domain=mem["domain"],
                source=mem["source"],
                emotional_state=mem["emotional_state"],
            )
            if idempotency_key:
                item_kwargs["idempotency_key"] = f"{idempotency_key}:{i}"
            try:
                rid = db.record(mem["text"], **item_kwargs)
            except Exception as e:
                if idempotency_key:
                    translated = _translate_idempotency_error(e, f"{idempotency_key}:{i}")
                    if translated is not None:
                        return translated
                raise
            results.append(rid)
        return json.dumps({"rids": results, "count": len(results), "status": "recorded"})

    # Single mode
    if not text or not text.strip():
        # Dogfood finding: an agent that passes the memory body under a
        # wrong parameter name (content=/message=/body=) has it silently
        # dropped by schema validation and lands here — so name the likely
        # cause, not just the symptom.
        raise ToolError(
            "text must be non-empty. The memory body goes in the `text` "
            "parameter (content=/message=/body= are not recognized and are "
            "silently dropped). For batch use memories=[...], for "
            "end-of-session capture use summary=."
        )
    text = text.strip()

    importance = max(0.0, min(1.0, importance))
    valence = max(-1.0, min(1.0, valence))
    certainty = max(0.0, min(1.0, certainty))

    valid_types = ("semantic", "episodic", "procedural")
    if memory_type not in valid_types:
        raise ToolError(f"memory_type must be one of {valid_types}, got '{memory_type}'")

    record_kwargs = dict(
        memory_type=memory_type, importance=importance, valence=valence,
        metadata=metadata or {}, namespace=namespace, certainty=certainty,
        domain=domain, source=source, emotional_state=emotional_state,
    )
    if idempotency_key:
        record_kwargs["idempotency_key"] = idempotency_key
    try:
        rid = db.record(text, **record_kwargs)
    except Exception as e:
        if idempotency_key:
            translated = _translate_idempotency_error(e, idempotency_key)
            if translated is not None:
                return translated
        raise
    return json.dumps({"rid": rid, "status": "recorded"})


# ── 2. recall ──


@mcp.tool(annotations=ToolAnnotations(title="Recall", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def recall(
    query: str,
    top_k: int = 10,
    memory_type: str | None = None,
    domain: str | None = None,
    source: str | None = None,
    namespace: str | None = None,
    include_consolidated: bool = False,
    include_superseded: bool = False,
    expand_entities: bool = True,
    refine_from: str | None = None,
    refine_exclude: list[str] | None = None,
    ctx: Context = None,
) -> str:
    """Search memories by semantic similarity, or refine low-confidence results.

    MODES:
    - **Search** (default): recall("project architecture decisions")
    - **Refine**: recall("PostgreSQL vs MySQL decision", refine_from="database choice", refine_exclude=["rid1"])
    (Relevance feedback moved to memory(action="feedback") in v0.10 — recall
    is now purely read-only.)

    WHEN TO USE: conversation start (summarize the user's first message); when
    the user references past decisions, people, preferences, or "last time";
    when unsure about something the user assumes you know. Refine when first
    confidence < 0.5. After USING a recalled memory, reinforce it via
    memory(action="feedback", rid=..., feedback="relevant").
    For "what is the CURRENT/latest X", prefer memory(action="chain_head") —
    similarity favors the most-similar revision, not the newest.

    QUERY: one short natural-language sentence (5-10 words), NOT a keyword
    list — keyword stuffing degrades quality. One focused question per call;
    separate calls for separate topics.

    TRUST SIGNALS: each hit's `why_retrieved` may carry staleness warnings
    ("aged", "rarely confirmed", "superseded by a newer record"). Treat
    flagged hits as weak evidence — prefer fresher results or chain_head,
    and note the flag if you act on one anyway.

    Args:
        query: Short natural language sentence (5-10 words). NOT a keyword list.
        top_k: Max results (default 10). 3-5 for focused, 10-20 for broad.
        memory_type: Filter: "semantic", "episodic", "procedural".
        domain: Filter: "work", "preference", "architecture", "people", etc.
        source: Filter: "user", "inference", "document", "system".
        namespace: Filter by namespace.
        include_consolidated: Include merged memories.
        include_superseded: v0.10 — recall EXCLUDES superseded records by
            default (current-by-default). Set True only for history /
            archaeology over a revision chain.
        expand_entities: Use knowledge graph boosting (default True).
        refine_from: Original query text to refine from. query becomes the refinement.
        refine_exclude: Memory IDs to exclude when refining.
    """
    db = _get_db(ctx)

    # Pre-v0.10 feedback calls used query="" with feedback_rid — that param
    # moved to memory(action="feedback") and schema validation now drops it,
    # which would land here as a bare empty search. Name the migration.
    if (not query or not query.strip()) and not refine_from:
        raise ToolError(
            "query must be non-empty. (Recall feedback moved to "
            "memory(action='feedback', rid=..., feedback='relevant'|"
            "'irrelevant') in v0.10 — recall is now read-only.)"
        )

    # Refine mode
    if refine_from:
        original_emb = db.embed(refine_from)
        response = db.recall_refine(
            original_query_embedding=original_emb, refinement_text=query,
            original_rids=refine_exclude or [], top_k=top_k,
            namespace=namespace, domain=domain, source=source,
        )
        items = [
            {"rid": r["rid"], "text": r["text"], "type": r["type"],
             "score": round(r["score"], 4),
             "importance": round(r["importance"], _FLOAT_ROUND_PLACES),
             "created_at": r["created_at"]}
            for r in response["results"]
        ]
        hints = [
            {"hint_type": h["hint_type"], "suggestion": h["suggestion"], "related_entities": h["related_entities"]}
            for h in response["hints"]
        ]
        return json.dumps({"count": len(items), "results": items, "confidence": round(response["confidence"], 4), "hints": hints})

    # Search mode (default)
    if include_superseded:
        # recall_with_response has no include_superseded flag (gap reported
        # to core 2026-07-17) — the archaeology path goes through db.recall,
        # which returns the same per-item fields minus the hints envelope.
        rows = db.recall(
            query=query, top_k=top_k, memory_type=memory_type,
            include_consolidated=include_consolidated,
            expand_entities=expand_entities,
            namespace=namespace, domain=domain, source=source,
            include_superseded=True,
        )
        response = {
            "results": rows or [],
            "confidence": max((r.get("score", 0.0) for r in rows or []), default=0.0),
            "hints": [],
        }
    else:
        response = db.recall_with_response(
            query=query, top_k=top_k, memory_type=memory_type,
            include_consolidated=include_consolidated, expand_entities=expand_entities,
            namespace=namespace, domain=domain, source=source,
        )
    items = []
    for r in response["results"]:
        item = {
            "rid": r["rid"], "text": r["text"], "type": r["type"],
            # A1: round importance — engine emits a raw temporal-decay float
            # like 0.8955510184601201; three places is well past what any
            # agent reasons over. score stays at 4 (ranking granularity).
            "score": round(r["score"], 4),
            "importance": round(r["importance"], _FLOAT_ROUND_PLACES),
            "why_retrieved": r["why_retrieved"],
            "emotional_state": r.get("emotional_state"),
        }
        # A3: prune only the allowlisted safe null (emotional_state).
        items.append(_prune_nulls(item, _RECALL_PRUNABLE_NULLS))
    hints = [
        {"hint_type": h["hint_type"], "suggestion": h["suggestion"]}
        for h in response["hints"]
    ]
    return json.dumps({
        "count": len(items), "results": items,
        "confidence": round(response["confidence"], 4),
        "hints": hints,
    })


# ── 3. forget ──


@mcp.tool(annotations=ToolAnnotations(title="Forget", readOnlyHint=False, destructiveHint=True, idempotentHint=True, openWorldHint=False))
def forget(
    rid: str | None = None,
    rids: list[str] | None = None,
    ctx: Context = None,
) -> str:
    """Permanently forget (tombstone) one or more memories.

    WHEN TO USE: When the user explicitly asks to forget something, or when a memory
    is clearly wrong and correction isn't appropriate. Prefer `correct` over `forget`
    when the memory just needs updating.

    Args:
        rid: Single memory ID to forget.
        rids: List of memory IDs to forget (batch mode).
    """
    if not rid and not rids:
        raise ToolError("rid or rids required")

    db = _get_db(ctx)
    if rids:
        forgotten = 0
        for r in rids:
            if db.forget(r):
                forgotten += 1
        return json.dumps({"forgotten": forgotten, "total": len(rids)})

    result = db.forget(rid)
    return json.dumps({"rid": rid, "forgotten": result})


# ── 4. correct ──


@mcp.tool(annotations=ToolAnnotations(title="Correct", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
def correct(
    rid: str,
    reason: str,
    new_text: str | None = None,
    new_importance: float | None = None,
    new_valence: float | None = None,
    metadata_merge: dict | None = None,
    ctx: Context = None,
) -> str:
    """Correct an existing memory in-place with a revision-history entry
    (engine v0.7.20+, Issue #47).

    WHEN TO USE: When the user corrects a recalled fact.
    - "Actually, we're using Python 3.12, not 3.11" → correct the memory.

    Preserves history via an append-only revision entry keyed on `reason`.
    Entity relationships stay attached to the same rid (in-place mutation,
    not a tombstone+new-rid dance).

    Args:
        rid: The memory ID to correct.
        reason: **Required** — why the correction was made. Non-empty.
            Recorded on the revision-history entry so future recall +
            audit can reconstruct why the memory changed.
        new_text: Optional new text (pass None to keep existing).
        new_importance: Optional updated importance (0.0-1.0).
        new_valence: Optional updated valence (-1.0 to 1.0).
        metadata_merge: Optional dict to merge into existing metadata
            (None = keep as-is).
    """
    if not reason or not reason.strip():
        raise ToolError("reason must be non-empty")
    if new_text is not None and not new_text.strip():
        raise ToolError("new_text may be omitted, but if provided must be non-empty")

    db = _get_db(ctx)
    try:
        result = db.correct(
            rid,
            reason.strip(),
            new_text=(new_text.strip() if new_text else None),
            metadata_merge=metadata_merge,
            new_importance=new_importance,
            new_valence=new_valence,
        )
    except Exception as e:
        return _err(str(e), rid=rid)
    # v0.7.20+ mutates in place: corrected_rid == rid. Return both for
    # backward-compatible response shape.
    return json.dumps({
        "rid": result.get("corrected_rid") or rid,
        "corrected_rid": result.get("corrected_rid") or rid,
        "original_rid": result.get("original_rid") or rid,
        "reason": reason.strip(),
    })


# ── 5. think ──


@mcp.tool(annotations=ToolAnnotations(title="Think", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def think(
    run_consolidation: bool = True,
    run_conflict_scan: bool = True,
    run_pattern_mining: bool = False,
    consolidation_time_window_days: float = 7.0,
    consolidation_limit: int = 5,
    maintenance_cycle: bool = False,
    last_cycle_only: bool = False,
    dry_run: bool = True,
    burn_down_conflicts: bool = True,
    prune_triggers_too: bool = True,
    max_pending_triggers: int = 64,
    recalibrate_importance: bool = True,
    backfill_entities: bool = True,
    auto_relate_in_cycle: bool = True,
    max_auto_relate_edges: int = 500,
    split_oversized: bool = False,
    split_min_chars: int = 1500,
    repair_artifacts: bool = False,
    maintenance_op: str | None = None,
    ctx: Context = None,
) -> str:
    """Run incremental cognitive maintenance — processes a small batch per call.

    DESIGNED TO BE CALLED OFTEN: Each call processes ~5 memories (configurable).
    Running regularly (e.g. at end of conversation) gradually maintains the
    entire database without blocking. Safe to call frequently.

    MODES:
    - Default: incremental think() — consolidation + conflict scan + (optional)
      pattern mining on a small batch.
    - maintenance_cycle=True: run the v0.9.0 autonomous-hygiene "sleep cycle" —
      think + burn-down-conflicts + prune-triggers + recalibrate-importance +
      backfill-entities + auto-relate (+ optional split_oversized + repair_artifacts).
    - last_cycle_only=True: just fetch the last persisted maintenance-cycle
      summary (read-only, no work performed).
    - maintenance_op="backfill_entities"|"rebuild_vec_index"|"rebuild_graph_index":
      run ONE targeted index-maintenance op and return. (Moved here from
      stats in v0.10 so stats could become read-only.)

    Args:
        run_consolidation: Merge similar memories (default on).
        run_conflict_scan: Detect contradictions (default on).
        run_pattern_mining: Mine cross-domain patterns (default off, slow).
        consolidation_time_window_days: Only consolidate memories within this
            window (default 7 days).
        consolidation_limit: Batch size — max memories to process per call
            (default 5). Keep small for fast returns.
        maintenance_cycle: Run the full autonomous hygiene cycle instead.
        last_cycle_only: Just fetch the last cycle summary (read-only).
        dry_run: For maintenance_cycle — preview without persisting changes.
        burn_down_conflicts / prune_triggers_too / max_pending_triggers /
        recalibrate_importance / backfill_entities / auto_relate_in_cycle /
        max_auto_relate_edges / split_oversized / split_min_chars /
        repair_artifacts: Maintenance-cycle knobs.
    """
    db = _get_db(ctx)

    # Targeted index-maintenance op (moved from stats in v0.10)
    if maintenance_op is not None:
        valid_ops = ("backfill_entities", "rebuild_vec_index", "rebuild_graph_index")
        if maintenance_op not in valid_ops:
            raise ToolError(f"maintenance_op must be one of {valid_ops}")
        t0 = time.time()
        if maintenance_op == "backfill_entities":
            result = db.backfill_memory_entities()
        elif maintenance_op == "rebuild_vec_index":
            db.rebuild_vec_index()
            result = "ok"
        else:
            db.rebuild_graph_index()
            result = "ok"
        elapsed_ms = round((time.time() - t0) * 1000, 1)
        return json.dumps({"action": maintenance_op, "result": result, "elapsed_ms": elapsed_ms})

    # last_cycle_only — read-only short-circuit
    if last_cycle_only:
        last = db.last_maintenance_cycle()
        return json.dumps({"last_maintenance_cycle": last})

    # Full v0.9.0 maintenance cycle
    if maintenance_cycle:
        result = db.run_maintenance_cycle(
            run_think=run_consolidation,
            burn_down_conflicts=burn_down_conflicts,
            prune_triggers=prune_triggers_too,
            max_pending_triggers=max_pending_triggers,
            recalibrate_importance=recalibrate_importance,
            backfill_entities=backfill_entities,
            auto_relate=auto_relate_in_cycle,
            max_auto_relate_edges=max_auto_relate_edges,
            split_oversized=split_oversized,
            split_min_chars=split_min_chars,
            repair_artifacts=repair_artifacts,
        )
        return json.dumps({"maintenance_cycle": result if isinstance(result, dict) else {"result": result}})

    # Default incremental think()
    config = {
        "run_consolidation": run_consolidation,
        "run_conflict_scan": run_conflict_scan,
        "run_pattern_mining": run_pattern_mining,
        "consolidation_time_window_days": consolidation_time_window_days,
        "consolidation_limit": consolidation_limit,
        "min_active_memories": 5,
    }
    result = db.think(config)
    # Also fetch patterns since they're often needed right after think
    pattern_list = db.get_patterns(status="active", limit=10)
    triggers = [
        {"trigger_type": t["trigger_type"], "reason": t["reason"],
         "urgency": t["urgency"], "suggested_action": t["suggested_action"]}
        for t in result["triggers"]
    ]
    return json.dumps({
        "triggers": triggers,
        "consolidation_count": result["consolidation_count"],
        "conflicts_found": result["conflicts_found"],
        "patterns_new": result["patterns_new"],
        "patterns_updated": result["patterns_updated"],
        "expired_triggers": result["expired_triggers"],
        "duration_ms": round(result["duration_ms"], 2),
        "patterns": pattern_list[:5] if pattern_list else [],
    })


# ── 6. memory ──


@mcp.tool(annotations=ToolAnnotations(title="Memory", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def memory(
    action: str,
    rid: str | None = None,
    importance: float | None = None,
    limit: int = 50,
    offset: int = 0,
    domain: str | None = None,
    memory_type: str | None = None,
    namespace: str | None = None,
    sort_by: str = "created_at",
    text_contains: str | None = None,
    feedback: str | None = None,
    feedback_query: str | None = None,
    feedback_score: float | None = None,
    feedback_rank: int | None = None,
    ctx: Context = None,
) -> str:
    """Manage individual memories — get, list, search, update importance, archive,
    hydrate, relevance feedback, fetch a chain-shaped namespace's head, or query
    revision history.

    ACTIONS:
    - "get":               Retrieve a single memory by rid.
    - "list":              Browse memories with filters.
    - "search":            Keyword substring search.
    - "update_importance": Change a memory's importance score.
    - "archive":           Move to cold storage.
    - "hydrate":           Restore archived memory.
    - "feedback":          v0.10 — relevance feedback on a recalled memory
                           (needs rid + feedback="relevant"|"irrelevant").
                           Call after USING a recalled memory; it tunes future
                           retrieval. (Moved here from recall, which is now
                           read-only.)
    - "chain_head":        The CURRENT value of a chain-shaped namespace
                           (narrative / decision / config chains). Use this —
                           not recall — for "what is the current/latest X":
                           similarity search favors the most-similar revision,
                           chain_head returns the newest.
    - "history":           v0.8.0 — revision history for a single rid (needs rid).

    Args:
        See action docs above. New args:
        namespace: Required for chain_head — the chain-shaped namespace.
        rid: Required for history/feedback — the record acted on.
        feedback: For feedback — "relevant" or "irrelevant".
        feedback_query: For feedback — the query that surfaced the memory.
        feedback_score / feedback_rank: For feedback — retrieval context.
    """
    valid = ("get", "list", "search", "update_importance", "archive", "hydrate",
             "feedback", "chain_head", "history")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    if action == "feedback":
        if not rid:
            raise ToolError("rid required for feedback")
        if feedback not in ("relevant", "irrelevant"):
            raise ToolError("feedback must be 'relevant' or 'irrelevant'")
        db = _get_db(ctx)
        db.recall_feedback(
            rid=rid, feedback=feedback, query_text=feedback_query or None,
            score_at_retrieval=feedback_score, rank_at_retrieval=feedback_rank,
        )
        return json.dumps({"rid": rid, "feedback": feedback, "status": "recorded"})

    db = _get_db(ctx)

    if action == "get":
        if not rid:
            raise ToolError("rid required for get")
        mem = db.get(rid)
        if mem is None:
            return _err("Memory not found", rid=rid)
        return json.dumps({
            "rid": mem["rid"], "text": mem["text"], "type": mem["type"],
            "importance": mem["importance"], "valence": mem["valence"],
            "created_at": mem["created_at"], "last_access": mem["last_access"],
            "consolidation_status": mem["consolidation_status"],
            "storage_tier": mem["storage_tier"], "metadata": mem["metadata"],
            "certainty": mem["certainty"], "domain": mem["domain"],
            "source": mem["source"], "emotional_state": mem["emotional_state"],
        })

    if action == "list":
        limit = max(1, min(200, limit))
        result = db.list_memories(limit=limit, offset=max(0, offset), domain=domain,
                                  memory_type=memory_type, namespace=namespace, sort_by=sort_by)
        items = [
            {"rid": m["rid"], "type": m["type"], "text": m["text"],
             "importance": m["importance"], "domain": m["domain"],
             "created_at": m["created_at"], "namespace": m["namespace"]}
            for m in result["memories"]
        ]
        return json.dumps({"count": len(items), "total": result["total"], "offset": result["offset"], "memories": items})

    if action == "search":
        if not text_contains or not text_contains.strip():
            raise ToolError("text_contains required for search")
        needle = text_contains.strip().lower()
        limit = max(1, min(200, limit))
        # Fetch in pages and filter client-side
        matched = []
        scan_offset = 0
        batch_size = 200
        while len(matched) < limit:
            result = db.list_memories(limit=batch_size, offset=scan_offset, domain=domain,
                                      memory_type=memory_type, namespace=namespace, sort_by="created_at")
            if not result["memories"]:
                break
            for m in result["memories"]:
                if needle in m["text"].lower():
                    matched.append({
                        "rid": m["rid"], "type": m["type"], "text": m["text"],
                        "importance": m["importance"], "domain": m["domain"],
                        "created_at": m["created_at"]})
                    if len(matched) >= limit:
                        break
            scan_offset += batch_size
            if scan_offset >= result["total"]:
                break
        return json.dumps({"count": len(matched), "query": text_contains, "memories": matched})

    if action == "update_importance":
        if not rid:
            raise ToolError("rid required")
        if importance is None:
            raise ToolError("importance required")
        mem = db.get(rid)
        if mem is None:
            return _err("Memory not found", rid=rid)
        # Engine v0.7.20+ signature: reason is required + first positional
        # after rid; new_text is optional (we keep the existing text here
        # since this action only mutates importance).
        result = db.correct(
            rid,
            f"Importance adjusted from {mem['importance']} to {importance}",
            new_importance=max(0.0, min(1.0, importance)),
        )
        corrected_rid = result.get("corrected_rid") or rid
        return json.dumps({"rid": corrected_rid, "old_importance": mem["importance"],
                           "new_importance": importance, "status": "updated"})

    if action == "archive":
        if not rid:
            raise ToolError("rid required")
        archived = db.archive(rid)
        if not archived:
            return _err(f"Memory '{rid}' not found or already archived")
        return json.dumps({"archived": rid, "status": "cold"})

    if action == "hydrate":
        if not rid:
            raise ToolError("rid required")
        hydrated = db.hydrate(rid)
        if not hydrated:
            return _err(f"Memory '{rid}' not found or already hot")
        return json.dumps({"hydrated": rid, "status": "hot"})

    if action == "chain_head":
        if not namespace:
            raise ToolError("namespace required for chain_head")
        head = db.chain_head(namespace)
        if head is None:
            return _err(f"no chain head in namespace {namespace!r}", namespace=namespace)
        return json.dumps(head if isinstance(head, dict) else {"head": head})

    if action == "history":
        if not rid:
            raise ToolError("rid required for history")
        revs = db.history(rid)
        return json.dumps({"rid": rid, "count": len(revs or []), "revisions": revs or []})


# ── 7. graph ──


@mcp.tool(annotations=ToolAnnotations(title="Graph", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def graph(
    action: str,
    entity: str | None = None,
    target: str | None = None,
    relationship: str = "related_to",
    weight: float = 1.0,
    rid: str | None = None,
    pattern: str | None = None,
    limit: int = 20,
    days: float = 90.0,
    namespace: str | None = None,
    # v0.9.0: record-to-record link primitives + auto_relate
    source_rid: str | None = None,
    target_rid: str | None = None,
    link_type: str = "related_to",
    direction: str = "both",
    dry_run: bool = True,
    max_edges: int = 500,
    query: str | None = None,
    top_k: int = 10,
    expand_links: int = 1,
    ctx: Context = None,
) -> str:
    """Knowledge graph operations — entity relationships, memory↔entity links,
    record-to-record links, co-occurrence auto-relate, and link-expanded recall.

    ACTIONS:
    - "relate":           Entity↔entity relationship (legacy).
    - "edges":            Get all relationships for entity.
    - "link":             Link a memory (rid) to an entity (legacy).
    - "search":           Find entities by pattern.
    - "profile":          Rich entity profile.
    - "depth":            How deeply the system knows an entity.
    - "auto_relate":      v0.8.0 — co-occurrence-driven edge backfill.
                          Set dry_run=False to persist.
    - "record_link":      v0.9.0 — add a record-to-record link
                          (needs source_rid + target_rid + link_type).
    - "record_unlink":    v0.9.0 — remove a record-to-record link.
    - "linked_records":   v0.9.0 — traverse links from rid (direction =
                          "outbound" | "inbound" | "both", optional link_type filter).
    - "recall_with_links": v0.9.0 — semantic recall with N-hop link expansion.

    Args:
        action: One of the actions above.
        entity / target / relationship / weight / rid / pattern / limit /
        days / namespace: Legacy entity-graph args.
        source_rid / target_rid / link_type: For record_link / record_unlink.
        direction: For linked_records — "outbound" / "inbound" / "both".
        dry_run: For auto_relate — preview without persisting.
        max_edges: For auto_relate — cap edges proposed/created.
        query: For recall_with_links — natural language search.
        top_k: For recall_with_links — max seed results.
        expand_links: For recall_with_links — hop budget for traversal.
    """
    valid = ("relate", "edges", "link", "search", "profile", "depth",
             "auto_relate", "record_link", "record_unlink",
             "linked_records", "recall_with_links")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

    if action == "relate":
        if not entity or not target:
            raise ToolError("entity and target required for relate")
        edge_id = db.relate(entity, target, relationship, weight)
        return json.dumps({"edge_id": edge_id, "source": entity, "target": target, "relationship": relationship})

    if action == "edges":
        if not entity:
            raise ToolError("entity required")
        edges = db.get_edges(entity)
        items = [{"edge_id": e["edge_id"], "src": e["src"], "dst": e["dst"],
                  "rel_type": e["rel_type"], "weight": e["weight"]} for e in edges]
        return json.dumps({"entity": entity, "count": len(items), "edges": items})

    if action == "link":
        if not rid or not entity:
            raise ToolError("rid and entity required for link")
        try:
            db.link_memory_entity(rid, entity)
            return json.dumps({"rid": rid, "entity": entity, "linked": True})
        except Exception as e:
            return _err(str(e), rid=rid, entity=entity)

    if action == "search":
        if not pattern:
            raise ToolError("pattern required for search")
        entities = db.search_entities(pattern=pattern, limit=limit)
        items = [{"name": e["name"], "type": e.get("entity_type"),
                  "mention_count": e.get("mention_count", 0)} for e in entities]
        return json.dumps({"pattern": pattern, "count": len(items), "entities": items})

    if action == "profile":
        if not entity:
            raise ToolError("entity required")
        profile = db.entity_profile(entity, days, namespace)
        return json.dumps(profile)

    if action == "depth":
        if not entity:
            raise ToolError("entity required")
        depth = db.relationship_depth(entity, namespace)
        return json.dumps(depth)

    if action == "auto_relate":
        result = db.auto_relate(dry_run=dry_run, max_edges=max_edges)
        return json.dumps(result if isinstance(result, dict) else {"result": result})

    if action == "record_link":
        if not source_rid or not target_rid:
            raise ToolError("source_rid and target_rid required for record_link")
        link_id = db.link(source_rid, target_rid, link_type)
        return json.dumps({
            "link_id": link_id, "source_rid": source_rid,
            "target_rid": target_rid, "link_type": link_type,
        })

    if action == "record_unlink":
        if not source_rid or not target_rid:
            raise ToolError("source_rid and target_rid required for record_unlink")
        removed = db.unlink(source_rid, target_rid, link_type)
        return json.dumps({"removed": removed})

    if action == "linked_records":
        if not rid:
            raise ToolError("rid required for linked_records")
        if direction not in ("outbound", "inbound", "both"):
            raise ToolError("direction must be 'outbound', 'inbound', or 'both'")
        rows = db.linked_records(rid, direction=direction, link_type=(link_type if link_type != "related_to" else None))
        return json.dumps({"rid": rid, "direction": direction, "count": len(rows or []), "linked": rows or []})

    if action == "recall_with_links":
        if not query:
            raise ToolError("query required for recall_with_links")
        result = db.recall_with_links(
            query=query, top_k=top_k, expand_links=expand_links,
            namespace=namespace,
        )
        return json.dumps(result if isinstance(result, dict) else {"results": result})


# ── 8. conflict ──


@mcp.tool(annotations=ToolAnnotations(title="Conflict", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
def conflict(
    action: str = "list",
    conflict_id: str | None = None,
    status: str | None = None,
    strategy: str | None = None,
    winner_rid: str | None = None,
    new_text: str | None = None,
    resolution_note: str | None = None,
    new_type: str | None = None,
    limit: int = 10,
    dry_run: bool = True,
    ctx: Context = None,
) -> str:
    """Manage memory conflicts (contradictions) — list, resolve, dismiss,
    reclassify, or batch-burn-down the unambiguous ones (v0.8.0+).

    ACTIONS:
    - "list":        List conflicts. Optional status filter.
    - "get":         Get single conflict by conflict_id.
    - "resolve":     Resolve with strategy: "keep_a"/"keep_b"/"keep_both"/"merge"/"dismiss".
    - "reclassify":  Reclassify conflict type.
    - "auto_resolve": v0.8.0 — burn down unambiguous conflicts in one pass.
                     Set dry_run=False to actually persist.

    Args:
        action: "list", "get", "resolve", "reclassify", "auto_resolve".
        conflict_id / status / strategy / winner_rid / new_text / resolution_note /
        new_type / limit: see action docs above.
        dry_run: For auto_resolve — preview without persisting.
    """
    valid = ("list", "get", "resolve", "reclassify", "auto_resolve")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

    if action == "auto_resolve":
        result = db.auto_resolve_conflicts(dry_run=dry_run)
        return json.dumps(result if isinstance(result, dict) else {"result": result})

    if action == "list":
        conflict_list = db.get_conflicts(status=status, limit=limit)
        items = [
            {"conflict_id": c["conflict_id"], "conflict_type": c["conflict_type"],
             "priority": c["priority"], "status": c["status"],
             "memory_a": c["memory_a"], "memory_b": c["memory_b"],
             "entity": c["entity"], "detection_reason": c["detection_reason"]}
            for c in conflict_list
        ]
        return json.dumps({"count": len(items), "conflicts": items})

    # auto_resolve already handled above; remaining actions require conflict_id
    if not conflict_id:
        raise ToolError("conflict_id required")

    if action == "get":
        c = db.get_conflict(conflict_id)
        if not c:
            return _err(f"Conflict not found: {conflict_id}")
        return json.dumps(c)

    if action == "resolve":
        valid_strategies = ("keep_a", "keep_b", "keep_both", "merge", "dismiss")
        if strategy not in valid_strategies:
            raise ToolError(f"strategy must be one of {valid_strategies}")
        if strategy == "merge" and (not new_text or not new_text.strip()):
            raise ToolError("new_text required for merge strategy")
        try:
            if strategy == "dismiss":
                db.dismiss_conflict(conflict_id, note=resolution_note)
                return json.dumps({"conflict_id": conflict_id, "strategy": "dismiss", "dismissed": True})
            result = db.resolve_conflict(conflict_id, strategy, winner_rid=winner_rid,
                                         new_text=new_text, resolution_note=resolution_note)
        except Exception as e:
            return _err(str(e), conflict_id=conflict_id)
        return json.dumps({
            "conflict_id": result["conflict_id"], "strategy": result["strategy"],
            "winner_rid": result.get("winner_rid"),
            "loser_tombstoned": result.get("loser_tombstoned", False),
            "new_memory_rid": result.get("new_memory_rid"),
        })

    if action == "reclassify":
        if not new_type:
            raise ToolError("new_type required for reclassify")
        result = db.reclassify_conflict(conflict_id, new_type)
        return json.dumps(result)


# ── 9. trigger ──


@_specialist_tool(annotations=ToolAnnotations(title="Trigger", readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def trigger(
    action: str = "pending",
    trigger_id: str | None = None,
    trigger_type: str | None = None,
    limit: int = 10,
    dry_run: bool = True,
    max_pending: int = 64,
    ctx: Context = None,
) -> str:
    """Manage proactive triggers + v0.8.0 bounded-backlog pruning.

    ACTIONS:
    - "pending":     Get pending triggers (default).
    - "history":     View past triggers.
    - "acknowledge": Mark trigger as seen.
    - "deliver":     Mark as shown to user.
    - "act":         Mark as acted upon.
    - "dismiss":     Dismiss as irrelevant.
    - "prune":       v0.8.0 — expire overdue triggers + evict oldest when over
                     `max_pending`. Set dry_run=False to actually persist.

    Args:
        action: One of the actions above.
        trigger_id: Required for acknowledge/deliver/act/dismiss.
        trigger_type: Filter by type (for pending/history).
        limit: Max results.
        dry_run: For prune — preview without persisting.
        max_pending: For prune — soft cap on the pending backlog (default 64).
    """
    valid = ("pending", "history", "acknowledge", "deliver", "act", "dismiss", "prune")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

    if action == "prune":
        result = db.prune_triggers(dry_run=dry_run, max_pending=max_pending)
        return json.dumps(result if isinstance(result, dict) else {"result": result})

    if action == "pending":
        trigger_list = db.get_pending_triggers(limit=limit)
        items = [
            {"trigger_id": t["trigger_id"], "trigger_type": t["trigger_type"],
             "urgency": t.get("urgency"), "reason": t.get("reason"),
             "suggested_action": t.get("suggested_action"), "source_rids": t.get("source_rids")}
            for t in trigger_list
        ]
        return json.dumps({"count": len(items), "triggers": items})

    if action == "history":
        trigger_list = db.get_trigger_history(trigger_type=trigger_type, limit=limit)
        items = [
            {"trigger_id": t["trigger_id"], "trigger_type": t["trigger_type"],
             "reason": t.get("reason"), "status": t.get("status")}
            for t in trigger_list
        ]
        return json.dumps({"count": len(items), "triggers": items})

    # Lifecycle actions
    if not trigger_id:
        raise ToolError("trigger_id required")
    action_map = {
        "acknowledge": db.acknowledge_trigger,
        "deliver": db.deliver_trigger,
        "act": db.act_on_trigger,
        "dismiss": db.dismiss_trigger,
    }
    result = action_map[action](trigger_id)
    return json.dumps({"trigger_id": trigger_id, "action": action, "result": result})


# ── 10. session ──


@mcp.tool(annotations=ToolAnnotations(title="Session", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def session(
    action: str,
    session_id: str | None = None,
    namespace: str = "default",
    client_id: str = "default",
    metadata: dict | None = None,
    summary: str | None = None,
    domain: str = "general",
    limit: int = 10,
    abandon_stale_hours: float | None = None,
    narrative_namespace: str | None = None,
    scope: str | None = None,
    include_gaps: bool = False,
    max_gaps: int = 5,
    max_decisions: int = 8,
    max_conflicts: int = 5,
    max_triggers: int = 5,
    snippet_chars: int = 240,
    ctx: Context = None,
) -> str:
    """Session lifecycle — start, end, history, active check, stale cleanup,
    and the v0.9.0 boot-time digest.

    ACTIONS:
    - "start":   Begin a new session. Returns session_id.
    - "end":     End a TRACKED session (needs session_id). Returns stats.
                 This closes session bookkeeping — it does NOT capture memories.
    - "capture": Segment a free-text session summary into atomic candidate
                 memories (needs summary; NO session_id — it operates on the
                 text, not on tracked-session state). Returns drafted rids.
                 Use at end of substantial work so the session leaves a trace.
    - "history": View past sessions.
    - "active":  Check if there's a running session.
    - "abandon_stale": Clean up orphaned sessions older than abandon_stale_hours.
    - "digest":  One-call boot-time briefing (v0.9.0) — narrative chain head,
                 open decisions/conflicts/triggers, top stale memories.
                 Call this at conversation start instead of N separate recalls.
                 Set include_gaps=True to fold known-unknowns (frequently-asked,
                 poorly-answered queries) into the briefing — the active-learning
                 loop. Set scope to filter content aggregates to one namespace
                 for a per-tenant digest.

    Args:
        action: "start", "end", "capture", "history", "active", "abandon_stale", "digest".
        session_id: For end.
        namespace: Memory namespace.
        client_id: Client identifier.
        metadata: For start — optional dict.
        summary: For end — optional closing note. For capture — REQUIRED,
            the session summary to segment into memories.
        domain: For capture — domain stamped on drafted memories.
        limit: For history.
        abandon_stale_hours: For abandon_stale — max age in hours.
        narrative_namespace: For digest — namespace for the narrative chain.
        scope: For digest — filter content aggregates to one namespace
            (per-tenant isolation); omit for a whole-DB digest.
        include_gaps: For digest — fold top knowledge gaps into the briefing.
        max_gaps: For digest — cap on gaps surfaced when include_gaps=True.
        max_decisions / max_conflicts / max_triggers: For digest — surface caps.
        snippet_chars: For digest — text-snippet length per item.
    """
    valid = ("start", "end", "capture", "history", "active", "abandon_stale", "digest")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

    if action == "start":
        sid = db.session_start(namespace, client_id, metadata or {})
        return json.dumps({"session_id": sid})

    if action == "end":
        if not session_id:
            raise ToolError(
                "session_id required — 'end' closes a TRACKED session. To "
                "capture a summary into memories (no session_id), use "
                "action='capture'."
            )
        result = db.session_end(session_id, summary)
        return json.dumps(result)

    if action == "capture":
        # Server ruling (2026-07-17): capture ≠ tracking-end. Capture
        # segments a summary STRING into candidate facts — no session_id
        # exists to look up. Same engine call as remember(summary=);
        # exposed here so end-of-session choreography lives on `session`.
        if not summary or not summary.strip():
            raise ToolError("summary required for capture")
        result = db.draft_memories_from_summary(
            summary.strip(), namespace=namespace, domain=domain,
        )
        # Normalize cross-backend: embedded returns {"drafted": [...]},
        # the HTTP gateway returns {"drafted": [...], "count": N}. Agents
        # get one shape either way.
        if not isinstance(result, dict):
            result = {"drafted": result}
        result.setdefault("count", len(result.get("drafted") or []))
        return json.dumps(result)

    if action == "history":
        sessions = db.session_history(namespace, client_id, limit)
        return json.dumps(sessions)

    if action == "active":
        active = db.active_session(namespace, client_id)
        return json.dumps({"active_session": active})

    if action == "abandon_stale":
        hours = abandon_stale_hours or 24.0
        count = db.session_abandon_stale(max_age_hours=hours)
        return json.dumps({"abandoned_sessions": count, "max_age_hours": hours})

    if action == "digest":
        # scope / include_gaps / max_gaps are the server v0.8.27 (and newer
        # embedded engine) additions. Older embedded engines don't accept
        # them, so pass them only when the caller opted in, and fall back to
        # the base keyword set on TypeError rather than failing the digest.
        digest_kwargs = dict(
            narrative_namespace=narrative_namespace,
            max_decisions=max_decisions,
            max_conflicts=max_conflicts,
            max_triggers=max_triggers,
            snippet_chars=snippet_chars,
        )
        extra_kwargs = {}
        if scope:
            extra_kwargs["scope"] = scope
        if include_gaps:
            extra_kwargs["include_gaps"] = True
            extra_kwargs["max_gaps"] = max_gaps
        dropped_params: list[str] = []
        try:
            digest = db.session_digest(**digest_kwargs, **extra_kwargs)
        except TypeError:
            if not extra_kwargs:
                raise
            dropped_params = sorted(extra_kwargs)
            log.warning(
                "session_digest: backend rejected %s — retrying without them "
                "(engine predates the scope/include_gaps surface)",
                dropped_params,
            )
            digest = db.session_digest(**digest_kwargs)
        # The engine returns the digest as a JSON STRING. Passing it through
        # untouched produced a double-encoded `{"digest": "{\"...\":...}"}`
        # envelope — the agent had to JSON-parse a string field to read the
        # briefing (measured in the v0.10.0 design review). Decode it here so
        # the client receives a first-class object.
        if isinstance(digest, str):
            try:
                digest = json.loads(digest)
            except json.JSONDecodeError:
                return json.dumps({"digest": digest})
        if not isinstance(digest, dict):
            digest = {"digest": digest}
        if dropped_params:
            # Honest surface: the agent asked for capabilities this engine
            # predates. Without this note, include_gaps=True silently yields
            # a digest with no knowledge_gaps and the agent can't tell
            # "no gaps exist" from "gaps weren't computed".
            digest["unsupported_params"] = dropped_params
            digest["unsupported_note"] = (
                "engine predates these digest params; they were ignored"
            )
        return json.dumps(digest)


# ── 11. temporal ──


@_specialist_tool(annotations=ToolAnnotations(title="Temporal", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def temporal(
    action: str,
    days: float = 30.0,
    limit: int = 20,
    namespace: str | None = None,
    ctx: Context = None,
) -> str:
    """Find stale or upcoming memories based on time.

    ACTIONS:
    - "stale": Important memories not accessed recently. Good for maintenance.
    - "upcoming": Memories with approaching deadlines/events. Good for proactive alerts.

    Args:
        action: "stale" or "upcoming".
        days: Inactivity threshold (stale) or look-ahead window (upcoming).
        limit: Max results.
        namespace: Optional filter.
    """
    if action not in ("stale", "upcoming"):
        raise ToolError("action must be 'stale' or 'upcoming'")

    db = _get_db(ctx)

    if action == "stale":
        memories = db.stale(days, limit, namespace)
        return json.dumps([
            {"rid": m["rid"], "text": m["text"], "importance": m["importance"],
             "days_since_access": (time.time() - m.get("last_access", time.time())) / 86400}
            for m in memories
        ])

    memories = db.upcoming(days, limit, namespace)
    return json.dumps([
        {"rid": m["rid"], "text": m["text"], "importance": m["importance"],
         "due_at": m.get("due_at"), "temporal_kind": m.get("temporal_kind")}
        for m in memories
    ])


# ── 12. procedure ──


@mcp.tool(annotations=ToolAnnotations(title="Procedure", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def procedure(
    action: str,
    text: str | None = None,
    query: str | None = None,
    rid: str | None = None,
    domain: str = "general",
    task_context: str = "",
    effectiveness: float = 0.5,
    outcome: float | None = None,
    top_k: int = 5,
    namespace: str | None = None,
    ctx: Context = None,
) -> str:
    """Procedural memory — learn, surface, and reinforce strategies.

    ACTIONS:
    - "learn": Store a procedure (needs text). What worked in a specific context.
    - "surface": Find relevant procedures (needs query). Returns ranked by effectiveness.
    - "reinforce": Update effectiveness (needs rid + outcome 0.0-1.0).

    EXAMPLES:
    - procedure(action="learn", text="For this repo, always run tests before committing", domain="work")
    - procedure(action="surface", query="how to handle code review in this repo")
    - procedure(action="reinforce", rid="abc", outcome=0.9)

    Args:
        action: "learn", "surface", "reinforce".
        text: Procedure description (for learn).
        query: What you're about to do (for surface).
        rid: Procedure ID (for reinforce).
        domain: Task domain.
        task_context: What kind of task (for learn).
        effectiveness: Initial effectiveness 0.0-1.0 (for learn).
        outcome: How well it worked 0.0-1.0 (for reinforce).
        top_k: Max results (for surface).
        namespace: Namespace.
    """
    if action not in ("learn", "surface", "reinforce"):
        raise ToolError("action must be 'learn', 'surface', or 'reinforce'")

    db = _get_db(ctx)

    if action == "learn":
        if not text or not text.strip():
            raise ToolError("text required")
        rid = db.record_procedural(text, None, domain, task_context, effectiveness, namespace or "default")
        return json.dumps({"rid": rid, "type": "procedural", "effectiveness": effectiveness})

    if action == "surface":
        if not query:
            raise ToolError("query required")
        emb = db.embed(query)
        results = db.surface_procedural(emb, query, domain if domain != "general" else None, top_k, namespace)
        return json.dumps([
            {"rid": r["rid"], "text": r["text"], "score": r["score"],
             "importance": r["importance"], "certainty": r.get("certainty", 0.5)}
            for r in results
        ])

    if action == "reinforce":
        if not rid:
            raise ToolError("rid required")
        if outcome is None:
            raise ToolError("outcome required (0.0-1.0)")
        result = db.reinforce_procedural(rid, outcome)
        return json.dumps({"rid": rid, "reinforced": result, "outcome": outcome})


# ── 13. category ──


@_specialist_tool(annotations=ToolAnnotations(title="Category", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
def category(
    action: str = "list",
    category_name: str | None = None,
    members: list[list] | None = None,
    source: str = "llm_suggested",
    ctx: Context = None,
) -> str:
    """Substitution categories for conflict detection — list, inspect, teach, or reset.

    ACTIONS:
    - "list": Show all categories with member counts.
    - "members": Show members of a specific category (needs category_name).
    - "learn": Teach new members (needs category_name + members as [[token, confidence], ...]).
    - "reset": Reset category to seed state (needs category_name).

    EXAMPLES:
    - category() → list all categories
    - category(action="members", category_name="databases")
    - category(action="learn", category_name="databases", members=[["tidb", 0.35]])
    - category(action="reset", category_name="editors_tools")

    Args:
        action: "list", "members", "learn", "reset".
        category_name: Required for members/learn/reset.
        members: For learn: [[token, confidence], ...].
        source: For learn: "llm_suggested", "user_confirmed", "seed".
    """
    valid = ("list", "members", "learn", "reset")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

    if action == "list":
        cats = db.substitution_categories()
        return json.dumps(cats)

    if not category_name:
        raise ToolError("category_name required")

    if action == "members":
        mems = db.substitution_members(category_name)
        return json.dumps({"category": category_name, "count": len(mems), "members": mems})

    if action == "learn":
        if not members:
            raise ToolError("members required as [[token, confidence], ...]")
        member_tuples = [(m[0], float(m[1])) for m in members]
        count = db.learn_category_members(category_name, member_tuples, source)
        return json.dumps({"category": category_name, "new_members": count, "source": source})

    if action == "reset":
        removed = db.reset_category_to_seed(category_name)
        return json.dumps({"category": category_name, "members_removed": removed})


# ── 14. personality ──


@_specialist_tool(annotations=ToolAnnotations(title="Personality", readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def personality(
    action: str = "get",
    trait_name: str | None = None,
    score: float | None = None,
    recompute: bool = False,
    ctx: Context = None,
) -> str:
    """AI personality traits derived from memory patterns.

    ACTIONS:
    - "get": Get current personality profile. Use recompute=True to refresh.
    - "set": Set a trait manually (needs trait_name + score).

    Traits: warmth, depth, energy, attentiveness (0.0-1.0).

    Args:
        action: "get" or "set".
        trait_name: For set: warmth, depth, energy, attentiveness.
        score: For set: 0.0-1.0.
        recompute: For get: re-derive from memory patterns first.
    """
    if action not in ("get", "set"):
        raise ToolError("action must be 'get' or 'set'")

    db = _get_db(ctx)

    if action == "get":
        if recompute:
            profile = db.derive_personality()
        else:
            profile = db.get_personality()
        return json.dumps(profile)

    if not trait_name or score is None:
        raise ToolError("trait_name and score required for set")
    updated = db.set_personality_trait(trait_name, score)
    return json.dumps({"trait": trait_name, "score": score, "updated": updated})


# ── 15. stats ──


@_specialist_tool(annotations=ToolAnnotations(title="Stats", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def stats(
    action: str = "stats",
    namespace: str | None = None,
    max_rids: int = 100,
    ctx: Context = None,
) -> str:
    """Engine statistics, health check, learned weights, privacy/leak audit,
    and skill substrate counts. Read-only — index maintenance moved to
    think(maintenance_op=...) in v0.10.

    ACTIONS:
    - "stats":       Detailed memory statistics (default).
    - "health":      Quick health check with latency.
    - "weights":     Show adapted recall scoring weights.
    - "audit_leak":  v0.8.0 windowed leak-candidate audit — surfaces recent
                     records that may have leaked sensitive content. Use for
                     privacy review.
    - "skill_outcomes": v0.9.0 — total skill outcomes recorded in the durable
                        timeline.

    Args:
        action: One of the actions above.
        namespace: Filter for stats.
        max_rids: For audit_leak — max candidate rids to inspect.
    """
    valid = ("stats", "health", "weights", "audit_leak", "skill_outcomes")
    if action == "maintenance":
        raise ToolError(
            "maintenance moved to think(maintenance_op="
            "'backfill_entities'|'rebuild_vec_index'|'rebuild_graph_index') "
            "in v0.10 — stats is now read-only."
        )
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

    if action == "stats":
        from .skill_security import COUNTERS, config as security_config
        result = db.stats(namespace=namespace)
        result["procedural"] = db.procedural_stats(namespace=namespace)
        # D4 — skill substrate counters + frozen config snapshot. Lets
        # operators query "are we under attack?" without parsing the
        # audit log.
        result["skill_substrate"] = {
            "counters": COUNTERS.snapshot(),
            "config": security_config().snapshot(),
        }
        return json.dumps(result)

    if action == "health":
        t0 = time.time()
        s = db.stats()
        latency_ms = round((time.time() - t0) * 1000, 1)
        return json.dumps({
            "status": "ok", "latency_ms": latency_ms,
            "active_memories": s.get("active_memories", 0),
            "total_entities": s.get("entities", 0),
            "open_conflicts": s.get("open_conflicts", 0),
        })

    if action == "weights":
        weights = db.learned_weights()
        return json.dumps(weights)

    if action == "audit_leak":
        report = db.audit_leak_candidates(max_rids=max_rids)
        return json.dumps(report if isinstance(report, dict) else {"audit": report})

    if action == "skill_outcomes":
        return json.dumps({"skill_outcomes_total": db.skill_outcome_count()})


# ── 16. skill ──


def _list_skill_memories(db, *, limit: int = 200):
    """Normalize `list_memories` to a flat list of memory dicts.

    Engine returns `{"memories": [...], "total": N, "offset": K}` for the
    embedded path; HTTP backend may return either shape depending on the
    server endpoint. Defensive enough to handle both.
    """
    raw = db.list_memories(limit=limit, namespace=SKILL_NAMESPACE_CONST) or {}
    if isinstance(raw, dict):
        return list(raw.get("memories") or [])
    return list(raw)


def _find_skill_rid_by_id(db, skill_id: str) -> str | None:
    for h in _list_skill_memories(db, limit=500):
        meta = (h.get("metadata") if isinstance(h, dict) else getattr(h, "metadata", None)) or {}
        if meta.get("skill_id") == skill_id:
            return h.get("rid") if isinstance(h, dict) else getattr(h, "rid", None)
    return None


# Module-level constant so the helpers above can reference it without
# importing inside the hot path of every skill() invocation.
SKILL_NAMESPACE_CONST = "skill_substrate"


def _skill_writes_enabled() -> bool:
    """Thin wrapper over skill_security.gate_open() that preserves the
    bool-returning shape for tests. Use `skill_security.gate_open()`
    directly when you need the reason-if-closed string."""
    from .skill_security import gate_open

    is_open, _reason = gate_open()
    return is_open


@_specialist_tool(annotations=ToolAnnotations(title="Skill", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
def skill(
    action: str,
    skill_id: str | None = None,
    body: str | None = None,
    skill_type: str = "procedure",
    applies_to: list[str] | None = None,
    triggers: list[str] | None = None,
    on_conflict: str = "reject",
    version: str | None = None,
    supersedes: str | None = None,
    query: str | None = None,
    top_k: int = 5,
    succeeded: bool | None = None,
    note: str | None = None,
    limit: int = 20,
    ctx: Context = None,
) -> str:
    """Substrate-native agent skill catalog — define, surface, record outcomes.

    Skills are structured catalog entries (`skill_id`, `applies_to`, `body`,
    `type`) — different from loose how-to memories (use `procedure` for those).
    Writes go to the `skill_substrate` namespace so every yantrikdb consumer
    (this MCP, yantrikdb-hermes-plugin, Lane B SDK, WisePick) sees the same
    catalog.

    Schema-validated at write time:
    - skill_id: lowercase dot-separated segments, e.g. "workflow.git.commit_clean"
    - body: 50–5000 chars
    - applies_to: 1–10 lowercase_underscore identifiers (no hyphens)
    - skill_type: one of procedure | reference | lesson | pattern | rule

    ACTIONS:
    - "define":  Create a skill (needs skill_id, body, skill_type, applies_to).
    - "surface": Find relevant skills (needs query). Returns ranked by score.
    - "outcome": Append a use outcome (needs skill_id, succeeded).
    - "get":     Fetch a single skill by id.
    - "list":    Catalog browse (filter by applies_to / skill_type).

    EXAMPLE: skill(action="define", skill_id="workflow.git.commit_clean",
    body="Before commit: run pytest + lint...", skill_type="procedure",
    applies_to=["git", "release"]) — then surface(query=...) before similar
    work, and outcome(skill_id=..., succeeded=True/False) after using one.

    Args:
        action: "define", "surface", "outcome", "get", "list".
        skill_id: Dot-separated id (for define/get/outcome).
        body: Skill body, 50–5000 chars (for define).
        skill_type: procedure|reference|lesson|pattern|rule (for define).
        applies_to: Non-empty identifier list ≤10 entries (for define;
            optional filter for surface/list).
        triggers: Optional list of trigger phrases (for define).
        on_conflict: "reject" (default) or "replace" if skill_id exists.
        version: Optional semver-shaped version string.
        supersedes: Optional skill_id this one replaces.
        query: Natural-language search (for surface).
        top_k: Max results for surface.
        succeeded: Outcome boolean (for outcome).
        note: Optional outcome note.
        limit: Max results for list.
    """
    from .skill_content_scanner import scan_body, scanner_report
    from .skill_security import (
        PENDING_NAMESPACE,
        COUNTERS,
        audit_event,
        author_attribution,
        body_sha256,
        check_cross_origin_replace,
        check_namespace_allowed,
        check_rate_limit,
        check_supersedes_integrity,
        config as security_config,
        gate_open,
        should_route_to_review,
        verify_body_hash,
    )
    from .skill_validation import (
        OUTCOME_NAMESPACE,
        SKILL_NAMESPACE,
        SKILL_TYPES,
        validate_skill_define_args,
        validate_skill_id,
    )

    if action not in ("define", "surface", "outcome", "get", "list"):
        raise ToolError(
            "action must be 'define', 'surface', 'outcome', 'get', or 'list'"
        )

    # Resolve a session_id for rate limiting + audit attribution. The
    # MCP client doesn't expose its own identity, but we can take the
    # request_id off the lifespan context if available, falling back
    # to the OS user as a last resort.
    session_id = None
    if ctx is not None:
        try:
            session_id = getattr(ctx.request_context, "request_id", None)
        except Exception:
            session_id = None
    session_id = session_id or "default"

    # ── Gate (C1 + C2 + base) ──
    # v0.8.1 (issue #8): define/replace use the strict YANTRIKDB_SKILLS_WRITE_ENABLED
    # gate; outcome uses the lighter YANTRIKDB_OUTCOMES_WRITE_ENABLED (default
    # true). `gate_open(action)` dispatches.
    if action in ("define", "outcome"):
        is_open, reason = gate_open(action)
        if not is_open:
            COUNTERS.reject_define(reason="gate_closed") if action == "define" else COUNTERS.reject_outcome("gate_closed")
            audit_event({
                "event": "reject",
                "action": action,
                "reason": "gate_closed",
                "detail": reason,
                "skill_id": skill_id,
                "session_id": session_id,
            })
            raise ToolError(
                f"action='{action}' refused by skill-writes gate. {reason}"
            )

        # ── D2 rate limit ──
        try:
            check_rate_limit(session_id)
        except ValueError as e:
            COUNTERS.reject_define("rate_limit") if action == "define" else COUNTERS.reject_outcome("rate_limit")
            audit_event({
                "event": "reject",
                "action": action,
                "reason": "rate_limit",
                "detail": str(e),
                "skill_id": skill_id,
                "session_id": session_id,
            })
            raise ToolError(str(e)) from e

    db = _get_db(ctx)

    # ── define ──
    if action == "define":
        if applies_to is None:
            applies_to = []

        # Order matters: schema first (cheap, deterministic), then B1
        # namespace, then content scanning (A1-A5), then B4 supersedes,
        # then existence/conflict (which requires DB I/O).
        try:
            validate_skill_define_args(skill_id or "", body or "", skill_type, applies_to)
        except ValueError as e:
            COUNTERS.reject_define("schema")
            audit_event({"event": "reject", "action": "define", "reason": "schema",
                         "detail": str(e), "skill_id": skill_id, "session_id": session_id})
            raise ToolError(str(e)) from e

        if on_conflict not in ("reject", "replace"):
            raise ToolError("on_conflict must be 'reject' or 'replace'")

        # B1 — namespace allowlist
        try:
            check_namespace_allowed(skill_id or "")
        except ValueError as e:
            COUNTERS.reject_define("namespace_not_allowed")
            audit_event({"event": "reject", "action": "define", "reason": "namespace_not_allowed",
                         "detail": str(e), "skill_id": skill_id, "session_id": session_id})
            raise ToolError(str(e)) from e

        # A1-A5 — content scanning. Capture the full report for audit
        # before raising, so we can record which scanners flagged.
        try:
            scan_body(body or "")
        except ValueError as e:
            report = scanner_report(body or "")
            flagged = [k for k, v in report.items() if v]
            COUNTERS.reject_define(f"content_scan:{flagged[0] if flagged else 'unknown'}")
            audit_event({
                "event": "reject", "action": "define", "reason": "content_scan",
                "scanners": flagged, "detail": str(e),
                "skill_id": skill_id, "session_id": session_id,
            })
            raise ToolError(str(e)) from e

        # Find any existing skill with this id (for both conflict resolution
        # AND for B3 cross-origin checks AND for B4 supersedure lookup).
        existing_meta: dict | None = None
        existing_rid: str | None = None
        try:
            for h in _list_skill_memories(db, limit=500):
                m = (h.get("metadata") if isinstance(h, dict) else getattr(h, "metadata", None)) or {}
                if m.get("skill_id") == skill_id:
                    existing_meta = m
                    existing_rid = h.get("rid") if isinstance(h, dict) else getattr(h, "rid", None)
                    break
        except Exception:
            existing_meta = None
            existing_rid = None

        # B4 — supersedes integrity (look up the superseded skill's metadata)
        superseded_meta: dict | None = None
        if supersedes:
            for h in _list_skill_memories(db, limit=500):
                m = (h.get("metadata") if isinstance(h, dict) else getattr(h, "metadata", None)) or {}
                if m.get("skill_id") == supersedes:
                    superseded_meta = m
                    break
            try:
                check_supersedes_integrity(supersedes, skill_id or "", superseded_meta)
            except ValueError as e:
                COUNTERS.reject_define("supersedes_integrity")
                audit_event({"event": "reject", "action": "define", "reason": "supersedes_integrity",
                             "detail": str(e), "skill_id": skill_id, "supersedes": supersedes,
                             "session_id": session_id})
                raise ToolError(str(e)) from e

        # on_conflict handling + B3 cross-origin check
        if existing_rid is not None:
            if on_conflict == "reject":
                COUNTERS.reject_define("on_conflict_reject")
                audit_event({"event": "reject", "action": "define", "reason": "on_conflict_reject",
                             "skill_id": skill_id, "session_id": session_id})
                raise ToolError(
                    f"skill {skill_id!r} already exists (on_conflict='reject'). "
                    "Use on_conflict='replace' or pick a new skill_id."
                )
            # replace: enforce B3 first
            try:
                check_cross_origin_replace(existing_meta, security_config().author_origin)
            except ValueError as e:
                COUNTERS.reject_define("cross_origin_replace")
                audit_event({"event": "reject", "action": "define", "reason": "cross_origin_replace",
                             "detail": str(e), "skill_id": skill_id, "session_id": session_id})
                raise ToolError(str(e)) from e
            try:
                db.forget(existing_rid)
            except Exception as e:
                raise ToolError(f"on_conflict='replace' failed to tombstone {existing_rid}: {e}") from e

        # G — review queue routing
        target_namespace = SKILL_NAMESPACE
        routed_to_review = False
        if should_route_to_review(skill_type, supersedes):
            target_namespace = PENDING_NAMESPACE
            routed_to_review = True

        clean_body = (body or "").strip()

        # B2 + E1 + E2 — author attribution, content hash, origin stamp
        attribution = author_attribution(session_id=session_id)
        metadata: dict = {
            "record_type": "skill",
            "skill_id": skill_id,
            "skill_type": skill_type,
            "applies_to": list(applies_to),
            "source": "mcp",
            "body_sha256": body_sha256(clean_body),
            **attribution,  # session_id, os_user, hostname, wall_clock, author_origin, audit_nonce
        }
        if triggers:
            metadata["triggers"] = list(triggers)
        if version:
            metadata["version"] = version
        if supersedes:
            metadata["supersedes_skill_id"] = supersedes
        if routed_to_review:
            metadata["pending_review"] = True

        rid = db.record(
            clean_body,
            memory_type="procedural",
            importance=0.7,
            valence=0.0,
            metadata=metadata,
            namespace=target_namespace,
            certainty=0.9,
            domain="skill",
            source="user",
            emotional_state=None,
        )

        if routed_to_review:
            COUNTERS.queue_for_review()
        else:
            COUNTERS.accept_define()

        audit_event({
            "event": "accept" if not routed_to_review else "queued_for_review",
            "action": "define",
            "skill_id": skill_id,
            "skill_type": skill_type,
            "applies_to": list(applies_to),
            "namespace": target_namespace,
            "body_sha256": metadata["body_sha256"],
            "body_length": len(clean_body),
            "audit_nonce": attribution["audit_nonce"],
            "session_id": session_id,
            "supersedes": supersedes,
            "replaced": existing_rid is not None,
            "config_snapshot": security_config().snapshot() if security_config().audit_log_path else None,
        })

        return json.dumps({
            "rid": rid, "skill_id": skill_id, "stored": True,
            "replaced": existing_rid is not None,
            "pending_review": routed_to_review,
            "namespace": target_namespace,
        })

    # ── surface ──
    if action == "surface":
        if not query or not query.strip():
            raise ToolError("query required for action='surface'")
        if applies_to is not None and not isinstance(applies_to, list):
            raise ToolError("applies_to must be a list")
        if skill_type and skill_type != "procedure" and skill_type not in SKILL_TYPES:
            raise ToolError(f"skill_type {skill_type!r} not in {sorted(SKILL_TYPES)}")

        # Reads only see the LIVE catalog, never the pending-review queue.
        # The pending queue is operator-only (use `skill(action="list",
        # namespace_hint="pending")` via a future tool extension or
        # query the DB directly).
        response = db.recall_with_response(
            query=query.strip(), top_k=top_k, namespace=SKILL_NAMESPACE,
        )
        items = []
        tampered_skipped = 0
        for r in (response.get("results") or []):
            meta = r.get("metadata") or {}
            if meta.get("record_type") != "skill":
                continue
            # E1 — verify body hash. If stored hash and current body
            # disagree, the body was tampered with out-of-band. Skip
            # the result and log it; do not surface possibly-poisoned
            # content to the caller.
            if not verify_body_hash(r.get("text", ""), meta.get("body_sha256")):
                tampered_skipped += 1
                audit_event({
                    "event": "tamper_detected",
                    "action": "surface",
                    "skill_id": meta.get("skill_id"),
                    "rid": r.get("rid"),
                    "session_id": session_id,
                })
                continue
            if applies_to:
                skill_applies = set(meta.get("applies_to") or [])
                if not (set(applies_to) & skill_applies):
                    continue
            if skill_type and skill_type != "procedure":
                if meta.get("skill_type") != skill_type:
                    continue
            items.append({
                "rid": r["rid"],
                "skill_id": meta.get("skill_id"),
                "skill_type": meta.get("skill_type"),
                "applies_to": meta.get("applies_to", []),
                "triggers": meta.get("triggers", []),
                "version": meta.get("version"),
                "author_origin": meta.get("author_origin"),
                "body": r["text"],
                "score": round(r.get("score", 0.0), 4),
            })
        out: dict = {
            "count": len(items),
            "results": items,
            "confidence": round(response.get("confidence", 0.0), 4),
        }
        if tampered_skipped:
            out["tampered_skipped"] = tampered_skipped
        return json.dumps(out)

    # ── outcome ──
    if action == "outcome":
        if not skill_id:
            raise ToolError("skill_id required for action='outcome'")
        try:
            validate_skill_id(skill_id)
        except ValueError as e:
            COUNTERS.reject_outcome("schema")
            audit_event({"event": "reject", "action": "outcome", "reason": "schema",
                         "detail": str(e), "skill_id": skill_id, "session_id": session_id})
            raise ToolError(str(e)) from e
        if succeeded is None:
            raise ToolError("succeeded (bool) required for action='outcome'")

        # D3 — outcome.note is bounded + scanned (same A1/A2/A4 risk as
        # body content; A3 URLs would be too restrictive on notes so
        # we keep the env-var gate; A5 is skipped for short notes).
        if note is not None:
            if not isinstance(note, str):
                raise ToolError("note must be a string")
            if len(note) > 500:
                COUNTERS.reject_outcome("note_too_long")
                audit_event({"event": "reject", "action": "outcome", "reason": "note_too_long",
                             "skill_id": skill_id, "note_length": len(note),
                             "session_id": session_id})
                raise ToolError(f"note length {len(note)} exceeds 500-char cap [D3]")
            try:
                from .skill_content_scanner import scan_body as _scan
                _scan(note, is_outcome_note=True)
            except ValueError as e:
                COUNTERS.reject_outcome("note_content_scan")
                audit_event({"event": "reject", "action": "outcome", "reason": "note_content_scan",
                             "detail": str(e), "skill_id": skill_id, "session_id": session_id})
                raise ToolError(str(e)) from e

        parts = [f"outcome: skill={skill_id} succeeded={bool(succeeded)}"]
        if note:
            parts.append(f"note: {note}")
        outcome_body = "\n".join(parts)

        attribution = author_attribution(session_id=session_id)
        meta_out: dict = {
            "record_type": "skill_outcome",
            "skill_id": skill_id,
            "succeeded": bool(succeeded),
            "source": "mcp",
            **attribution,
        }
        if note:
            meta_out["note"] = note

        rid = db.record(
            outcome_body,
            memory_type="episodic",
            importance=0.5,
            valence=0.0,
            metadata=meta_out,
            namespace=OUTCOME_NAMESPACE,
            certainty=1.0,
            domain="skill_outcome",
            source="system",
            emotional_state=None,
        )

        COUNTERS.record_outcome()
        audit_event({
            "event": "accept", "action": "outcome",
            "skill_id": skill_id, "succeeded": bool(succeeded),
            "note_present": note is not None, "rid": rid,
            "audit_nonce": attribution["audit_nonce"],
            "session_id": session_id,
        })
        return json.dumps({"rid": rid, "skill_id": skill_id, "recorded": True})

    # ── get ──
    if action == "get":
        if not skill_id:
            raise ToolError("skill_id required for action='get'")
        try:
            validate_skill_id(skill_id)
        except ValueError as e:
            raise ToolError(str(e)) from e
        for h in _list_skill_memories(db, limit=500):
            meta = (h.get("metadata") if isinstance(h, dict) else getattr(h, "metadata", None)) or {}
            if meta.get("skill_id") != skill_id:
                continue
            body_text = h.get("text") if isinstance(h, dict) else getattr(h, "text", None)
            # E1 — tamper check on direct lookup too.
            if not verify_body_hash(body_text or "", meta.get("body_sha256")):
                audit_event({
                    "event": "tamper_detected", "action": "get",
                    "skill_id": skill_id,
                    "rid": h.get("rid") if isinstance(h, dict) else getattr(h, "rid", None),
                    "session_id": session_id,
                })
                return _err(
                    f"skill {skill_id!r} body hash mismatch — possible tampering. "
                    "Refusing to surface. See audit log for details.",
                    skill_id=skill_id,
                )
            return json.dumps({
                "rid": h.get("rid") if isinstance(h, dict) else getattr(h, "rid", None),
                "skill_id": skill_id,
                "skill_type": meta.get("skill_type"),
                "applies_to": meta.get("applies_to", []),
                "triggers": meta.get("triggers", []),
                "version": meta.get("version"),
                "author_origin": meta.get("author_origin"),
                "wall_clock_at_define": meta.get("wall_clock_at_define"),
                "body": body_text,
            })
        return _err(f"skill {skill_id!r} not found", skill_id=skill_id)

    # ── list ──
    if action == "list":
        rows = _list_skill_memories(db, limit=max(1, min(500, limit * 5)))
        items = []
        for h in rows:
            meta = (h.get("metadata") if isinstance(h, dict) else getattr(h, "metadata", None)) or {}
            if meta.get("record_type") != "skill":
                continue
            if applies_to:
                skill_applies = set(meta.get("applies_to") or [])
                if not (set(applies_to) & skill_applies):
                    continue
            if skill_type and skill_type != "procedure":
                if meta.get("skill_type") != skill_type:
                    continue
            items.append({
                "rid": h.get("rid") if isinstance(h, dict) else getattr(h, "rid", None),
                "skill_id": meta.get("skill_id"),
                "skill_type": meta.get("skill_type"),
                "applies_to": meta.get("applies_to", []),
                "version": meta.get("version"),
                "author_origin": meta.get("author_origin"),
            })
            if len(items) >= limit:
                break
        return json.dumps({"count": len(items), "results": items})

    raise ToolError(f"unreachable: action={action!r}")



# ── 17. gaps ──


@_specialist_tool(annotations=ToolAnnotations(title="Gaps", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False))
def gaps(
    min_count: int = 3,
    max_avg_top_score: float = 0.4,
    limit: int = 20,
    ctx: Context = None,
) -> str:
    """Surface knowledge gaps — frequently-asked, poorly-answered queries
    (v0.9.0 engine demand log).

    The substrate logs every recall and tracks how often each query is asked
    + what top scores it surfaces. `knowledge_gaps()` returns the queries
    that are asked often but answered poorly — the substrate's "known
    unknowns". Use this to drive proactive learning: when the agent sees a
    gap, it can ask the user, fetch info, or note the limitation.

    Args:
        min_count: Only surface queries asked at least this many times.
        max_avg_top_score: Only surface queries whose best recall score
            averages below this (lower = poorer answer).
        limit: Max gaps to return.
    """
    db = _get_db(ctx)
    rows = db.knowledge_gaps(min_count=min_count, max_avg_top_score=max_avg_top_score, limit=limit)
    return json.dumps({"count": len(rows or []), "gaps": rows or []})


# ── 18. conversation ──


@_specialist_tool(annotations=ToolAnnotations(title="Conversation", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
def conversation(
    action: str,
    namespace: str = "default",
    role: str | None = None,
    content: str | None = None,
    max_turns: int = 10,
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """Bounded encrypted working-memory ring buffer for raw conversation
    turns (v0.9.0 engine conversation primitive).

    Unlike `remember` (which stores extracted semantic memories), this stores
    verbatim turns — useful for short-horizon working memory, e.g. "what
    exactly did the user say two messages ago". The ring is bounded per
    namespace; oldest turns evict when `max_turns` is exceeded.

    ACTIONS:
    - "record":  Append a turn (needs role + content).
    - "recent":  Retrieve last N turns, oldest-first.
    - "clear":   Drop the buffer for a namespace.

    Args:
        action: "record" | "recent" | "clear".
        namespace: Ring buffer namespace (separate buffers per agent / topic).
        role: "user" | "assistant" | "system" | "tool" — caller's choice.
        content: The verbatim turn text.
        max_turns: Ring size at record time (default 10).
        limit: How many recent turns to return.
    """
    if action not in ("record", "recent", "clear"):
        raise ToolError("action must be 'record', 'recent', or 'clear'")
    db = _get_db(ctx)

    if action == "record":
        if not role or not content:
            raise ToolError("role and content required for action='record'")
        db.record_turn(namespace, role, content, max_turns=max_turns)
        return json.dumps({"recorded": True, "namespace": namespace, "role": role})

    if action == "recent":
        turns = db.recent_turns(namespace, limit=limit) or []
        return json.dumps({"count": len(turns), "namespace": namespace, "turns": turns})

    if action == "clear":
        removed = db.clear_turns(namespace)
        return json.dumps({"namespace": namespace, "removed": removed})

    raise ToolError(f"unreachable: action={action!r}")


# ── 19. task ──


@_specialist_tool(annotations=ToolAnnotations(title="Task", readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False))
def task(
    action: str,
    namespace: str = "default",
    title: str | None = None,
    priority: str = "medium",
    parent_id: str | None = None,
    task_id: str | None = None,
    status: str | None = None,
    ctx: Context = None,
) -> str:
    """Substrate-backed task / chore store (v0.9.0 engine).

    A thin general-purpose to-do tracker baked into yantrikdb — survives
    sessions, lives next to memories so future agents see open tasks at
    session_digest time.

    ACTIONS:
    - "add":    Create a task (needs title; optional priority + parent_id).
    - "get":    Fetch one task by id.
    - "list":   List tasks in a namespace, optionally filtered by status.
    - "update": Update status and/or priority (needs task_id).
    - "delete": Delete a task (needs task_id).

    PRIORITY: "low" | "medium" | "high" — priority-ordered in `list`.
    STATUS:   typically "open" | "doing" | "done" | "blocked".

    Args:
        action: "add" | "get" | "list" | "update" | "delete".
        namespace: Per-project / per-agent isolation.
        title: Task description (for add).
        priority: "low" | "medium" | "high" (for add / update).
        parent_id: Optional parent task id (for add — sub-task tree).
        task_id: Task id (for get / update / delete).
        status: Filter (for list) or new value (for update).
    """
    if action not in ("add", "get", "list", "update", "delete"):
        raise ToolError("action must be 'add', 'get', 'list', 'update', or 'delete'")
    db = _get_db(ctx)

    if action == "add":
        if not title or not title.strip():
            raise ToolError("title required for action='add'")
        tid = db.task_add(namespace, title.strip(), priority=priority, parent_id=parent_id)
        return json.dumps({"task_id": tid, "title": title.strip(), "priority": priority})

    if action == "get":
        if not task_id:
            raise ToolError("task_id required for action='get'")
        row = db.task_get(task_id)
        if row is None:
            return _err(f"task {task_id!r} not found", task_id=task_id)
        return json.dumps(row)

    if action == "list":
        rows = db.task_list(namespace, status=status) or []
        return json.dumps({"count": len(rows), "namespace": namespace, "tasks": rows})

    if action == "update":
        if not task_id:
            raise ToolError("task_id required for action='update'")
        if status is None and priority is None:
            raise ToolError("at least one of status / priority required for action='update'")
        existed = db.task_update(task_id, status=status, priority=priority)
        return json.dumps({"task_id": task_id, "updated": existed})

    if action == "delete":
        if not task_id:
            raise ToolError("task_id required for action='delete'")
        removed = db.task_delete(task_id)
        return json.dumps({"task_id": task_id, "removed": removed})

    raise ToolError(f"unreachable: action={action!r}")


# ── 20. admin (operator-gated, cluster mode) ──
# Registers ONLY when an operator sets YANTRIKDB_ENABLE_ADMIN_TOOLS=1 —
# absent beats advertised-but-degraded (agreement #6), and cluster
# maintenance behind a master token is not an agent-facing default.

if os.environ.get("YANTRIKDB_ENABLE_ADMIN_TOOLS", "").strip().lower() in ("1", "true", "yes"):

    @mcp.tool(annotations=ToolAnnotations(title="Admin", readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False))
    def admin(
        action: str,
        tenant: str | None = None,
        split_oversized: bool = False,
        repair_artifacts: bool = False,
        ctx: Context = None,
    ) -> str:
        """Cluster maintenance operations (operator-gated; master token).

        ACTIONS:
        - "maintenance_run":    Run a maintenance cycle server-side.
                                tenant=<db> for one DB, omit for all.
        - "maintenance_status": Worker/tenant maintenance status.

        Cluster (YANTRIKDB_SERVER_URL) only — in embedded mode use
        think(maintenance_cycle=True) instead; the engine is in-process.

        Args:
            action: "maintenance_run" or "maintenance_status".
            tenant: Optional single-tenant scope.
            split_oversized / repair_artifacts: Heavy passes (run only).
        """
        if action not in ("maintenance_run", "maintenance_status"):
            raise ToolError("action must be 'maintenance_run' or 'maintenance_status'")
        if not os.environ.get("YANTRIKDB_SERVER_URL", "").strip():
            raise ToolError(
                "admin is cluster-only. In embedded mode the engine is "
                "in-process — use think(maintenance_cycle=True)."
            )
        db = _get_db(ctx)
        if action == "maintenance_run":
            result = db.maintenance_run(
                tenant=tenant, split_oversized=split_oversized,
                repair_artifacts=repair_artifacts,
            )
        else:
            result = db.maintenance_status(tenant=tenant)
        return json.dumps(result if isinstance(result, dict) else {"result": result})
