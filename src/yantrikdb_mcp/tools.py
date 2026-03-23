"""MCP tool implementations for YantrikDB cognitive memory engine."""

import json
import time

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.exceptions import ToolError

from .server import mcp


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


# ── 1. remember ──


@mcp.tool()
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
    ctx: Context = None,
) -> str:
    """Store one or more memories in persistent cognitive memory.

    WHEN TO USE: Call proactively whenever the conversation reveals something
    worth remembering — decisions, preferences, facts about people, project context.
    Do NOT store ephemeral task details, code snippets, or git-derivable info.

    SINGLE: remember(text="User prefers dark mode", domain="preference", importance=0.7)
    BATCH:  remember(memories=[{"text": "Alice is DevOps lead", "domain": "people"}, {"text": "Deadline March 30", "domain": "work", "importance": 0.9}])

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
        memories: List of memory dicts for batch. Each needs "text", optional: memory_type, importance, domain, source, valence, metadata, certainty, emotional_state.
    """
    db = _get_db(ctx)

    # Batch mode
    if memories:
        # Validate all items before committing any
        valid_types = ("semantic", "episodic", "procedural")
        for i, mem in enumerate(memories):
            t = (mem.get("text") or "").strip()
            if not t:
                raise ToolError(f"memories[{i}].text must be non-empty")
            mt = mem.get("memory_type", "semantic")
            if mt not in valid_types:
                raise ToolError(f"memories[{i}].memory_type must be one of {valid_types}, got '{mt}'")
        results = []
        for mem in memories:
            rid = db.record(
                mem["text"].strip(),
                memory_type=mem.get("memory_type", "semantic"),
                importance=max(0.0, min(1.0, mem.get("importance", 0.5))),
                valence=max(-1.0, min(1.0, mem.get("valence", 0.0))),
                metadata=mem.get("metadata", {}),
                namespace=namespace,
                certainty=max(0.0, min(1.0, mem.get("certainty", 0.8))),
                domain=mem.get("domain", "general"),
                source=mem.get("source", "user"),
                emotional_state=mem.get("emotional_state"),
            )
            results.append(rid)
        return json.dumps({"rids": results, "count": len(results), "status": "recorded"})

    # Single mode
    if not text or not text.strip():
        raise ToolError("text must be non-empty")
    text = text.strip()

    importance = max(0.0, min(1.0, importance))
    valence = max(-1.0, min(1.0, valence))
    certainty = max(0.0, min(1.0, certainty))

    valid_types = ("semantic", "episodic", "procedural")
    if memory_type not in valid_types:
        raise ToolError(f"memory_type must be one of {valid_types}, got '{memory_type}'")

    rid = db.record(
        text, memory_type=memory_type, importance=importance, valence=valence,
        metadata=metadata or {}, namespace=namespace, certainty=certainty,
        domain=domain, source=source, emotional_state=emotional_state,
    )
    return json.dumps({"rid": rid, "status": "recorded"})


# ── 2. recall ──


@mcp.tool()
def recall(
    query: str,
    top_k: int = 10,
    memory_type: str | None = None,
    domain: str | None = None,
    source: str | None = None,
    namespace: str | None = None,
    include_consolidated: bool = False,
    expand_entities: bool = True,
    refine_from: str | None = None,
    refine_exclude: list[str] | None = None,
    feedback_rid: str | None = None,
    feedback: str | None = None,
    feedback_score: float | None = None,
    feedback_rank: int | None = None,
    ctx: Context = None,
) -> str:
    """Search memories by semantic similarity, refine low-confidence results, or give feedback.

    MODES:
    - **Search** (default): recall("project architecture decisions")
    - **Refine**: recall("PostgreSQL vs MySQL decision", refine_from="database choice", refine_exclude=["rid1"])
    - **Feedback**: recall(query="", feedback_rid="abc", feedback="relevant")

    WHEN TO USE:
    - At conversation start: recall a summary of the user's first message.
    - When user references past decisions, people, preferences, or "last time".
    - When unsure about something the user assumes you know.
    - Use refine_from when first recall had low confidence (< 0.5).
    - Use feedback_rid after using a recalled memory to improve future retrieval.

    Args:
        query: Natural language search query. Be descriptive.
        top_k: Max results (default 10). 3-5 for focused, 10-20 for broad.
        memory_type: Filter: "semantic", "episodic", "procedural".
        domain: Filter: "work", "preference", "architecture", "people", etc.
        source: Filter: "user", "inference", "document", "system".
        namespace: Filter by namespace.
        include_consolidated: Include merged memories.
        expand_entities: Use knowledge graph boosting (default True).
        refine_from: Original query text to refine from. query becomes the refinement.
        refine_exclude: Memory IDs to exclude when refining.
        feedback_rid: Memory ID to give feedback on (switches to feedback mode).
        feedback: "relevant" or "irrelevant" (required with feedback_rid).
        feedback_score: Score at retrieval (helps learning).
        feedback_rank: Rank position at retrieval.
    """
    db = _get_db(ctx)

    # Feedback mode
    if feedback_rid:
        if feedback not in ("relevant", "irrelevant"):
            raise ToolError("feedback must be 'relevant' or 'irrelevant'")
        db.recall_feedback(
            rid=feedback_rid, feedback=feedback,
            query_text=query or None, score_at_retrieval=feedback_score,
            rank_at_retrieval=feedback_rank,
        )
        return json.dumps({"rid": feedback_rid, "feedback": feedback, "status": "recorded"})

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
             "score": round(r["score"], 4), "importance": r["importance"], "created_at": r["created_at"]}
            for r in response["results"]
        ]
        hints = [
            {"hint_type": h["hint_type"], "suggestion": h["suggestion"], "related_entities": h["related_entities"]}
            for h in response["hints"]
        ]
        return json.dumps({"count": len(items), "results": items, "confidence": round(response["confidence"], 4), "hints": hints})

    # Search mode (default)
    response = db.recall_with_response(
        query=query, top_k=top_k, memory_type=memory_type,
        include_consolidated=include_consolidated, expand_entities=expand_entities,
        namespace=namespace, domain=domain, source=source,
    )
    items = [
        {"rid": r["rid"], "text": r["text"], "type": r["type"],
         "score": round(r["score"], 4), "importance": r["importance"],
         "why_retrieved": r["why_retrieved"]}
        for r in response["results"]
    ]
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


@mcp.tool()
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


@mcp.tool()
def correct(
    rid: str,
    new_text: str,
    new_importance: float | None = None,
    new_valence: float | None = None,
    correction_note: str | None = None,
    ctx: Context = None,
) -> str:
    """Correct an existing memory — tombstones the old one and creates a corrected version.

    WHEN TO USE: When the user corrects a recalled fact.
    - "Actually, we're using Python 3.12, not 3.11" → correct the memory.
    Preserves history and transfers entity relationships to the new memory.

    Args:
        rid: The memory ID to correct.
        new_text: The corrected text content.
        new_importance: Optional updated importance (0.0-1.0).
        new_valence: Optional updated valence (-1.0 to 1.0).
        correction_note: Why the correction was made.
    """
    if not new_text or not new_text.strip():
        raise ToolError("new_text must be non-empty")

    db = _get_db(ctx)
    try:
        result = db.correct(rid, new_text, new_importance=new_importance,
                            new_valence=new_valence, correction_note=correction_note)
    except Exception as e:
        return _err(str(e), rid=rid)
    return json.dumps({
        "original_rid": result["original_rid"],
        "corrected_rid": result["corrected_rid"],
        "original_tombstoned": result["original_tombstoned"],
    })


# ── 5. think ──


@mcp.tool()
def think(
    run_consolidation: bool = True,
    run_conflict_scan: bool = True,
    run_pattern_mining: bool = True,
    ctx: Context = None,
) -> str:
    """Run cognitive maintenance — consolidate, detect conflicts, mine patterns.

    WHEN TO USE:
    - End of long conversations with many new memories.
    - When you suspect contradictions exist.
    - Periodically to keep memory healthy.

    Returns consolidation count, conflicts found, patterns, and triggers.
    """
    db = _get_db(ctx)
    config = {
        "run_consolidation": run_consolidation,
        "run_conflict_scan": run_conflict_scan,
        "run_pattern_mining": run_pattern_mining,
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


@mcp.tool()
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
    ctx: Context = None,
) -> str:
    """Manage individual memories — get, list, search, update importance, archive, or hydrate.

    ACTIONS:
    - "get": Retrieve a single memory by rid.
    - "list": Browse memories with filters (domain, type, namespace, sort).
    - "search": Keyword search — exact substring match (case-insensitive). Use instead of recall when you need exact keyword lookups like "Postgres version".
    - "update_importance": Change a memory's importance score.
    - "archive": Move memory to cold storage (excluded from recall).
    - "hydrate": Restore archived memory to active.

    Args:
        action: One of "get", "list", "search", "update_importance", "archive", "hydrate".
        rid: Memory ID (required for get/update_importance/archive/hydrate).
        importance: New importance 0.0-1.0 (for update_importance).
        limit: Max results for list/search (default 50, max 200).
        offset: Pagination offset for list.
        domain: Filter for list/search.
        memory_type: Filter for list/search: "semantic", "episodic", "procedural".
        namespace: Filter for list/search.
        sort_by: For list: "created_at", "importance", "last_access".
        text_contains: For search: case-insensitive substring to match.
    """
    valid = ("get", "list", "search", "update_importance", "archive", "hydrate")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

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
        result = db.correct(rid, mem["text"], new_importance=max(0.0, min(1.0, importance)),
                            correction_note=f"Importance adjusted from {mem['importance']} to {importance}")
        return json.dumps({"rid": result["corrected_rid"], "old_importance": mem["importance"],
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


# ── 7. graph ──


@mcp.tool()
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
    ctx: Context = None,
) -> str:
    """Knowledge graph operations — create relationships, query entities, link memories.

    ACTIONS:
    - "relate": Create relationship. Needs entity (source), target, relationship.
    - "edges": Get all relationships for entity.
    - "link": Link a memory (rid) to an entity.
    - "search": Find entities by pattern (substring match).
    - "profile": Rich entity profile with temporal/emotional dimensions.
    - "depth": Measure how deeply the system knows an entity (0.0-1.0 score).

    EXAMPLES:
    - graph(action="relate", entity="Alice", target="Backend Team", relationship="manages")
    - graph(action="edges", entity="Alice")
    - graph(action="link", rid="abc123", entity="Alice")
    - graph(action="search", pattern="Ali")
    - graph(action="profile", entity="Alice")
    - graph(action="depth", entity="Alice")

    Args:
        action: "relate", "edges", "link", "search", "profile", "depth".
        entity: Entity name (required for all except link when pattern is used).
        target: Target entity (for relate).
        relationship: Relationship type (for relate). Default "related_to".
        weight: Relationship strength 0.0-1.0 (for relate).
        rid: Memory ID (for link).
        pattern: Search pattern (for search).
        limit: Max results (for search).
        days: Time window (for profile).
        namespace: Namespace filter (for profile/depth).
    """
    valid = ("relate", "edges", "link", "search", "profile", "depth")
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


# ── 8. conflict ──


@mcp.tool()
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
    ctx: Context = None,
) -> str:
    """Manage memory conflicts (contradictions) — list, resolve, dismiss, or reclassify.

    ACTIONS:
    - "list": List conflicts. Optional status filter ("open", "resolved", "dismissed").
    - "get": Get single conflict by conflict_id.
    - "resolve": Resolve with strategy: "keep_a", "keep_b", "keep_both", "merge", "dismiss".
    - "reclassify": Reclassify conflict type and teach substitution patterns.

    EXAMPLES:
    - conflict() → list open conflicts
    - conflict(action="get", conflict_id="abc")
    - conflict(action="resolve", conflict_id="abc", strategy="keep_a")
    - conflict(action="resolve", conflict_id="abc", strategy="dismiss", resolution_note="false positive")
    - conflict(action="reclassify", conflict_id="abc", new_type="preference")

    Args:
        action: "list", "get", "resolve", "reclassify".
        conflict_id: Required for get/resolve/reclassify.
        status: Filter for list: "open", "resolved", "dismissed".
        strategy: For resolve: "keep_a", "keep_b", "keep_both", "merge", "dismiss".
        winner_rid: For keep_a/keep_b (optional, inferred).
        new_text: For "merge" strategy.
        resolution_note: Why this resolution was chosen.
        new_type: For reclassify: "identity_fact", "preference", "temporal".
        limit: Max results for list.
    """
    valid = ("list", "get", "resolve", "reclassify")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

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


@mcp.tool()
def trigger(
    action: str = "pending",
    trigger_id: str | None = None,
    trigger_type: str | None = None,
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """Manage proactive triggers — insights, warnings, and suggestions from the memory system.

    ACTIONS:
    - "pending": Get pending triggers (default).
    - "history": View past triggers.
    - "acknowledge": Mark trigger as seen.
    - "deliver": Mark as shown to user.
    - "act": Mark as acted upon.
    - "dismiss": Dismiss as irrelevant.

    Args:
        action: "pending", "history", "acknowledge", "deliver", "act", "dismiss".
        trigger_id: Required for acknowledge/deliver/act/dismiss.
        trigger_type: Filter by type (for pending/history).
        limit: Max results.
    """
    valid = ("pending", "history", "acknowledge", "deliver", "act", "dismiss")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

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


@mcp.tool()
def session(
    action: str,
    session_id: str | None = None,
    namespace: str = "default",
    client_id: str = "default",
    metadata: dict | None = None,
    summary: str | None = None,
    limit: int = 10,
    abandon_stale_hours: float | None = None,
    ctx: Context = None,
) -> str:
    """Session lifecycle — start, end, history, active check, and stale cleanup.

    ACTIONS:
    - "start": Begin a new session. Returns session_id.
    - "end": End a session (needs session_id). Returns stats.
    - "history": View past sessions.
    - "active": Check if there's a running session.
    - "abandon_stale": Clean up orphaned sessions older than abandon_stale_hours.

    Args:
        action: "start", "end", "history", "active", "abandon_stale".
        session_id: For end.
        namespace: Memory namespace.
        client_id: Client identifier.
        metadata: For start — optional dict.
        summary: For end — what happened.
        limit: For history.
        abandon_stale_hours: For abandon_stale — max age in hours.
    """
    valid = ("start", "end", "history", "active", "abandon_stale")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

    if action == "start":
        sid = db.session_start(namespace, client_id, metadata or {})
        return json.dumps({"session_id": sid})

    if action == "end":
        if not session_id:
            raise ToolError("session_id required")
        result = db.session_end(session_id, summary)
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


# ── 11. temporal ──


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
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


@mcp.tool()
def stats(
    action: str = "stats",
    namespace: str | None = None,
    maintenance_op: str | None = None,
    ctx: Context = None,
) -> str:
    """Engine statistics, health check, learned weights, and maintenance operations.

    ACTIONS:
    - "stats": Detailed memory statistics (default).
    - "health": Quick health check with latency.
    - "weights": Show adapted recall scoring weights.
    - "maintenance": Run maintenance (needs maintenance_op).

    MAINTENANCE OPS:
    - "backfill_entities": Create missing memory↔entity links.
    - "rebuild_vec_index": Rebuild vector similarity index.
    - "rebuild_graph_index": Rebuild knowledge graph index.

    Args:
        action: "stats", "health", "weights", "maintenance".
        namespace: Filter for stats.
        maintenance_op: For maintenance: "backfill_entities", "rebuild_vec_index", "rebuild_graph_index".
    """
    valid = ("stats", "health", "weights", "maintenance")
    if action not in valid:
        raise ToolError(f"action must be one of {valid}")

    db = _get_db(ctx)

    if action == "stats":
        result = db.stats(namespace=namespace)
        result["procedural"] = db.procedural_stats(namespace=namespace)
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

    if action == "maintenance":
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
