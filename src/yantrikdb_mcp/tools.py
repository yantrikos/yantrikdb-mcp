"""MCP tool implementations for YantrikDB cognitive memory engine."""

import json
import time

from mcp.server.fastmcp import Context

from .server import mcp


def _get_db(ctx: Context):
    """Get the YantrikDB instance and lock from the lifespan context."""
    lc = ctx.request_context.lifespan_context
    return lc["db"], lc["lock"]


# ── Core Memory Tools ──


@mcp.tool()
def remember(
    text: str,
    memory_type: str = "semantic",
    importance: float = 0.5,
    domain: str = "general",
    source: str = "user",
    valence: float = 0.0,
    metadata: dict | None = None,
    namespace: str = "default",
    certainty: float = 0.8,
    emotional_state: str | None = None,
    ctx: Context = None,
) -> str:
    """Store a new memory in persistent cognitive memory.

    WHEN TO USE: Call this proactively whenever the conversation reveals something
    worth remembering across sessions — decisions, preferences, facts about people,
    project context, or corrections. Do NOT store ephemeral task details, code
    snippets, or anything derivable from git/files.

    EXAMPLES:
    - User says "I'm switching to Neovim" → remember("User is switching to Neovim as their primary editor", domain="preference", importance=0.7)
    - Decision made: "We'll use PostgreSQL for the new service" → remember("Decision: use PostgreSQL for the new microservice", domain="architecture", importance=0.8)
    - User shares: "Alice is our DevOps lead" → remember("Alice is the DevOps lead on the team", domain="people", importance=0.7)

    IMPORTANCE GUIDE:
    - 0.8-1.0: Critical decisions, strong preferences, key people
    - 0.5-0.7: Useful context, project details, minor preferences
    - 0.3-0.5: Nice-to-know, background information

    Args:
        text: The memory content. Be specific and searchable — "User prefers dark mode in VS Code" not "likes dark".
        memory_type: "semantic" (facts/knowledge), "episodic" (events/experiences), "procedural" (how-to/processes).
        importance: How important (0.0-1.0). Higher = remembered longer.
        domain: Topic area — "work", "preference", "architecture", "people", "infrastructure", "health", "finance", "general".
        source: Who provided this — "user" (user said it), "inference" (you deduced it), "document" (from a file), "system".
        valence: Emotional tone (-1.0 negative to 1.0 positive). 0.0 is neutral.
        metadata: Optional key-value pairs for extra context (e.g. {"project": "acme", "sprint": "23"}).
        namespace: Memory namespace for isolation (default: "default"). Use for per-project memory separation.
        certainty: How confident you are in this memory (0.0-1.0). Lower for inferences.
        emotional_state: Emotion label if relevant — joy, frustration, excitement, concern, neutral.

    Returns the memory ID (rid) of the stored memory.
    """
    # Input validation
    if not text or not text.strip():
        return json.dumps({"error": "text must be non-empty"})
    text = text.strip()

    # Clamp numeric fields to valid ranges
    importance = max(0.0, min(1.0, importance))
    valence = max(-1.0, min(1.0, valence))
    certainty = max(0.0, min(1.0, certainty))

    valid_types = ("semantic", "episodic", "procedural")
    if memory_type not in valid_types:
        return json.dumps({"error": f"memory_type must be one of {valid_types}, got '{memory_type}'"})

    db, lock = _get_db(ctx)
    with lock:
        rid = db.record(
            text,
            memory_type=memory_type,
            importance=importance,
            valence=valence,
            metadata=metadata or {},
            namespace=namespace,
            certainty=certainty,
            domain=domain,
            source=source,
            emotional_state=emotional_state,
        )
    return json.dumps({"rid": rid, "status": "recorded"})


@mcp.tool()
def bulk_remember(
    memories: list[dict],
    namespace: str = "default",
    ctx: Context = None,
) -> str:
    """Store multiple memories at once — efficient for conversation summaries or batch imports.

    WHEN TO USE: At the end of a conversation when you have several things to remember,
    or when processing a document that contains multiple facts. More efficient than
    calling remember() in a loop.

    EXAMPLE:
    bulk_remember(memories=[
        {"text": "User prefers Python 3.12", "domain": "preference", "importance": 0.7},
        {"text": "Project deadline is March 30", "domain": "work", "importance": 0.9},
        {"text": "Alice handles frontend, Bob handles backend", "domain": "people", "importance": 0.6}
    ])

    Args:
        memories: List of memory objects. Each must have "text" and can optionally include:
            memory_type (default "semantic"), importance (default 0.5), domain (default "general"),
            source (default "user"), valence (default 0.0), metadata (default {}),
            certainty (default 0.8), emotional_state (default None).
        namespace: Namespace for all memories in this batch.

    Returns list of memory IDs created.
    """
    db, lock = _get_db(ctx)
    results = []
    with lock:
        for mem in memories:
            rid = db.record(
                mem["text"],
                memory_type=mem.get("memory_type", "semantic"),
                importance=mem.get("importance", 0.5),
                valence=mem.get("valence", 0.0),
                metadata=mem.get("metadata", {}),
                namespace=namespace,
                certainty=mem.get("certainty", 0.8),
                domain=mem.get("domain", "general"),
                source=mem.get("source", "user"),
                emotional_state=mem.get("emotional_state"),
            )
            results.append(rid)
    return json.dumps({"rids": results, "count": len(results), "status": "recorded"})


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
    ctx: Context = None,
) -> str:
    """Search memories by semantic similarity to a natural language query.

    WHEN TO USE:
    - At conversation start: recall a summary of the user's first message to load context.
    - When the user references past decisions, people, preferences, or "last time we...".
    - When you're unsure about something the user assumes you know.
    - When the user asks "do you remember..." or "what did we decide about...".

    EXAMPLES:
    - User asks about a project → recall("project X architecture decisions")
    - User mentions a person → recall("Alice role and responsibilities")
    - User references a preference → recall("user editor preferences and tooling setup")
    - Filter to work context only → recall("deployment process", domain="work")

    TIPS:
    - Use natural language queries, not keywords — "what database did we choose" works better than "database choice".
    - If results have low confidence, try recall_refine with a rephrased query.
    - Check the 'hints' field in results — it suggests follow-up queries.

    Uses multi-signal scoring: vector similarity, temporal decay, recency,
    importance weighting, and knowledge graph expansion.

    Args:
        query: Natural language search query. Be descriptive.
        top_k: Maximum results to return (default 10). Use 3-5 for focused queries, 10-20 for broad exploration.
        memory_type: Filter: "semantic", "episodic", or "procedural". None for all types.
        domain: Filter by domain: "work", "preference", "architecture", "people", etc. None for all.
        source: Filter by source: "user", "inference", "document", "system". None for all.
        namespace: Filter by namespace. None returns all namespaces.
        include_consolidated: Include merged/consolidated memories (default False).
        expand_entities: Use knowledge graph to find related memories (default True). Disable for faster, narrower search.

    Returns memories ranked by relevance, with confidence score and retrieval hints.
    """
    db, lock = _get_db(ctx)
    with lock:
        response = db.recall_with_response(
            query=query,
            top_k=top_k,
            memory_type=memory_type,
            include_consolidated=include_consolidated,
            expand_entities=expand_entities,
            namespace=namespace,
            domain=domain,
            source=source,
        )
    items = []
    for r in response["results"]:
        items.append({
            "rid": r["rid"],
            "text": r["text"],
            "type": r["type"],
            "score": round(r["score"], 4),
            "importance": r["importance"],
            "created_at": r["created_at"],
            "scores": {
                "similarity": round(r["scores"]["similarity"], 4),
                "decay": round(r["scores"]["decay"], 4),
                "recency": round(r["scores"]["recency"], 4),
                "importance": round(r["scores"]["importance"], 4),
                "graph_proximity": round(r["scores"]["graph_proximity"], 4),
            },
            "why_retrieved": r["why_retrieved"],
        })
    hints = [
        {
            "hint_type": h["hint_type"],
            "suggestion": h["suggestion"],
            "related_entities": h["related_entities"],
        }
        for h in response["hints"]
    ]
    return json.dumps({
        "count": len(items),
        "results": items,
        "confidence": round(response["confidence"], 4),
        "retrieval_summary": {
            "top_similarity": round(response["retrieval_summary"]["top_similarity"], 4),
            "score_spread": round(response["retrieval_summary"]["score_spread"], 4),
            "sources_used": response["retrieval_summary"]["sources_used"],
            "candidate_count": response["retrieval_summary"]["candidate_count"],
        },
        "hints": hints,
    })


@mcp.tool()
def recall_refine(
    original_query: str,
    refinement_text: str,
    original_rids: list[str] | None = None,
    top_k: int = 10,
    namespace: str | None = None,
    domain: str | None = None,
    source: str | None = None,
    ctx: Context = None,
) -> str:
    """Refine a previous recall with a follow-up query when results were unsatisfying.

    WHEN TO USE: When recall() returned low-confidence results (< 0.5) or the hints
    suggested rephrasing. Combines the original and refined queries (weighted 0.4/0.6)
    and excludes already-seen results.

    EXAMPLE:
    - First: recall("database choice") → low confidence
    - Then: recall_refine("database choice", "PostgreSQL vs MySQL decision for new service", original_rids=["rid1", "rid2"])

    Args:
        original_query: The original search query text.
        refinement_text: The improved/clarifying query text.
        original_rids: Memory IDs from the first recall to exclude.
        top_k: Maximum results (default 10).
        namespace: Filter by namespace.
        domain: Filter by domain.
        source: Filter by source.

    Returns new results with confidence and hints.
    """
    db, lock = _get_db(ctx)
    with lock:
        original_emb = db.embed(original_query)
        response = db.recall_refine(
            original_query_embedding=original_emb,
            refinement_text=refinement_text,
            original_rids=original_rids or [],
            top_k=top_k,
            namespace=namespace,
            domain=domain,
            source=source,
        )
    items = []
    for r in response["results"]:
        items.append({
            "rid": r["rid"],
            "text": r["text"],
            "type": r["type"],
            "score": round(r["score"], 4),
            "importance": r["importance"],
            "created_at": r["created_at"],
        })
    hints = [
        {
            "hint_type": h["hint_type"],
            "suggestion": h["suggestion"],
            "related_entities": h["related_entities"],
        }
        for h in response["hints"]
    ]
    return json.dumps({
        "count": len(items),
        "results": items,
        "confidence": round(response["confidence"], 4),
        "hints": hints,
    })


@mcp.tool()
def get_memory(rid: str, ctx: Context = None) -> str:
    """Get a specific memory by its ID.

    WHEN TO USE: When you have a memory ID (from recall results, conflict details,
    or trigger source_rids) and need the full record.

    Args:
        rid: The memory ID to retrieve.

    Returns the full memory record with all fields.
    """
    db, lock = _get_db(ctx)
    with lock:
        mem = db.get(rid)
    if mem is None:
        return json.dumps({"error": "Memory not found", "rid": rid})
    return json.dumps({
        "rid": mem["rid"],
        "text": mem["text"],
        "type": mem["type"],
        "importance": mem["importance"],
        "valence": mem["valence"],
        "created_at": mem["created_at"],
        "last_access": mem["last_access"],
        "consolidation_status": mem["consolidation_status"],
        "storage_tier": mem["storage_tier"],
        "metadata": mem["metadata"],
        "certainty": mem["certainty"],
        "domain": mem["domain"],
        "source": mem["source"],
        "emotional_state": mem["emotional_state"],
    })


@mcp.tool()
def forget(rid: str, ctx: Context = None) -> str:
    """Permanently forget (tombstone) a memory.

    WHEN TO USE: When the user explicitly asks to forget something, or when a memory
    is clearly wrong and correction isn't appropriate (e.g., it was about the wrong person entirely).
    Prefer `correct` over `forget` when the memory just needs updating.

    Args:
        rid: The memory ID to forget.

    Returns whether the memory was found and forgotten.
    """
    db, lock = _get_db(ctx)
    with lock:
        forgotten = db.forget(rid)
    return json.dumps({"rid": rid, "forgotten": forgotten})


@mcp.tool()
def bulk_forget(rids: list[str], ctx: Context = None) -> str:
    """Forget multiple memories at once.

    WHEN TO USE: Cleanup after testing, removing a batch of outdated memories,
    or clearing a namespace. More efficient than calling forget() in a loop.

    Args:
        rids: List of memory IDs to forget.

    Returns count of successfully forgotten memories.
    """
    if not rids:
        return json.dumps({"error": "rids list must not be empty"})
    db, lock = _get_db(ctx)
    forgotten = 0
    with lock:
        for rid in rids:
            if db.forget(rid):
                forgotten += 1
    return json.dumps({"forgotten": forgotten, "total": len(rids)})


@mcp.tool()
def list_memories(
    limit: int = 50,
    offset: int = 0,
    domain: str | None = None,
    memory_type: str | None = None,
    namespace: str | None = None,
    sort_by: str = "created_at",
    ctx: Context = None,
) -> str:
    """Browse stored memories without a search query. Useful for auditing what's stored.

    WHEN TO USE: When you need to see what memories exist, audit stored data,
    or browse memories by domain/type without a specific search query.

    Args:
        limit: Maximum memories to return (default 50, max 200).
        offset: Skip first N results for pagination.
        domain: Filter by domain (e.g., "work", "preference", "people").
        memory_type: Filter by type: "semantic", "episodic", "procedural".
        namespace: Filter by namespace.
        sort_by: Sort order — "created_at" (newest first), "importance" (highest first), "last_access" (most recent first).

    Returns a paginated list of memories with their metadata.
    """
    limit = max(1, min(200, limit))
    offset = max(0, offset)

    db, lock = _get_db(ctx)
    with lock:
        result = db.list_memories(
            limit=limit, offset=offset, domain=domain,
            memory_type=memory_type, namespace=namespace, sort_by=sort_by,
        )
    items = [
        {
            "rid": m["rid"], "type": m["type"], "text": m["text"],
            "importance": m["importance"], "valence": m["valence"],
            "domain": m["domain"], "source": m["source"],
            "created_at": m["created_at"], "last_access": m["last_access"],
            "namespace": m["namespace"],
        }
        for m in result["memories"]
    ]
    return json.dumps({
        "count": len(items), "total": result["total"],
        "offset": result["offset"], "memories": items,
    })


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

    WHEN TO USE: When the user corrects a fact you recalled from memory.
    - User: "Actually, we're using Python 3.12, not 3.11" → correct the memory.
    - User: "No, Alice is the backend lead, not frontend" → correct the memory.
    Preserves history and transfers entity relationships to the new memory.

    EXAMPLE:
    correct(rid="abc123", new_text="Project uses Python 3.12", correction_note="User corrected version from 3.11 to 3.12")

    Args:
        rid: The memory ID to correct.
        new_text: The corrected text content.
        new_importance: Optional updated importance (0.0-1.0).
        new_valence: Optional updated valence (-1.0 to 1.0).
        correction_note: Why the correction was made — helps with audit trail.

    Returns the original and corrected memory IDs.
    """
    if not new_text or not new_text.strip():
        return json.dumps({"error": "new_text must be non-empty"})

    db, lock = _get_db(ctx)
    try:
        with lock:
            result = db.correct(
                rid,
                new_text,
                new_importance=new_importance,
                new_valence=new_valence,
                correction_note=correction_note,
            )
    except Exception as e:
        return json.dumps({"error": str(e), "rid": rid})
    return json.dumps({
        "original_rid": result["original_rid"],
        "corrected_rid": result["corrected_rid"],
        "original_tombstoned": result["original_tombstoned"],
    })


@mcp.tool()
def update_importance(
    rid: str,
    importance: float,
    ctx: Context = None,
) -> str:
    """Adjust the importance of an existing memory.

    WHEN TO USE: When you realize a memory is more or less important than originally scored.
    - A previously minor preference turns out to be critical → increase importance.
    - A "decision" turns out to be tentative → decrease importance.

    Args:
        rid: The memory ID to update.
        importance: New importance score (0.0-1.0).

    Returns confirmation of the update.
    """
    db, lock = _get_db(ctx)
    with lock:
        mem = db.get(rid)
        if mem is None:
            return json.dumps({"error": "Memory not found", "rid": rid})
        result = db.correct(
            rid,
            mem["text"],
            new_importance=importance,
            correction_note=f"Importance adjusted from {mem['importance']} to {importance}",
        )
    return json.dumps({
        "rid": result["corrected_rid"],
        "old_importance": mem["importance"],
        "new_importance": importance,
        "status": "updated",
    })


# ── Entity / Graph Tools ──


@mcp.tool()
def relate(
    source: str,
    target: str,
    relationship: str = "related_to",
    weight: float = 1.0,
    ctx: Context = None,
) -> str:
    """Create a relationship between two entities in the knowledge graph.

    WHEN TO USE: When the conversation reveals relationships between people,
    projects, technologies, organizations, or concepts. Building the knowledge
    graph improves recall — memories connected to entities you search for
    get boosted in results.

    EXAMPLES:
    - "Alice manages the backend team" → relate("Alice", "backend team", "manages")
    - "Project X uses React and TypeScript" → relate("Project X", "React", "uses") + relate("Project X", "TypeScript", "uses")
    - "Bob reports to Alice" → relate("Bob", "Alice", "reports_to")
    - "Redis is our cache layer" → relate("Redis", "infrastructure", "serves_as", weight=0.9)

    NAMING: Use consistent, capitalized entity names — "Alice" not "alice", "Project X" not "project x".

    Args:
        source: Source entity name (e.g. "Alice", "Project X", "PostgreSQL").
        target: Target entity name.
        relationship: Relationship type — "works_at", "manages", "reports_to", "uses", "knows", "related_to", etc.
        weight: Relationship strength (0.0-1.0). 1.0 = definite, 0.5 = possible.

    Returns the edge ID of the created relationship.
    """
    db, lock = _get_db(ctx)
    with lock:
        edge_id = db.relate(source, target, relationship, weight)
    return json.dumps({"edge_id": edge_id, "source": source, "target": target, "relationship": relationship})


@mcp.tool()
def entity_edges(entity: str, ctx: Context = None) -> str:
    """Get all relationships for an entity from the knowledge graph.

    WHEN TO USE: When you want to understand everything connected to a person,
    project, or concept. Useful for building context about an entity before
    responding.

    EXAMPLE: entity_edges("Alice") → shows all of Alice's relationships (manages, works_at, knows, etc.)

    Args:
        entity: The entity name to look up.

    Returns all edges (relationships) connected to this entity.
    """
    db, lock = _get_db(ctx)
    with lock:
        edges = db.get_edges(entity)
    items = [
        {
            "edge_id": e["edge_id"],
            "src": e["src"],
            "dst": e["dst"],
            "rel_type": e["rel_type"],
            "weight": e["weight"],
        }
        for e in edges
    ]
    return json.dumps({"entity": entity, "count": len(items), "edges": items})


@mcp.tool()
def search_entities(
    pattern: str,
    limit: int = 20,
    ctx: Context = None,
) -> str:
    """Search for entities in the knowledge graph by name pattern.

    WHEN TO USE: When you want to find entities matching a partial name, or browse
    what entities exist. Useful before calling entity_edges() or relate().

    EXAMPLES:
    - search_entities("Ali") → finds "Alice", "Alibaba", etc.
    - search_entities("project") → finds all project entities

    Args:
        pattern: Search pattern (case-insensitive substring match).
        limit: Maximum results (default 20).

    Returns matching entity names with their edge counts.
    """
    db, lock = _get_db(ctx)
    with lock:
        entities = db.search_entities(pattern=pattern, limit=limit)
    items = [
        {
            "name": e["name"],
            "type": e.get("entity_type"),
            "mention_count": e.get("mention_count", 0),
            "first_seen": e.get("first_seen"),
            "last_seen": e.get("last_seen"),
        }
        for e in entities
    ]
    return json.dumps({"pattern": pattern, "count": len(items), "entities": items})


# ── Cognition Tools ──


@mcp.tool()
def think(
    run_consolidation: bool = True,
    run_conflict_scan: bool = True,
    run_pattern_mining: bool = True,
    ctx: Context = None,
) -> str:
    """Run the cognitive maintenance loop — consolidate, detect conflicts, mine patterns.

    WHEN TO USE:
    - At the end of a long conversation with many new memories stored.
    - When you suspect contradictory memories exist.
    - Periodically to keep the memory system healthy and efficient.
    - When recall returns many similar/overlapping memories (consolidation needed).

    WHAT IT DOES:
    - Consolidation: Merges highly similar memories into summaries, reducing noise.
    - Conflict scan: Finds contradictions (e.g., "uses Postgres" vs "uses MySQL").
    - Pattern mining: Detects recurring themes and trends across memories.
    - Trigger generation: Creates actionable alerts (decaying memories, insights).

    Args:
        run_consolidation: Merge similar memories (default True).
        run_conflict_scan: Find contradictions (default True).
        run_pattern_mining: Detect patterns (default True).

    Returns a summary: consolidations made, conflicts found, patterns detected, and triggers.
    """
    db, lock = _get_db(ctx)
    config = {
        "run_consolidation": run_consolidation,
        "run_conflict_scan": run_conflict_scan,
        "run_pattern_mining": run_pattern_mining,
    }
    with lock:
        result = db.think(config)
    triggers = []
    for t in result["triggers"]:
        triggers.append({
            "trigger_type": t["trigger_type"],
            "reason": t["reason"],
            "urgency": t["urgency"],
            "suggested_action": t["suggested_action"],
        })
    return json.dumps({
        "triggers": triggers,
        "consolidation_count": result["consolidation_count"],
        "conflicts_found": result["conflicts_found"],
        "patterns_new": result["patterns_new"],
        "patterns_updated": result["patterns_updated"],
        "expired_triggers": result["expired_triggers"],
        "duration_ms": round(result["duration_ms"], 2),
    })


# ── Conflict Tools ──


@mcp.tool()
def conflicts(
    status: str | None = None,
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """List memory conflicts (contradictions) that need resolution.

    WHEN TO USE:
    - After `think` reports conflicts_found > 0.
    - When a user says something that contradicts what you recall.
    - Proactively check after storing memories that might conflict with existing ones.

    EXAMPLE: A conflict might look like:
    - Memory A: "Team uses PostgreSQL for the API database"
    - Memory B: "Team decided to switch to MySQL for the API"
    → Resolve by asking the user which is current, then use conflict_resolve.

    Args:
        status: Filter: "open", "resolved", "dismissed". None for all.
        limit: Maximum conflicts to return (default 10).

    Returns conflicts with IDs, types, priorities, and the conflicting memories.
    """
    db, lock = _get_db(ctx)
    with lock:
        conflict_list = db.get_conflicts(status=status, limit=limit)
    items = [
        {
            "conflict_id": c["conflict_id"],
            "conflict_type": c["conflict_type"],
            "priority": c["priority"],
            "status": c["status"],
            "memory_a": c["memory_a"],
            "memory_b": c["memory_b"],
            "entity": c["entity"],
            "detection_reason": c["detection_reason"],
        }
        for c in conflict_list
    ]
    return json.dumps({"count": len(items), "conflicts": items})


@mcp.tool()
def conflict_resolve(
    conflict_id: str,
    strategy: str,
    winner_rid: str | None = None,
    new_text: str | None = None,
    resolution_note: str | None = None,
    ctx: Context = None,
) -> str:
    """Resolve a memory conflict using a strategy.

    WHEN TO USE: After reviewing a conflict from `conflicts` and determining
    the correct resolution — either by checking with the user or by inference.

    STRATEGIES:
    - "keep_a": Memory A is correct, tombstone B.
    - "keep_b": Memory B is correct, tombstone A.
    - "keep_both": Both are valid (not actually conflicting), mark resolved.
    - "merge": Combine into a new memory with new_text.

    Args:
        conflict_id: The conflict ID to resolve.
        strategy: One of "keep_a", "keep_b", "keep_both", "merge".
        winner_rid: For keep_a/keep_b, which memory wins (optional, inferred from strategy).
        new_text: For "merge" strategy, the combined text.
        resolution_note: Why this resolution was chosen.

    Returns the resolution result.
    """
    valid_strategies = ("keep_a", "keep_b", "keep_both", "merge")
    if strategy not in valid_strategies:
        return json.dumps({"error": f"strategy must be one of {valid_strategies}, got '{strategy}'"})
    if strategy == "merge" and (not new_text or not new_text.strip()):
        return json.dumps({"error": "new_text is required for 'merge' strategy"})

    db, lock = _get_db(ctx)
    try:
        with lock:
            result = db.resolve_conflict(
                conflict_id,
                strategy,
                winner_rid=winner_rid,
                new_text=new_text,
                resolution_note=resolution_note,
            )
    except Exception as e:
        return json.dumps({"error": str(e), "conflict_id": conflict_id})
    return json.dumps({
        "conflict_id": result["conflict_id"],
        "strategy": result["strategy"],
        "winner_rid": result.get("winner_rid"),
        "loser_tombstoned": result.get("loser_tombstoned", False),
        "new_memory_rid": result.get("new_memory_rid"),
    })


# ── Feedback / Learning Tools ──


@mcp.tool()
def recall_feedback(
    rid: str,
    feedback: str,
    query_text: str | None = None,
    score_at_retrieval: float | None = None,
    rank_at_retrieval: int | None = None,
    ctx: Context = None,
) -> str:
    """Provide feedback on a recall result to improve future retrieval.

    WHEN TO USE: When you can determine a recalled memory was clearly relevant
    or clearly irrelevant to what you needed. After ~20 feedback signals, the
    engine adapts its scoring weights automatically.

    Args:
        rid: The memory ID to provide feedback on.
        feedback: "relevant" or "irrelevant".
        query_text: The original query (helps learning).
        score_at_retrieval: The score the memory received.
        rank_at_retrieval: The rank position in results.

    Returns confirmation.
    """
    db, lock = _get_db(ctx)
    with lock:
        db.recall_feedback(
            rid=rid,
            feedback=feedback,
            query_text=query_text,
            score_at_retrieval=score_at_retrieval,
            rank_at_retrieval=rank_at_retrieval,
        )
    return json.dumps({"rid": rid, "feedback": feedback, "status": "recorded"})


# ── Trigger Tools ──


@mcp.tool()
def triggers(limit: int = 10, ctx: Context = None) -> str:
    """Get pending proactive triggers — insights, warnings, and suggestions from the memory system.

    WHEN TO USE: After calling `think`, check triggers for actionable items like:
    - Important memories that are decaying and need reinforcement
    - Consolidation opportunities (many similar memories)
    - Detected patterns across memories
    - Relationship insights from the knowledge graph

    Args:
        limit: Maximum triggers to return (default 10).

    Returns pending triggers sorted by urgency.
    """
    db, lock = _get_db(ctx)
    with lock:
        trigger_list = db.get_pending_triggers(limit=limit)
    items = [
        {
            "trigger_id": t["trigger_id"],
            "trigger_type": t["trigger_type"],
            "urgency": t["urgency"],
            "reason": t["reason"],
            "suggested_action": t["suggested_action"],
            "source_rids": t["source_rids"],
        }
        for t in trigger_list
    ]
    return json.dumps({"count": len(items), "triggers": items})


# ── Health & Stats ──


@mcp.tool()
def health_check(ctx: Context = None) -> str:
    """Verify that the YantrikDB memory system is operational.

    WHEN TO USE: At the start of a session if you're unsure the memory server
    is working, or to diagnose issues when other tools fail.

    Returns server status, database path, memory counts, and uptime.
    """
    db, lock = _get_db(ctx)
    t0 = time.time()
    with lock:
        stats = db.stats()
    latency_ms = round((time.time() - t0) * 1000, 1)
    return json.dumps({
        "status": "ok",
        "latency_ms": latency_ms,
        "active_memories": stats.get("active_memories", 0),
        "total_entities": stats.get("entities", 0),
        "total_edges": stats.get("edges", 0),
        "open_conflicts": stats.get("open_conflicts", 0),
    })


@mcp.tool()
def stats(namespace: str | None = None, ctx: Context = None) -> str:
    """Get detailed memory engine statistics.

    WHEN TO USE: When you want to understand the state of the memory system —
    how many memories exist, how many are active vs archived, conflict counts, etc.

    Args:
        namespace: Filter to a specific namespace. None for global stats.

    Returns comprehensive statistics: memory counts by status, entity/edge counts,
    conflicts, triggers, patterns, and index sizes.
    """
    db, lock = _get_db(ctx)
    with lock:
        result = db.stats(namespace=namespace)
    return json.dumps(result)


# ── Personality (V13) ──


@mcp.tool()
def personality(recompute: bool = False, ctx: Context = None) -> str:
    """Get the AI personality profile — trait scores derived from memory patterns.

    WHEN TO USE: When you want to understand or communicate personality traits
    (warmth, depth, energy, attentiveness) that have emerged from stored memories.
    Call with recompute=True after significant new memories to refresh traits.

    EXAMPLES:
    - personality() → current cached profile
    - personality(recompute=True) → recompute from memory signals then return

    Args:
        recompute: If True, re-derive traits from memory patterns before returning.

    Returns trait scores (0-1), confidence, and sample counts.
    """
    db, lock = _get_db(ctx)
    with lock:
        if recompute:
            profile = db.derive_personality()
        else:
            profile = db.get_personality()
    return json.dumps(profile)


@mcp.tool()
def set_personality(trait_name: str, score: float, ctx: Context = None) -> str:
    """Manually set a personality trait score.

    WHEN TO USE: When the user explicitly wants to tune the AI personality —
    e.g., "be warmer" or "be more attentive". Also useful for testing.

    EXAMPLES:
    - set_personality("warmth", 0.8) → make the AI warmer
    - set_personality("energy", 0.3) → make the AI calmer

    Args:
        trait_name: One of: warmth, depth, energy, attentiveness.
        score: Value between 0.0 and 1.0.

    Returns whether the trait was updated.
    """
    db, lock = _get_db(ctx)
    with lock:
        updated = db.set_personality_trait(trait_name, score)
    return json.dumps({"trait": trait_name, "score": score, "updated": updated})


# ── Patterns (V13) ──


@mcp.tool()
def patterns(
    pattern_type: str | None = None,
    status: str = "active",
    limit: int = 20,
    ctx: Context = None,
) -> str:
    """Get discovered patterns across memories — co-occurrences, trends, cross-domain links.

    WHEN TO USE: After calling `think`, check what patterns the engine has discovered.
    Cross-domain patterns reveal surprising connections (e.g., work stress correlating
    with health entries). Entity bridges show people/concepts that span multiple domains.

    EXAMPLES:
    - patterns() → all active patterns
    - patterns(pattern_type="cross_domain") → only cross-domain discoveries
    - patterns(pattern_type="entity_bridge") → entities bridging domains

    Pattern types: co_occurrence, temporal_cluster, valence_trend, topic_cluster,
    entity_hub, cross_domain, entity_bridge.

    Args:
        pattern_type: Filter by type, or None for all.
        status: Filter by status (active, stale). Default "active".
        limit: Maximum patterns to return.

    Returns discovered patterns with confidence, evidence, and context.
    """
    db, lock = _get_db(ctx)
    with lock:
        result = db.get_patterns(
            pattern_type=pattern_type, status=status, limit=limit
        )
    return json.dumps({"count": len(result), "patterns": result})


# ── Archive / Hydrate (V13) ──


@mcp.tool()
def archive(rid: str, ctx: Context = None) -> str:
    """Move a memory to cold storage (archived). Reduces clutter while preserving data.

    WHEN TO USE: When a memory is old or low-relevance but shouldn't be forgotten.
    Archived memories are excluded from recall but can be hydrated back.

    Args:
        rid: The memory ID to archive.

    Returns confirmation with the archived memory ID.
    """
    db, lock = _get_db(ctx)
    with lock:
        archived = db.archive(rid)
    if not archived:
        return json.dumps({"error": f"Memory '{rid}' not found or already archived"})
    return json.dumps({"archived": rid, "status": "cold"})


@mcp.tool()
def hydrate(rid: str, ctx: Context = None) -> str:
    """Restore an archived memory back to active (hot) storage.

    WHEN TO USE: When an archived memory becomes relevant again — user asks about
    something old, or a pattern links to archived data.

    Args:
        rid: The memory ID to hydrate.

    Returns confirmation with the restored memory ID.
    """
    db, lock = _get_db(ctx)
    with lock:
        hydrated = db.hydrate(rid)
    if not hydrated:
        return json.dumps({"error": f"Memory '{rid}' not found or already hot"})
    return json.dumps({"hydrated": rid, "status": "hot"})


# ── Learned Weights (V13) ──


@mcp.tool()
def learned_weights(ctx: Context = None) -> str:
    """Show how the recall scoring system has adapted from feedback.

    WHEN TO USE: For transparency — understand why certain memories rank higher.
    Shows the current weights for similarity, decay, recency, and other scoring signals.
    These weights evolve as you provide recall_feedback.

    Returns scoring weights: w_sim, w_decay, w_recency, gate_tau, alpha_imp,
    keyword_boost, and the generation count (how many learning iterations).
    """
    db, lock = _get_db(ctx)
    with lock:
        weights = db.learned_weights()
    return json.dumps(weights)


# ── Trigger Lifecycle (V13) ──


@mcp.tool()
def acknowledge_trigger(trigger_id: str, ctx: Context = None) -> str:
    """Mark a trigger as acknowledged — you've seen it and may act on it.

    WHEN TO USE: After surfacing a trigger to the user. Completes the trigger
    lifecycle: pending → delivered → acknowledged → acted/dismissed.

    Args:
        trigger_id: The trigger ID to acknowledge.

    Returns whether the trigger was acknowledged.
    """
    db, lock = _get_db(ctx)
    with lock:
        result = db.acknowledge_trigger(trigger_id)
    return json.dumps({"trigger_id": trigger_id, "acknowledged": result})


# ── Session Management (V14) ──


@mcp.tool()
def session_start(
    namespace: str = "default",
    client_id: str = "default",
    metadata: dict | None = None,
    ctx: Context = None,
) -> str:
    """Start a new interaction session. Memories recorded during the session
    are automatically linked to it.

    WHEN TO USE: At the start of a conversation or work session. Only one
    active session per (namespace, client_id) is allowed.

    Args:
        namespace: Memory namespace (default: "default").
        client_id: Identifier for the client/agent (default: "default").
        metadata: Optional metadata dict (e.g. {"workspace": "myproject"}).

    Returns the session_id.
    """
    db, lock = _get_db(ctx)
    with lock:
        meta = metadata or {}
        sid = db.session_start(namespace, client_id, meta)
    return json.dumps({"session_id": sid})


@mcp.tool()
def session_end(
    session_id: str,
    summary: str | None = None,
    ctx: Context = None,
) -> str:
    """End an active session. Computes summary stats (memory count, avg
    valence, topics, duration).

    WHEN TO USE: At the end of a conversation. The summary is stored and
    available for future session_awareness triggers.

    Args:
        session_id: The session ID to end.
        summary: Optional summary of what happened in this session.

    Returns session stats.
    """
    db, lock = _get_db(ctx)
    with lock:
        result = db.session_end(session_id, summary)
    return json.dumps(result)


@mcp.tool()
def session_history(
    namespace: str = "default",
    client_id: str = "default",
    limit: int = 10,
    ctx: Context = None,
) -> str:
    """View past sessions for a client. Shows timing, memory count, topics,
    and emotional valence for each session.

    WHEN TO USE: To understand interaction history — when was the last session,
    what was discussed, how did the user feel.

    Args:
        namespace: Memory namespace.
        client_id: Client identifier.
        limit: Max sessions to return.

    Returns list of past sessions.
    """
    db, lock = _get_db(ctx)
    with lock:
        sessions = db.session_history(namespace, client_id, limit)
    return json.dumps(sessions)


# ── Temporal Queries (V14) ──


@mcp.tool()
def stale_memories(
    days: float = 30.0,
    limit: int = 20,
    namespace: str | None = None,
    ctx: Context = None,
) -> str:
    """Find important memories that haven't been accessed recently.

    WHEN TO USE: During cognitive maintenance — identify knowledge that may be
    outdated. Good to run periodically or when the user asks "what have I forgotten?"

    Args:
        days: How many days of inactivity counts as stale.
        limit: Max results.
        namespace: Optional namespace filter.

    Returns stale memories ordered by importance.
    """
    db, lock = _get_db(ctx)
    with lock:
        memories = db.stale(days, limit, namespace)
    return json.dumps([
        {"rid": m["rid"], "text": m["text"], "importance": m["importance"],
         "days_since_access": (time.time() - m.get("last_access", time.time())) / 86400}
        for m in memories
    ])


@mcp.tool()
def upcoming_memories(
    days: float = 7.0,
    limit: int = 20,
    namespace: str | None = None,
    ctx: Context = None,
) -> str:
    """Find memories with upcoming due dates (deadlines, reminders, events).

    WHEN TO USE: Proactively at session start to surface what's coming up.
    Also useful when user asks about their schedule or upcoming events.

    Args:
        days: Look ahead window in days.
        limit: Max results.
        namespace: Optional namespace filter.

    Returns memories due within the specified window, sorted by due date.
    """
    db, lock = _get_db(ctx)
    with lock:
        memories = db.upcoming(days, limit, namespace)
    return json.dumps([
        {"rid": m["rid"], "text": m["text"], "importance": m["importance"],
         "due_at": m.get("due_at"), "temporal_kind": m.get("temporal_kind")}
        for m in memories
    ])


# ── Entity Intelligence (V14) ──


@mcp.tool()
def entity_profile(
    entity: str,
    days: float = 90.0,
    namespace: str | None = None,
    ctx: Context = None,
) -> str:
    """Get a rich profile of an entity: mention frequency, valence trends,
    domains, sessions, and emotional patterns.

    WHEN TO USE: When the user asks about a person, project, or topic.
    Reveals how deeply the system knows an entity and the emotional context
    around it.

    Args:
        entity: Entity name (e.g. "Alice", "ProjectX", "Redis").
        days: Time window in days for analysis.
        namespace: Optional namespace filter.

    Returns entity profile with temporal and emotional dimensions.
    """
    db, lock = _get_db(ctx)
    with lock:
        profile = db.entity_profile(entity, days, namespace)
    return json.dumps(profile)


@mcp.tool()
def relationship_depth(
    entity: str,
    namespace: str | None = None,
    ctx: Context = None,
) -> str:
    """Measure how deeply the system knows an entity — not just mention count,
    but sessions together, domain breadth, graph connections, and interaction
    richness.

    WHEN TO USE: To differentiate between "mentioned once" and "deeply known".
    The depth_score (0.0-1.0) reflects true relationship depth. Use this to
    calibrate how confidently you speak about an entity.

    Args:
        entity: Entity name to profile.
        namespace: Optional namespace filter.

    Returns relationship depth metrics including composite depth_score.
    """
    db, lock = _get_db(ctx)
    with lock:
        depth = db.relationship_depth(entity, namespace)
    return json.dumps(depth)


# ── Procedural Memory (V14) ──


@mcp.tool()
def learn_procedure(
    text: str,
    domain: str = "general",
    task_context: str = "",
    effectiveness: float = 0.5,
    namespace: str = "default",
    ctx: Context = None,
) -> str:
    """Store a learned procedure — what worked in a specific context.

    WHEN TO USE: When you discover an effective approach, strategy, or technique
    during a task. The engine will surface relevant procedures for similar future
    tasks. Over time, effective procedures get reinforced and ineffective ones decay.

    EXAMPLES:
    - "When searching this codebase, the Agent tool with Explore subtype works better than direct Grep for architectural questions"
    - "For this user, always show code diffs rather than describing changes verbally"
    - "Breaking large PRs into domain-specific commits gets faster reviews in this repo"

    Args:
        text: Description of the procedure/strategy.
        domain: Task domain this applies to.
        task_context: What kind of task this procedure helps with.
        effectiveness: How well it worked (0.0-1.0). Higher = more important.
        namespace: Memory namespace.

    Returns the memory RID.
    """
    db, lock = _get_db(ctx)
    with lock:
        rid = db.record_procedural(text, None, domain, task_context, effectiveness, namespace)
    return json.dumps({"rid": rid, "type": "procedural", "effectiveness": effectiveness})


@mcp.tool()
def surface_procedures(
    query: str,
    domain: str | None = None,
    top_k: int = 5,
    namespace: str | None = None,
    ctx: Context = None,
) -> str:
    """Find learned procedures relevant to the current task.

    WHEN TO USE: At the start of a task, to check if past sessions produced
    relevant strategies. The engine returns procedures that matched similar
    contexts, ranked by effectiveness.

    Args:
        query: Describe what you're about to do.
        domain: Optional domain filter.
        top_k: Max procedures to return.
        namespace: Optional namespace filter.

    Returns relevant procedures with effectiveness scores.
    """
    db, lock = _get_db(ctx)
    with lock:
        emb = db.embed(query)
        results = db.surface_procedural(emb, query, domain, top_k, namespace)
    return json.dumps([
        {"rid": r["rid"], "text": r["text"], "score": r["score"],
         "importance": r["importance"], "certainty": r.get("certainty", 0.5)}
        for r in results
    ])


@mcp.tool()
def reinforce_procedure(
    rid: str,
    outcome: float,
    ctx: Context = None,
) -> str:
    """Update a procedure's effectiveness based on how well it worked.

    WHEN TO USE: After using a surfaced procedure, report whether it was helpful.
    1.0 = fully successful, 0.5 = partially useful, 0.0 = not helpful at all.
    This drives adaptive learning — good procedures get stronger over time.

    Args:
        rid: The procedure's memory RID.
        outcome: How effective the procedure was (0.0-1.0).

    Returns whether the reinforcement was applied.
    """
    db, lock = _get_db(ctx)
    with lock:
        result = db.reinforce_procedural(rid, outcome)
    return json.dumps({"rid": rid, "reinforced": result, "outcome": outcome})


# ── Substitution Category Tools (V14) ──


@mcp.tool()
def reclassify_conflict(
    conflict_id: str,
    new_type: str,
    ctx: Context = None,
) -> str:
    """Reclassify a conflict and teach the engine about substitution patterns.

    WHEN TO USE: When the engine misclassified a conflict type. For example, if
    "PostgreSQL" vs "MySQL" was flagged as an entity substitution but should be
    a preference, reclassify it. The engine learns from this: it adds the tokens
    to the appropriate substitution category so future conflicts are caught
    automatically.

    Args:
        conflict_id: The conflict to reclassify.
        new_type: The correct conflict type — one of "identity_fact", "preference", "temporal".

    Returns the reclassification result including any newly learned category members.
    """
    db, lock = _get_db(ctx)
    with lock:
        result = db.reclassify_conflict(conflict_id, new_type)
    return json.dumps(result)


@mcp.tool()
def substitution_categories(
    ctx: Context = None,
) -> str:
    """List all substitution categories the engine knows about.

    WHEN TO USE: To inspect what semantic categories the engine uses for
    conflict detection. Shows databases, cloud providers, languages, etc.
    with member counts. Useful for understanding what the engine can detect
    and what gaps exist.

    Returns a list of categories with their member counts and status.
    """
    db, lock = _get_db(ctx)
    with lock:
        cats = db.substitution_categories()
    return json.dumps(cats)


@mcp.tool()
def learn_category_members(
    category_name: str,
    members: list[list],
    source: str = "llm_suggested",
    ctx: Context = None,
) -> str:
    """Teach the engine new members of a substitution category.

    WHEN TO USE: When you identify tokens that belong to an existing category
    but aren't recognized yet. For example, adding "tidb" to the "databases"
    category. LLM-suggested members start at low confidence and only affect
    conflict detection after user confirmation.

    Args:
        category_name: The category to add members to (e.g. "databases").
        members: List of [token, confidence] pairs. Example: [["tidb", 0.35], ["surrealdb", 0.35]].
        source: Origin of the suggestion — "llm_suggested", "user_confirmed", or "seed".

    Returns the number of new members added.
    """
    db, lock = _get_db(ctx)
    member_tuples = [(m[0], float(m[1])) for m in members]
    with lock:
        count = db.learn_category_members(category_name, member_tuples, source)
    return json.dumps({"category": category_name, "new_members": count, "source": source})
