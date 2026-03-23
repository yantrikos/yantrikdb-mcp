"""MCP resource definitions for YantrikDB."""

import json

from mcp.server.fastmcp import Context

from .server import mcp


def _get_db(ctx: Context):
    lc = ctx.request_context.lifespan_context
    lazy = lc["lazy"]
    return lazy.db


@mcp.resource("yantrikdb://stats")
def stats_resource(ctx: Context = None) -> str:
    """Current YantrikDB engine statistics — memory counts, entities, conflicts, patterns."""
    db = _get_db(ctx)
    stats = db.stats()
    return json.dumps(stats, indent=2)


@mcp.resource("yantrikdb://memory/{rid}")
def memory_resource(rid: str, ctx: Context = None) -> str:
    """A specific memory record by ID."""
    db = _get_db(ctx)
    mem = db.get(rid)
    if mem is None:
        return json.dumps({"error": "Memory not found", "rid": rid})
    return json.dumps(mem, indent=2)


@mcp.resource("yantrikdb://health")
def health_resource(ctx: Context = None) -> str:
    """Server health status — use to verify the memory system is operational."""
    db = _get_db(ctx)
    stats = db.stats()
    return json.dumps({
        "status": "ok",
        "active_memories": stats.get("active", 0),
        "total_entities": stats.get("entities", 0),
        "total_edges": stats.get("edges", 0),
    }, indent=2)
