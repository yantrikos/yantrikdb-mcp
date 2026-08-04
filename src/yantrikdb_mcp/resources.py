"""MCP resource definitions for YantrikDB.

STATIC vs TEMPLATED resources and `Context`
-------------------------------------------
mcp 2.x refuses to register a STATIC resource (one whose URI has no template
variables) whose handler declares a `Context` parameter:

    ValueError: Resource 'yantrikdb://stats' has no URI template variables,
    but the handler declares a Context parameter. Context injection for
    static resources is not supported.

That is a real behavioural divergence from 1.x, not a symbol move — it was
caught by running the suite against mcp 2.0.0 rather than by reading the
diff. Static resources here therefore reach the engine through the
process-singleton `_get_lazy_singleton()` instead of through the request
context. That is not a workaround: the singleton is the *intended* owner of
the handle (yantrikos/yantrikdb-mcp#11), the lifespan context merely hands
out the same object, and a resource that needs no per-request state has no
reason to demand one. The result registers identically on both SDK lines.

Templated resources (e.g. `yantrikdb://memory/{rid}`) may still take
`Context` on both lines, but use the same accessor for consistency.
"""

import json

from .server import _get_lazy_singleton, mcp


def _db():
    """The process-singleton engine handle.

    Deliberately not `ctx.request_context.lifespan_context` — see the module
    docstring. The lifespan yields this very object, so both paths return the
    same instance; this one just doesn't require a Context to exist.
    """
    return _get_lazy_singleton().db


@mcp.resource("yantrikdb://stats")
def stats_resource() -> str:
    """Current YantrikDB engine statistics — memory counts, entities, conflicts, patterns."""
    return json.dumps(_db().stats(), indent=2)


@mcp.resource("yantrikdb://memory/{rid}")
def memory_resource(rid: str) -> str:
    """A specific memory record by ID."""
    mem = _db().get(rid)
    if mem is None:
        return json.dumps({"error": "Memory not found", "rid": rid})
    return json.dumps(mem, indent=2)


@mcp.resource("yantrikdb://health")
def health_resource() -> str:
    """Server health status — use to verify the memory system is operational."""
    stats = _db().stats()
    return json.dumps({
        "status": "ok",
        # Engine key is `active_memories` (both embedded dict and the HTTP
        # _Stats wrapper) — reading "active" here silently reported 0
        # memories on every healthy server.
        "active_memories": stats.get("active_memories", 0),
        "total_entities": stats.get("entities", 0),
        "total_edges": stats.get("edges", 0),
    }, indent=2)
