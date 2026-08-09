"""YantrikDB integration: persistent-memory tools auto-discovered from the MCP server.

Usage in the kernel:

    import yantrikdb
    hits = await yantrikdb.recall(query="past decisions about the deploy")
"""

from __future__ import annotations

from rlm import McpIntegration

__all__ = ["YantrikDB", "yantrikdb"]


class YantrikDB(McpIntegration):
    server = "yantrikdb"
    # Fallback when no mcpServers["yantrikdb"] entry exists in settings.json;
    # the host config wins when present.
    url = "http://127.0.0.1:8420/mcp"
    # Without this every call raises NotEnabled: the host only injects OAuth
    # tokens from auth.json, and yantrikdb-mcp uses a static bearer token.
    bearer_token_env = "YANTRIKDB_API_KEY"


yantrikdb = YantrikDB()

# Names the kernel bootstrap probes to decide if a module is a callable skill.
# Forwarding them would make `getattr(module, "run")` return an MCP tool stub
# and the module would be wrapped as callable, breaking `await yantrikdb.<tool>()`.
_RESERVED = {"run", "__wrapped__", "__call__"}


def __getattr__(name: str):
    # Forward bare module-level access (e.g. yantrikdb.recall) to the instance,
    # so `import yantrikdb; await yantrikdb.recall(...)` works without `.yantrikdb`.
    if name.startswith("_") or name in _RESERVED:
        raise AttributeError(name)
    return getattr(yantrikdb, name)
