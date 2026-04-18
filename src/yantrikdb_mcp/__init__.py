"""YantrikDB MCP Server — expose the cognitive memory engine as MCP tools."""

import importlib.metadata
import sys

from .server import mcp

__version__ = importlib.metadata.version("yantrikdb-mcp")


def _cli_arg(name):
    """Get a CLI argument value by --name."""
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return None


def main():
    """Entry point for the yantrikdb-mcp console script."""
    if "--version" in sys.argv or "-V" in sys.argv:
        print(f"yantrikdb-mcp {__version__}")
        sys.exit(0)

    if "--help" in sys.argv or "-h" in sys.argv:
        print(f"yantrikdb-mcp {__version__}")
        print("Usage: yantrikdb-mcp [OPTIONS]")
        print()
        print("Options:")
        print("  --transport <stdio|sse|streamable-http>  Transport protocol (default: stdio)")
        print("  --host <host>          Bind address for SSE/HTTP (default: 0.0.0.0)")
        print("  --port <port>          Port for SSE/HTTP (default: 8420)")
        print("  --version, -V          Show version and exit")
        print("  --help, -h             Show this help and exit")
        print()
        print("Environment variables:")
        print("  YANTRIKDB_DB_PATH          Database file path (default: ~/.yantrikdb/memory.db)")
        print("  YANTRIKDB_EMBEDDING_MODEL  Sentence transformer model (default: all-MiniLM-L6-v2)")
        print("  YANTRIKDB_EMBEDDING_DIM    Embedding dimension (default: 384)")
        print("  YANTRIKDB_API_KEY          Bearer token for SSE/HTTP auth (required for network transports)")
        print()
        print("Remote cluster mode (set YANTRIKDB_SERVER_URL to use HTTP backend instead of embedded engine):")
        print("  YANTRIKDB_SERVER_URL       Comma-separated cluster node URLs (e.g. http://node1:7438,http://node2:7438)")
        print("  YANTRIKDB_TOKEN            Bearer token for the cluster database")
        sys.exit(0)

    transport = _cli_arg("--transport") or "stdio"

    # Network transports: configure host/port and optional auth
    if transport in ("sse", "streamable-http"):
        _run_network(transport)
    else:
        mcp.run(transport="stdio")


def _run_network(transport: str):
    """Run with SSE or streamable-http transport, with optional bearer token auth."""
    import logging
    import os

    import anyio
    import uvicorn

    log = logging.getLogger("yantrikdb.mcp")

    host = _cli_arg("--host") or "0.0.0.0"
    port = int(_cli_arg("--port") or "8420")

    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.transport_security.enable_dns_rebinding_protection = False
    mcp.settings.transport_security.allowed_hosts = ["*"]
    mcp.settings.transport_security.allowed_origins = ["*"]

    # Build the ASGI app
    if transport == "sse":
        app = mcp.sse_app()
    else:
        app = mcp.streamable_http_app()

    # Wrap with bearer token auth if configured
    api_key = os.environ.get("YANTRIKDB_API_KEY")
    if api_key:
        from .auth import BearerTokenMiddleware
        app = BearerTokenMiddleware(app, api_key=api_key)
        log.info("Bearer token auth enabled")
    else:
        log.warning(
            "No YANTRIKDB_API_KEY set — server is UNAUTHENTICATED. "
            "Set YANTRIKDB_API_KEY env var to require bearer token auth."
        )

    log.info("Starting %s transport on %s:%d", transport, host, port)

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    anyio.run(server.serve)
