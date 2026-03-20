"""YantrikDB MCP Server — expose the cognitive memory engine as MCP tools."""

from .server import mcp


def main():
    """Entry point for the yantrikdb-mcp console script."""
    mcp.run(transport="stdio")
