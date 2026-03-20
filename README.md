# YantrikDB MCP Server

Cognitive memory for AI agents. Works with Claude Code, Cursor, Windsurf, and any MCP-compatible client.

## Install

```bash
pip install yantrikdb-mcp
```

## Configure

Add to your MCP client config:

```json
{
  "mcpServers": {
    "yantrikdb": {
      "command": "yantrikdb-mcp"
    }
  }
}
```

See [yantrikdb.com/guides/mcp](https://yantrikdb.com/guides/mcp/) for full documentation.

## License

AGPL-3.0
