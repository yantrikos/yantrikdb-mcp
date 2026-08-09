---
name: yantrikdb
description: Persistent cognitive memory via YantrikDB's MCP server. Semantic recall, remember, belief revision, knowledge graph, contradiction detection, and procedural learning that survive across sessions and agents. Tools are auto-discovered from the server at runtime.
---

# YantrikDB — persistent memory

Talk to a YantrikDB memory server from the IPython kernel. Memory persists
across sessions and across harnesses: what you `remember` here, a future
session — or a different agent pointed at the same database — can `recall`.

## Setup

Requires a running yantrikdb-mcp server on the streamable-http transport:

```bash
YANTRIKDB_API_KEY=<your-token> yantrikdb-mcp --transport streamable-http --host 127.0.0.1 --port 8420
```

and the same `YANTRIKDB_API_KEY` value in Prime Agent's environment. The
server URL defaults to `http://127.0.0.1:8420/mcp`; override it with a
`yantrikdb` entry under `mcpServers` in `~/.prime/agent/settings.json`.

## Usage

Discover before you call — the server defines the tool set:

```python
import yantrikdb

for tool in await yantrikdb.list_tools():
    print(tool["name"], "-", tool["description"])

# Store a durable fact
await yantrikdb.remember(text="User prefers dark mode in the editor", importance=0.7)

# Retrieve by meaning, not keywords
hits = await yantrikdb.recall(query="what editor settings does the user like")
print(hits)

# Revise a fact without losing history (do NOT remember a contradiction)
await yantrikdb.correct(query="editor preference", new_text="User switched to light mode", reason="user said so")
```

Notes:
- Every tool is an `async` method — always `await`.
- Results are already-parsed Python (a `dict` for structured output, otherwise a string).
- Escape hatch for non-identifier tool names: `await yantrikdb.call_tool("name", {...})`.
- If a call raises `NotEnabled`, the `YANTRIKDB_API_KEY` environment variable
  is not visible to Prime Agent — set it in the shell that launches the agent.
