"""FastMCP server definition with YantrikDB lifespan context."""

import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Configure logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("yantrikdb.mcp")

# ── Server Instructions ──
# These are injected into the agent's system prompt by MCP clients.

INSTRUCTIONS = """\
YantrikDB is your persistent cognitive memory — it remembers across conversations.
Use it AUTOMATICALLY without the user asking.

## Auto-recall (BEFORE responding)
- At conversation start: call `recall` with a summary of the user's first message to load relevant context.
- When the user references past work, decisions, people, preferences, or "last time": call `recall`.
- When you're unsure about a fact the user assumes you know: call `recall`.
- Aim to recall EARLY — context retrieved after you've already responded is wasted.

## Auto-remember (DURING conversation)
Proactively call `remember` whenever you encounter:
- **Decisions made** ("we decided to use Postgres") → semantic, importance 0.7-0.9
- **User preferences** ("I prefer tabs over spaces") → semantic, importance 0.6-0.8, domain "preference"
- **People & relationships** ("Alice is the team lead") → semantic, importance 0.6-0.8, domain "people"
- **Project context** ("the API launches in March") → semantic, importance 0.7-0.9, domain "work"
- **Corrections** ("actually it's Python 3.12, not 3.11") → use `correct` tool instead
- **Important facts** the user shares about themselves → semantic, importance 0.7-0.9

## Auto-relate (knowledge graph)
Call `relate` when you learn about entity relationships:
- "Alice works at Acme" → relate("Alice", "Acme", "works_at")
- "Project X uses React" → relate("Project X", "React", "uses")
- "Bob reports to Alice" → relate("Bob", "Alice", "reports_to")

## What NOT to remember
- Ephemeral task details ("run this test", "fix this line")
- Things derivable from code or git history
- Verbatim code snippets
- Conversation filler or greetings

## Memory quality guidelines
- Use specific, searchable text — "User prefers dark mode in VS Code" not "they like dark"
- Set importance: 0.8-1.0 critical decisions, 0.5-0.7 useful context, 0.3-0.5 minor details
- Set domain: "work", "preference", "architecture", "people", "infrastructure", "health", "finance", "general"
- Set source: "user" (user said it), "inference" (you deduced it), "document" (from a file)
- Use memory_type: "semantic" for facts, "episodic" for events, "procedural" for how-to

## Cognitive maintenance
- Call `think` at the end of long conversations to consolidate and detect conflicts.
- If `think` surfaces conflicts, resolve them or ask the user.
- If recall returns low-confidence results, try `recall_refine` with a rephrased query.
- After `think`, call `patterns` to check for cross-domain discoveries and entity bridges.
- Call `personality` to understand emergent trait scores; call with recompute=True after big sessions.
- Call `acknowledge_trigger` after surfacing triggers to the user.
- Use `archive` for old memories cluttering recall; `hydrate` to restore them when relevant again.
"""


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Initialize YantrikDB and embedder on server start, clean up on shutdown."""
    from sentence_transformers import SentenceTransformer

    from yantrikdb import YantrikDB

    db_path = os.environ.get("YANTRIKDB_DB_PATH", str(Path.home() / ".yantrikdb" / "memory.db"))
    model_name = os.environ.get("YANTRIKDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dim = int(os.environ.get("YANTRIKDB_EMBEDDING_DIM", "384"))

    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log.info("Loading embedding model: %s", model_name)
    embedder = SentenceTransformer(model_name)
    log.info("Model loaded in %.1fs", time.time() - t0)

    log.info("Opening YantrikDB at: %s (dim=%d)", db_path, embedding_dim)
    db = YantrikDB(db_path=db_path, embedding_dim=embedding_dim, embedder=embedder)

    # Lock to serialize access to the unsendable PyYantrikDB
    lock = threading.Lock()

    log.info("YantrikDB MCP server ready (startup: %.1fs)", time.time() - t0)

    try:
        yield {"db": db, "lock": lock, "embedder": embedder}
    finally:
        log.info("Shutting down YantrikDB")
        db.close()


mcp = FastMCP("yantrikdb", instructions=INSTRUCTIONS, lifespan=lifespan)

# Import tools and resources so they register with the server
from . import resources as _resources  # noqa: F401, E402
from . import tools as _tools  # noqa: F401, E402
