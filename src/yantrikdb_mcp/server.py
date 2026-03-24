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

# Pre-import C extensions in the main thread to avoid deadlocks when
# FastMCP dispatches sync tool handlers to worker threads.  Importing
# numpy, onnxruntime, or Rust .pyd modules for the first time from a
# non-main thread on Windows can deadlock due to GIL/import-lock
# interactions.
import numpy as np  # noqa: F401
from yantrikdb import YantrikDB  # noqa: F401

from .embedder import load_embedder  # noqa: E402

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
Call `graph` with action="relate" when you learn about entity relationships:
- "Alice works at Acme" → graph(action="relate", entity="Alice", target="Acme", relationship="works_at")
- "Project X uses React" → graph(action="relate", entity="Project X", target="React", relationship="uses")
- "Bob reports to Alice" → graph(action="relate", entity="Bob", target="Alice", relationship="reports_to")

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

## Auto-learn procedures
When you discover a working approach (deployment steps, debugging strategy, review process):
- Call `procedure(action="learn", text="...", domain="work")` to save it.
- Before starting a task, call `procedure(action="surface", query="how to ...")` to check for known strategies.
- After using a procedure, call `procedure(action="reinforce", rid="...", outcome=0.9)` to improve ranking.

## Proactive time checks
- At conversation start: call `temporal(action="upcoming", days=7)` to surface approaching deadlines.
- During maintenance: call `temporal(action="stale", days=30)` to find neglected high-importance memories.

## Cognitive maintenance
- Call `think` at the end of long conversations to consolidate and detect conflicts.
- If `think` surfaces conflicts, use `conflict(action="resolve", ...)` or ask the user.
- If recall returns low-confidence results, use `recall(query="...", refine_from="original query")`.
- After `think`, call `patterns` to check for cross-domain discoveries and entity bridges.
- Call `personality(recompute=True)` after big sessions to refresh trait scores.
- Use `trigger(action="acknowledge", ...)` after surfacing triggers to the user.
- Use `memory(action="archive", ...)` for old memories cluttering recall; `memory(action="hydrate", ...)` to restore.
"""


class _LazyDB:
    """Defers SentenceTransformer + YantrikDB init until first tool call.

    The Rust engine uses internal Mutex/RwLock for thread safety, so no
    Python-level lock is needed around individual operations.
    """

    def __init__(self):
        self._db = None
        self._init_lock = threading.Lock()

    def _ensure_init(self):
        if self._db is not None:
            return
        with self._init_lock:
            if self._db is not None:
                return  # another thread beat us

            db_path = os.environ.get("YANTRIKDB_DB_PATH", str(Path.home() / ".yantrikdb" / "memory.db"))
            model_name = os.environ.get("YANTRIKDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            embedding_dim = int(os.environ.get("YANTRIKDB_EMBEDDING_DIM", "384"))
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            embedder = load_embedder(model_name)
            embedding_dim = embedder.dim() if hasattr(embedder, 'dim') and callable(embedder.dim) else embedding_dim

            log.info("Opening YantrikDB at: %s (dim=%d)", db_path, embedding_dim)
            self._db = YantrikDB(db_path=db_path, embedding_dim=embedding_dim, embedder=embedder)
            log.info("YantrikDB ready (init: %.1fs)", time.time() - t0)

    @property
    def db(self):
        self._ensure_init()
        return self._db

    def close(self):
        if self._db is not None:
            self._db.close()


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Provide a lazy-initialized YantrikDB context — model loads on first tool call."""
    lazy = _LazyDB()
    log.info("YantrikDB MCP server started (model loads on first use)")
    try:
        yield {"lazy": lazy}
    finally:
        log.info("Shutting down YantrikDB")
        lazy.close()


mcp = FastMCP("yantrikdb", instructions=INSTRUCTIONS, lifespan=lifespan)

# Import tools and resources so they register with the server
from . import resources as _resources  # noqa: F401, E402
from . import tools as _tools  # noqa: F401, E402
