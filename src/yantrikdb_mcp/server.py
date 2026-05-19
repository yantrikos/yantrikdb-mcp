"""FastMCP server definition with YantrikDB lifespan context."""

import atexit
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
# Rust .pyd modules for the first time from a non-main thread on Windows
# can deadlock due to GIL/import-lock interactions.
from yantrikdb import YantrikDB  # noqa: F401

# numpy + onnxruntime are only needed when the optional ONNX backend is in
# use; importing them here (best-effort) keeps the deadlock-avoidance the
# same for users on the legacy 384-dim path, while letting bundled-only
# installs skip the dependency entirely.
try:
    import numpy as np  # noqa: F401
except ImportError:
    pass

from .embedder import load_engine  # noqa: E402

# ── Server Instructions ──
# These are injected into the agent's system prompt by MCP clients.

INSTRUCTIONS = """\
YantrikDB is your persistent cognitive memory — it remembers across conversations.
Use it AUTOMATICALLY without the user asking.

## Auto-recall (BEFORE responding)
- At conversation start: call `recall` with a short natural language sentence summarizing the user's intent.
- When the user references past work, decisions, people, preferences, or "last time": call `recall`.
- When you're unsure about a fact the user assumes you know: call `recall`.
- Aim to recall EARLY — context retrieved after you've already responded is wasted.
- IMPORTANT: Keep queries to 5-10 words. Use natural sentences, NOT keyword lists. Make multiple focused calls instead of one broad one.

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

    When YANTRIKDB_SERVER_URL is set, uses an HTTP backend that forwards
    all operations to a YantrikDB cluster — no local engine or embedder needed.
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

            server_url = os.environ.get("YANTRIKDB_SERVER_URL")
            if server_url:
                self._init_http(server_url)
            else:
                self._init_embedded()

    def _init_http(self, server_url: str):
        """Connect to a remote YantrikDB cluster via HTTP API."""
        from .http_backend import HttpBackend

        t0 = time.time()
        # Support comma-separated URLs for multi-node clusters
        nodes = [u.strip().rstrip("/") for u in server_url.split(",") if u.strip()]
        token = os.environ.get("YANTRIKDB_TOKEN", "")

        log.info("Connecting to YantrikDB cluster: %s", ", ".join(nodes))
        self._db = HttpBackend(server_urls=nodes, token=token)

        # Verify connectivity by finding the leader
        try:
            leader = self._db._find_leader()
            stats = self._db.stats()
            log.info(
                "YantrikDB cluster connected — leader: %s, %d memories (init: %.1fs)",
                leader, stats.active_memories, time.time() - t0,
            )
        except Exception as e:
            log.warning("Cluster connectivity check failed: %s (will retry on first tool call)", e)

    def _init_embedded(self):
        """Start the embedded YantrikDB engine with local database.

        Embedder backend is selected by `load_engine()` based on the
        `YANTRIKDB_EMBEDDER` env var and whether the DB already has data.
        See `embedder.load_engine` for the full decision matrix.
        """
        db_path = os.environ.get("YANTRIKDB_DB_PATH", str(Path.home() / ".yantrikdb" / "memory.db"))
        model_name = os.environ.get("YANTRIKDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        log.info("Opening YantrikDB at: %s", db_path)
        t0 = time.time()
        self._db = load_engine(db_path, model_name=model_name)
        log.info("YantrikDB ready (total init: %.1fs)", time.time() - t0)

    @property
    def db(self):
        self._ensure_init()
        return self._db

    def close(self):
        """Idempotent — safe to call multiple times (atexit + any other
        cleanup path). After the first call, `_db` is nulled so a second
        call is a no-op and there's no double-close on the SQLite handle.
        Also tolerates engines that don't expose a `close()` method
        (HTTP backend) by swallowing AttributeError."""
        with self._init_lock:
            db, self._db = self._db, None
        if db is None:
            return
        try:
            db.close()
        except AttributeError:
            # HTTP backend doesn't need explicit close; engine handle
            # has no close on older variants either.
            pass


_safety_warnings_emitted = False
_safety_warnings_lock = threading.Lock()


def _emit_skill_safety_warnings() -> None:
    """F — log startup warnings about dangerous skill-substrate
    configurations. Emitted once per process lifetime — under SSE
    transport the lifespan may be entered multiple times (per-session)
    and we don't want to spam the journal or pollute the audit log
    with duplicate startup-warning events on every reconnect."""
    global _safety_warnings_emitted
    from .skill_security import audit_event, config, startup_safety_checks

    with _safety_warnings_lock:
        if _safety_warnings_emitted:
            return
        _safety_warnings_emitted = True

    is_cluster = bool(os.environ.get("YANTRIKDB_SERVER_URL", "").strip())
    # We can't easily probe actor_ids before DB open — pass None and
    # leave the F.1 multi-tenant check to the operator's audit-log
    # review. Cluster mode + gate-on is the most important warning.
    warnings = startup_safety_checks(is_cluster_mode=is_cluster, db_actor_ids=None)
    for w in warnings:
        log.warning(w)
    if warnings:
        audit_event({
            "event": "startup_safety_warnings",
            "warnings": warnings,
            "config_snapshot": config().snapshot(),
        })


# Process-singleton `_LazyDB`. Under SSE transport, FastMCP/uvicorn
# enters the `lifespan` context once per client session. The v0.8.1
# implementation constructed a fresh `_LazyDB` on every entry and
# called `close()` on every exit — when sessions overlap (rapid
# reconnects, kill mid-drain, macOS sleep/wake), multiple instances
# raced to checkpoint+close the same SQLite WAL, partially overwriting
# the main DB on any interrupted close. Pinned as a singleton here per
# yantrikos/yantrikdb-mcp#11. The Rust engine is internally
# Mutex/RwLock-protected (per the docstring on `_LazyDB`), so one
# shared instance across sessions is the *intended* threading model
# — we're just making the implementation match.
_lazy_singleton: "_LazyDB | None" = None
_lazy_singleton_lock = threading.Lock()


def _get_lazy_singleton() -> "_LazyDB":
    global _lazy_singleton
    if _lazy_singleton is None:
        with _lazy_singleton_lock:
            if _lazy_singleton is None:
                _lazy_singleton = _LazyDB()
                # Register a single orderly close at interpreter exit.
                # We deliberately do NOT register signal handlers — they
                # interact badly with uvicorn's own SIGTERM handling.
                # `atexit` runs once after Python's normal shutdown which
                # is the right hook for "the only owner of the SQLite
                # handle is closing now."
                atexit.register(_lazy_singleton.close)
    return _lazy_singleton


@asynccontextmanager
async def lifespan(app: FastMCP):
    """Provide a process-singleton YantrikDB context.

    Under stdio transport this runs once. Under SSE transport FastMCP
    may re-enter this for each new client session — that's fine, every
    session yields the same `_LazyDB` instance. Close happens once via
    `atexit`, not per-session. See yantrikos/yantrikdb-mcp#11.
    """
    lazy = _get_lazy_singleton()
    log.info("YantrikDB MCP server started (model loads on first use)")
    _emit_skill_safety_warnings()
    try:
        yield {"lazy": lazy}
    finally:
        # No per-session close. The singleton lives for the lifetime of
        # the process and is closed exactly once by the atexit hook.
        pass


mcp = FastMCP("yantrikdb", instructions=INSTRUCTIONS, lifespan=lifespan)

# Import tools and resources so they register with the server
from . import resources as _resources  # noqa: F401, E402
from . import tools as _tools  # noqa: F401, E402
