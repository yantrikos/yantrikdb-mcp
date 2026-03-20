"""FastMCP server definition with YantrikDB lifespan context."""

import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Configure logging to stderr (stdout is reserved for MCP JSON-RPC)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("yantrikdb.mcp")


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

    log.info("Loading embedding model: %s", model_name)
    embedder = SentenceTransformer(model_name)

    log.info("Opening YantrikDB at: %s (dim=%d)", db_path, embedding_dim)
    db = YantrikDB(db_path=db_path, embedding_dim=embedding_dim, embedder=embedder)

    # Lock to serialize access to the unsendable PyYantrikDB
    lock = threading.Lock()

    try:
        yield {"db": db, "lock": lock}
    finally:
        log.info("Shutting down YantrikDB")
        db.close()


mcp = FastMCP("yantrikdb", lifespan=lifespan)

# Import tools and resources so they register with the server
from . import resources as _resources  # noqa: F401, E402
from . import tools as _tools  # noqa: F401, E402
