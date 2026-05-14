"""Embedder + engine loader for YantrikDB MCP server.

v0.7.0+ supports three backends:

1. **Bundled** (default for new English installs) — engine ships its own
   potion-base-2M Rust embedder (64 dim). ~80 ms cold start, no extra
   Python deps. Use via `YantrikDB.with_default(db_path)`.

2. **ONNX MiniLM-L6-v2** (legacy / opt-in) — 384-dim, higher English
   recall quality. Requires `pip install yantrikdb-mcp[onnx]`.

3. **Multilingual** (opt-in, v0.7.0+) — `potion-multilingual-128M` from
   the engine's downloadable registry. 256-dim, 101 languages. Engine
   downloads the model on first use; no extra Python deps. Requires
   yantrikdb engine v0.7.9+.

Selection is automatic ("auto" — the default):
  - Existing DB with memories → ONNX 384 (back-compat with prior installs).
  - New / empty DB → bundled 64 (fast, lean).
  - Multilingual is opt-in only — never auto-selected (different dim,
    incompatible with existing DBs).

Override with `YANTRIKDB_EMBEDDER=auto|bundled|onnx|multilingual`.
"""

import logging
import os
import sqlite3
import time
from pathlib import Path

log = logging.getLogger("yantrikdb.mcp")


# ─────────────────────────────────────────────────────────────────────
# ONNX MiniLM-L6-v2 embedder (384-dim, legacy/opt-in)
# ─────────────────────────────────────────────────────────────────────


class OnnxEmbedder:
    """Minimal ONNX embedder that implements the YantrikDB Embedder interface."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self._np = np
        t0 = time.time()
        # Try local cache first (no network), fall back to download
        try:
            model_path = hf_hub_download(model_name, "onnx/model.onnx", local_files_only=True)
            tokenizer_path = hf_hub_download(model_name, "tokenizer.json", local_files_only=True)
        except Exception:
            model_path = hf_hub_download(model_name, "onnx/model.onnx")
            tokenizer_path = hf_hub_download(model_name, "tokenizer.json")

        self._session = ort.InferenceSession(model_path)
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=256)
        self._dim = 384
        log.info("ONNX MiniLM-L6-v2 embedder loaded in %.1fs (dim=384)", time.time() - t0)

    def encode(self, text: str) -> list[float]:
        """Alias for compatibility with sentence-transformers API."""
        return self.embed(text)

    def embed(self, text: str) -> list[float]:
        np = self._np
        enc = self._tokenizer.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids], dtype=np.int64)

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (token_embeddings * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / norm

        return pooled[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        np = self._np
        encodings = self._tokenizer.encode_batch(texts)
        max_len = max(len(e.ids) for e in encodings)

        input_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(texts), max_len), dtype=np.int64)
        token_type_ids = np.zeros((len(texts), max_len), dtype=np.int64)

        for i, enc in enumerate(encodings):
            length = len(enc.ids)
            input_ids[i, :length] = enc.ids
            attention_mask[i, :length] = enc.attention_mask
            token_type_ids[i, :length] = enc.type_ids

        outputs = self._session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        token_embeddings = outputs[0]
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (token_embeddings * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / norms

        return pooled.tolist()

    def dim(self) -> int:
        return self._dim


# ─────────────────────────────────────────────────────────────────────
# Auto-detect helpers
# ─────────────────────────────────────────────────────────────────────


def db_has_memories(db_path: str | os.PathLike) -> bool:
    """Probe a YantrikDB SQLite file for existing active memories.

    Returns False for missing/empty/locked files — anything that fails
    is treated as "no prior data", so we default to the lean path.
    """
    p = Path(db_path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2.0)
        try:
            cur = conn.cursor()
            # Try a few likely shapes — schema has evolved; we only need a yes/no.
            for q in (
                "SELECT 1 FROM memories WHERE tombstoned = 0 LIMIT 1",
                "SELECT 1 FROM memories LIMIT 1",
            ):
                try:
                    row = cur.execute(q).fetchone()
                    if row:
                        return True
                except sqlite3.OperationalError:
                    continue
            return False
        finally:
            conn.close()
    except Exception as e:
        log.debug("db_has_memories probe failed for %s: %s", p, e)
        return False


def _onnx_deps_available() -> bool:
    try:
        import huggingface_hub  # noqa: F401
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────
# Engine loader (the v0.5.0 entry point)
# ─────────────────────────────────────────────────────────────────────


# Engine registry name for the multilingual backend (engine v0.7.9+).
# Held as a constant so tests can monkeypatch the dim/name without touching
# load_engine internals.
MULTILINGUAL_MODEL = "potion-multilingual-128M"
MULTILINGUAL_DIM = 256


def load_engine(db_path: str | os.PathLike, model_name: str = "all-MiniLM-L6-v2"):
    """Open a YantrikDB engine with the right embedder for this install.

    Resolves backend in this order:

    1. `YANTRIKDB_EMBEDDER` env var: "auto" (default), "bundled", "onnx",
       or "multilingual".
    2. In auto mode: existing DB with memories → ONNX 384;
       new/empty DB → engine bundled 64-dim.
       Multilingual is opt-in only — never auto-selected because its
       256-dim vectors are incompatible with existing bundled (64-dim) or
       ONNX (384-dim) databases.

    Returns the ready-to-use `YantrikDB` instance.
    """
    from yantrikdb import YantrikDB

    choice = os.environ.get("YANTRIKDB_EMBEDDER", "auto").strip().lower()
    if choice not in ("auto", "bundled", "onnx", "multilingual"):
        log.warning("Unknown YANTRIKDB_EMBEDDER=%r, treating as 'auto'", choice)
        choice = "auto"

    has_data = db_has_memories(db_path)

    # Resolve auto → bundled / onnx
    if choice == "auto":
        if has_data:
            choice = "onnx"
            log.info("Existing DB with memories detected → ONNX MiniLM (384 dim) for back-compat")
        else:
            choice = "bundled"
            log.info("New/empty DB → engine bundled embedder (64 dim, fast cold start)")

    # Bundled: engine handles everything
    if choice == "bundled":
        if not hasattr(YantrikDB, "with_default"):
            # Engine too old — fall through to ONNX (if available) or fail loudly.
            log.warning(
                "yantrikdb engine too old for bundled embedder (need >=0.7.8). "
                "Falling back to ONNX."
            )
            choice = "onnx"
        else:
            t0 = time.time()
            db = YantrikDB.with_default(str(db_path))
            log.info("YantrikDB opened with bundled embedder in %.2fs", time.time() - t0)
            return db

    # Multilingual: engine downloads the model on first use, no Python deps.
    if choice == "multilingual":
        if not hasattr(YantrikDB, "with_default") or not hasattr(YantrikDB, "set_embedder_named"):
            raise RuntimeError(
                "Multilingual embedder requires yantrikdb engine >=0.7.9 "
                "(with set_embedder_named + with_default). "
                "Run: pip install --upgrade 'yantrikdb-mcp[onnx]' to refresh "
                "the engine, or unset YANTRIKDB_EMBEDDER to fall back to auto."
            )
        # Opening an existing non-multilingual DB with this backend would
        # silently produce zero-recall results because the stored vectors
        # are at a different dim. Block it early with a helpful error.
        if has_data:
            raise RuntimeError(
                f"YANTRIKDB_EMBEDDER=multilingual requested but the existing "
                f"database at {db_path} contains memories from a different "
                f"embedder (different vector dim).\n"
                f"Multilingual ({MULTILINGUAL_DIM} dim) is only valid for "
                f"fresh databases. Either point YANTRIKDB_DB_PATH at a new "
                f"file, or unset YANTRIKDB_EMBEDDER to use the back-compat "
                f"backend for this DB."
            )
        t0 = time.time()
        # Open at the multilingual model's dim explicitly — `with_default()`
        # would pin dim=64 from the bundled embedder, and the engine
        # rejects `set_embedder_named` for a mismatched dim. Constructing
        # bare YantrikDB at dim=256 then attaching the named embedder is
        # the supported swap path.
        db = YantrikDB(db_path=str(db_path), embedding_dim=MULTILINGUAL_DIM)
        db.set_embedder_named(MULTILINGUAL_MODEL)
        log.info(
            "YantrikDB opened with multilingual embedder (%s, %d dim) in %.2fs",
            MULTILINGUAL_MODEL, MULTILINGUAL_DIM, time.time() - t0,
        )
        return db

    # ONNX path
    if choice == "onnx":
        if not _onnx_deps_available():
            msg = (
                "ONNX embedder requested but optional deps not installed.\n"
                "  Install with:  pip install 'yantrikdb-mcp[onnx]'\n"
                "  Or set YANTRIKDB_EMBEDDER=bundled to use the engine's "
                "built-in embedder (64 dim — note: existing 384-dim DBs will "
                "not recall properly with the bundled embedder)."
            )
            if has_data:
                raise RuntimeError(
                    "Existing DB has memories embedded with the 384-dim ONNX "
                    "model, but ONNX deps are missing.\n\n" + msg
                )
            raise RuntimeError(msg)

        hf_name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
        embedder = OnnxEmbedder(hf_name)
        embedding_dim = embedder.dim()
        t0 = time.time()
        db = YantrikDB(db_path=str(db_path), embedding_dim=embedding_dim, embedder=embedder)
        log.info(
            "YantrikDB opened with ONNX embedder (dim=%d) in %.2fs",
            embedding_dim, time.time() - t0,
        )
        return db

    raise RuntimeError(f"Unreachable: choice={choice!r}")


# ─────────────────────────────────────────────────────────────────────
# Backwards compatibility — keep load_embedder for external imports
# ─────────────────────────────────────────────────────────────────────


def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Deprecated since v0.5.0 — prefer `load_engine()`.

    Still works: returns an `OnnxEmbedder` (or raises if ONNX deps missing).
    """
    if not _onnx_deps_available():
        raise RuntimeError(
            "load_embedder() requires ONNX deps. "
            "Install with: pip install 'yantrikdb-mcp[onnx]'"
        )
    hf_name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return OnnxEmbedder(hf_name)
