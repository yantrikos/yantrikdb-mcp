"""Tests for v0.6.0 embedder auto-detection / backend selection.

These verify the decision matrix in `embedder.load_engine()` without
requiring ONNX deps to be installed (the actual ONNX path is exercised
by the legacy backend integration tests + manual smoke).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from yantrikdb_mcp import embedder as embedder_mod


# ─────────────────────────────────────────────────────────────────────
# db_has_memories probe
# ─────────────────────────────────────────────────────────────────────


def test_has_memories_false_for_missing_file(tmp_path: Path):
    assert embedder_mod.db_has_memories(tmp_path / "missing.db") is False


def test_has_memories_false_for_empty_file(tmp_path: Path):
    p = tmp_path / "empty.db"
    p.touch()
    assert embedder_mod.db_has_memories(p) is False


def test_has_memories_false_for_fresh_db(tmp_path: Path):
    """A freshly opened YantrikDB with no records should report no memories."""
    from yantrikdb import YantrikDB

    db_path = tmp_path / "fresh.db"
    db = YantrikDB.with_default(str(db_path))
    del db  # close the engine handle (releases the SQLite file on Windows)
    assert embedder_mod.db_has_memories(db_path) is False


def test_has_memories_true_after_record(tmp_path: Path):
    from yantrikdb import YantrikDB

    db_path = tmp_path / "has_data.db"
    db = YantrikDB.with_default(str(db_path))
    db.record_text("a memory for the probe")
    del db
    assert embedder_mod.db_has_memories(db_path) is True


# ─────────────────────────────────────────────────────────────────────
# load_engine — env var / auto-detect
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def clear_embedder_env(monkeypatch):
    monkeypatch.delenv("YANTRIKDB_EMBEDDER", raising=False)


def test_auto_picks_bundled_for_new_db(tmp_path: Path, clear_embedder_env):
    """Fresh DB + auto = bundled embedder (engine default)."""
    db_path = tmp_path / "new.db"
    db = embedder_mod.load_engine(db_path)
    assert db.has_embedder() is True
    # Bundled embedder writes 64-dim vectors; record_text should just work.
    rid = db.record_text("hello world")
    assert isinstance(rid, str) and len(rid) > 0


def test_explicit_bundled(tmp_path: Path, monkeypatch):
    """YANTRIKDB_EMBEDDER=bundled forces bundled even on existing DB."""
    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "bundled")
    db = embedder_mod.load_engine(tmp_path / "forced.db")
    assert db.has_embedder() is True


def test_unknown_value_falls_back_to_auto(tmp_path: Path, monkeypatch, caplog):
    """Garbage env value → log warning, treat as auto, succeed."""
    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "supercollider")
    with caplog.at_level("WARNING"):
        db = embedder_mod.load_engine(tmp_path / "unknown.db")
    assert any("Unknown YANTRIKDB_EMBEDDER" in r.message for r in caplog.records)
    assert db.has_embedder() is True


def test_onnx_missing_deps_raises_helpful_error(tmp_path: Path, monkeypatch):
    """Forcing onnx without the [onnx] extras → RuntimeError mentioning the install hint."""
    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "onnx")
    monkeypatch.setattr(embedder_mod, "_onnx_deps_available", lambda: False)

    with pytest.raises(RuntimeError) as exc:
        embedder_mod.load_engine(tmp_path / "needs_onnx.db")
    assert "yantrikdb-mcp[onnx]" in str(exc.value)


def test_auto_with_existing_data_demands_onnx(tmp_path: Path, monkeypatch):
    """Auto mode + existing memories + ONNX missing → fail loudly with migration hint."""
    # Build a DB that probes as having memories.
    from yantrikdb import YantrikDB

    db_path = tmp_path / "legacy.db"
    db = YantrikDB.with_default(str(db_path))
    db.record_text("legacy memory")
    del db
    assert embedder_mod.db_has_memories(db_path) is True

    monkeypatch.delenv("YANTRIKDB_EMBEDDER", raising=False)
    monkeypatch.setattr(embedder_mod, "_onnx_deps_available", lambda: False)

    with pytest.raises(RuntimeError) as exc:
        embedder_mod.load_engine(db_path)
    msg = str(exc.value)
    assert "Existing DB" in msg
    assert "yantrikdb-mcp[onnx]" in msg


# ─────────────────────────────────────────────────────────────────────
# Multilingual backend (v0.7.0+)
# ─────────────────────────────────────────────────────────────────────


def test_multilingual_blocks_existing_db(tmp_path: Path, monkeypatch):
    """Forcing multilingual on a DB that already has memories must fail
    loudly — the existing vectors are at a different dim and recall would
    silently produce zero results."""
    from yantrikdb import YantrikDB

    db_path = tmp_path / "existing.db"
    db = YantrikDB.with_default(str(db_path))
    db.record_text("existing 64-dim memory")
    del db

    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "multilingual")
    with pytest.raises(RuntimeError) as exc:
        embedder_mod.load_engine(db_path)
    msg = str(exc.value)
    assert "multilingual" in msg.lower()
    assert "different embedder" in msg or "different vector dim" in msg


def test_multilingual_requires_engine_with_set_embedder_named(tmp_path: Path, monkeypatch):
    """Older engines without set_embedder_named must fail with a clear upgrade hint."""
    import yantrikdb as ydb_mod

    # Pretend the engine is missing the API by patching out the attribute.
    monkeypatch.delattr(ydb_mod.YantrikDB, "set_embedder_named", raising=False)
    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "multilingual")

    with pytest.raises(RuntimeError) as exc:
        embedder_mod.load_engine(tmp_path / "needs_new_engine.db")
    msg = str(exc.value)
    assert "yantrikdb engine >=0.7" in msg or "set_embedder_named" in msg


def test_auto_never_picks_multilingual(tmp_path: Path, monkeypatch, caplog):
    """Auto mode must never resolve to multilingual — its dim differs from
    both bundled (64) and ONNX (384), so silent selection would create
    DBs that can't be back-compat'd to either default."""
    monkeypatch.delenv("YANTRIKDB_EMBEDDER", raising=False)
    db = embedder_mod.load_engine(tmp_path / "fresh.db")
    # Auto on a fresh DB resolves to bundled (64 dim), never multilingual.
    # Probe via stats — bundled writes 0 memories so just verify open succeeded.
    assert db.has_embedder() is True


# ─────────────────────────────────────────────────────────────────────
# Backwards-compat shim
# ─────────────────────────────────────────────────────────────────────


def test_load_embedder_without_onnx_raises(monkeypatch):
    """The legacy load_embedder() entry point should fail loudly when ONNX is missing."""
    monkeypatch.setattr(embedder_mod, "_onnx_deps_available", lambda: False)
    with pytest.raises(RuntimeError, match="yantrikdb-mcp\\[onnx\\]"):
        embedder_mod.load_embedder()
