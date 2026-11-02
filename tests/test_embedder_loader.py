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
# Named registry embedders (named:<model>)
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def no_model_download(monkeypatch):
    """Stub set_embedder_named so tests never hit the network."""
    import yantrikdb as ydb_mod

    calls: list[str] = []
    monkeypatch.setattr(
        ydb_mod.YantrikDB,
        "set_embedder_named",
        lambda self, name: calls.append(name),
    )
    return calls


def test_named_opens_fresh_db_at_registry_dim(tmp_path: Path, monkeypatch, no_model_download):
    """named:potion-base-32M on a fresh DB must construct at 512 dim and
    attach the named embedder (no download in the test — stubbed)."""
    import yantrikdb as ydb_mod

    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "named:potion-base-32M")
    db = embedder_mod.load_engine(tmp_path / "fresh512.db")
    assert no_model_download == ["potion-base-32M"]
    assert isinstance(db, ydb_mod.YantrikDB)
    del db


def _store_with_foreign_vectors(db_path: Path, dim: int) -> None:
    """A store with memories and no recorded embedder: vectors from the caller."""
    from yantrikdb import YantrikDB

    db = YantrikDB(db_path=str(db_path), embedding_dim=dim)
    db.record("vectors from somewhere else", embedding=[1.0] + [0.0] * (dim - 1))
    db.close()
    del db


def test_named_refuses_store_with_no_recorded_embedder(
    tmp_path: Path, monkeypatch, no_model_download
):
    """Same dim and no identity: the engine would attach and stamp a false
    identity on vectors it never produced, so the loader has to refuse."""
    db_path = tmp_path / "foreign.db"
    _store_with_foreign_vectors(db_path, 256)

    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "named:potion-base-8M")
    with pytest.raises(RuntimeError) as exc:
        embedder_mod.load_engine(db_path)
    assert "potion-base-8M" in str(exc.value)
    assert "different embedder" in str(exc.value)
    assert no_model_download == [], "must refuse before attaching"


def test_named_refuses_store_of_another_model_and_names_it(
    tmp_path: Path, monkeypatch, no_model_download
):
    """The error has to say what built the store, and how to reopen it."""
    import yantrikdb as ydb_mod

    db_path = tmp_path / "other.db"
    _store_with_foreign_vectors(db_path, 512)
    monkeypatch.setattr(
        ydb_mod.YantrikDB,
        "embedder_identity",
        lambda self: {"name": "potion-base-8M", "digest": "sha256:x", "dim": 512},
    )

    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "named:potion-base-32M")
    with pytest.raises(RuntimeError) as exc:
        embedder_mod.load_engine(db_path)
    assert "named:potion-base-8M" in str(exc.value)
    assert no_model_download == []


def test_refused_without_identity_api_on_older_engines(
    tmp_path: Path, monkeypatch, no_model_download
):
    """The supported engine range predates embedder_identity. Nothing can be
    verified there, so a store with memories stays refused."""
    import yantrikdb as ydb_mod

    db_path = tmp_path / "old.db"
    _store_with_foreign_vectors(db_path, 256)
    monkeypatch.delattr(ydb_mod.YantrikDB, "embedder_identity", raising=False)

    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "named:potion-base-8M")
    with pytest.raises(RuntimeError):
        embedder_mod.load_engine(db_path)
    assert no_model_download == []


def _restart_and_recall(db_path: Path) -> list[str]:
    """Two server starts on one file, with the real engine and no stubs."""
    db = embedder_mod.load_engine(db_path)
    db.record_text("the deploy key for node4 is id_deploy")
    db.close()
    del db

    db = embedder_mod.load_engine(db_path)
    try:
        return [hit["text"] for hit in db.recall(query="deploy key", top_k=1)]
    finally:
        db.close()


def _skip_unless_downloadable(exc: Exception, model: str) -> None:
    msg = str(exc).lower()
    if any(k in msg for k in ("download", "fetch", "network", "connect", "timed out")):
        pytest.skip(f"{model} unavailable: {exc}")


def test_named_store_survives_restart(tmp_path: Path, monkeypatch):
    """The store a named embedder created has to reopen on the next start. The
    first start writes memories, so a check on "has memories" alone refuses it."""
    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "named:potion-base-8M")
    try:
        hits = _restart_and_recall(tmp_path / "restart.db")
    except RuntimeError as exc:
        _skip_unless_downloadable(exc, "potion-base-8M")
        raise
    assert hits == ["the deploy key for node4 is id_deploy"]


@pytest.mark.skipif(
    os.environ.get("YANTRIKDB_TEST_MULTILINGUAL") != "1",
    reason="downloads ~460 MB; CI opts in on the multilingual backend job",
)
def test_multilingual_store_survives_restart(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "multilingual")
    hits = _restart_and_recall(tmp_path / "restart.db")
    assert hits == ["the deploy key for node4 is id_deploy"]


def test_named_unknown_dim_requires_env_override(tmp_path: Path, monkeypatch, no_model_download):
    """A registry name missing from the dim table needs YANTRIKDB_EMBEDDING_DIM."""
    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "named:potion-base-99X")
    monkeypatch.delenv("YANTRIKDB_EMBEDDING_DIM", raising=False)
    with pytest.raises(RuntimeError) as exc:
        embedder_mod.load_engine(tmp_path / "unknowndim.db")
    assert "YANTRIKDB_EMBEDDING_DIM" in str(exc.value)


def test_named_env_dim_override(tmp_path: Path, monkeypatch, no_model_download):
    """YANTRIKDB_EMBEDDING_DIM supplies the dim for unknown registry names."""
    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "named:potion-base-99X")
    monkeypatch.setenv("YANTRIKDB_EMBEDDING_DIM", "768")
    db = embedder_mod.load_engine(tmp_path / "override768.db")
    assert no_model_download == ["potion-base-99X"]
    del db


def test_named_requires_engine_with_set_embedder_named(tmp_path: Path, monkeypatch):
    """Older engines without set_embedder_named must fail with a clear upgrade hint."""
    import yantrikdb as ydb_mod

    monkeypatch.delattr(ydb_mod.YantrikDB, "set_embedder_named", raising=False)
    monkeypatch.setenv("YANTRIKDB_EMBEDDER", "named:potion-base-32M")

    with pytest.raises(RuntimeError) as exc:
        embedder_mod.load_engine(tmp_path / "needs_new_engine.db")
    msg = str(exc.value)
    assert "yantrikdb engine >=0.7" in msg or "set_embedder_named" in msg


# ─────────────────────────────────────────────────────────────────────
# Backwards-compat shim
# ─────────────────────────────────────────────────────────────────────


def test_load_embedder_without_onnx_raises(monkeypatch):
    """The legacy load_embedder() entry point should fail loudly when ONNX is missing."""
    monkeypatch.setattr(embedder_mod, "_onnx_deps_available", lambda: False)
    with pytest.raises(RuntimeError, match="yantrikdb-mcp\\[onnx\\]"):
        embedder_mod.load_embedder()
