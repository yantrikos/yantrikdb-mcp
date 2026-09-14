"""`atlas(...)` — export this store's Memory Atlas and serve it on localhost.

The tool runs the engine package's exporter as a CHILD process (a second
SQLite library may never open the store inside the engine's own process)
and then serves the output directory. These tests exercise the tool's own
contract — refusals that teach, the served page, the failure path — with
the exporter stubbed, because the exporter belongs to the engine package
and is tested there.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from yantrikdb_mcp._compat import ToolError
from yantrikdb_mcp import tools
from yantrikdb_mcp.tools import atlas


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fake embedded store path in the env, no cluster URL."""
    db = tmp_path / "memory.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 64)
    monkeypatch.delenv("YANTRIKDB_SERVER_URL", raising=False)
    monkeypatch.setenv("YANTRIKDB_DB_PATH", str(db))
    monkeypatch.setattr(tools, "_atlas_exporter_path", lambda: Path("/fake/yantrikdb/atlas/export_atlas.py"))
    monkeypatch.setattr(tools, "_atlas_serve_available", lambda: True)
    return db


# Serving goes through the engine package's shared allowlist server; the
# tests that exercise it need an engine that ships `yantrikdb.atlas.serve`.
_serve = pytest.importorskip("yantrikdb.atlas.serve", reason="engine lacks yantrikdb.atlas.serve")


def _fake_exporter(out_dir_holder):
    """Stand-in for the exporter script: writes what the real one
    writes (page, data, report) into the --out directory it was given."""
    def run(cmd):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "index.html").write_text("<title>YantrikDB · Memory Atlas</title>", encoding="utf-8")
        (out / "data.json").write_text(json.dumps({"format_version": 2, "memories": [{"id": 0}]}), encoding="utf-8")
        (out / "export-report.json").write_text(json.dumps({"memories": 1, "claims": 0}), encoding="utf-8")
        out_dir_holder.append(out)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
    return run


def test_refuses_in_cluster_mode_and_names_the_alternative(monkeypatch):
    monkeypatch.setenv("YANTRIKDB_SERVER_URL", "http://10.0.0.1:7438")
    with pytest.raises(ToolError) as e:
        atlas(ctx=object())
    assert "embedded" in str(e.value) and "yantrikdb atlas" in str(e.value)


def test_refuses_when_the_store_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("YANTRIKDB_SERVER_URL", raising=False)
    monkeypatch.setenv("YANTRIKDB_DB_PATH", str(tmp_path / "nope.db"))
    with pytest.raises(ToolError, match="nope.db"):
        atlas(ctx=object())


def test_refuses_when_the_engine_has_no_exporter(store, monkeypatch):
    monkeypatch.setattr(tools, "_atlas_exporter_path", lambda: None)
    with pytest.raises(ToolError) as e:
        atlas(ctx=object())
    assert "yantrikdb/atlas/" in str(e.value), "the refusal names what the engine must ship"


def test_refuses_an_out_dir_that_is_the_store_directory(store, tmp_path, monkeypatch):
    """Serving that directory would expose the store itself (review P1)."""
    monkeypatch.setattr(tools, "_run_exporter", _fake_exporter([]))
    with pytest.raises(ToolError, match="store's own directory"):
        atlas(out_dir=str(store.parent), ctx=object())


def test_served_directory_hands_out_only_the_artifacts(store, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_run_exporter", _fake_exporter([]))
    out = json.loads(atlas(out_dir=str(tmp_path / "atlas"), ctx=object()))
    (tmp_path / "atlas" / "secret.txt").write_text("never", encoding="utf-8")
    import urllib.error
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(out["url"] + "secret.txt", timeout=5)
    assert e.value.code == 404


def test_export_serves_the_page_and_returns_its_url(store, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "_run_exporter", _fake_exporter(seen))
    out = json.loads(atlas(out_dir=str(tmp_path / "atlas"), ctx=object()))

    assert out["url"].startswith("http://127.0.0.1:"), out
    assert out["store"] == str(store)
    assert out["report"] == {"memories": 1, "claims": 0}
    # The exporter was pointed at THIS store and THIS out dir.
    assert seen == [tmp_path / "atlas"]

    # Test the use: the served page and its data are reachable.
    with urllib.request.urlopen(out["url"], timeout=5) as r:
        assert "Memory Atlas" in r.read().decode("utf-8")
    with urllib.request.urlopen(out["url"].rstrip("/") + "/data.json", timeout=5) as r:
        assert json.load(r)["format_version"] == 2


def test_exporter_is_run_as_a_script_on_the_named_store_only(store, tmp_path, monkeypatch):
    """By absolute script path (never `-m`, so the child never imports the
    engine package) and with the store FILE, never its directory."""
    cmds = []
    def spy(cmd):
        cmds.append(cmd)
        return _fake_exporter([])(cmd)
    monkeypatch.setattr(tools, "_run_exporter", spy)
    atlas(out_dir=str(tmp_path / "atlas"), ctx=object())
    cmd = cmds[0]
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("export_atlas.py") and "-m" not in cmd
    assert cmd[cmd.index("--stores") + 1] == str(store)


def test_second_export_to_the_same_dir_reuses_the_server(store, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_run_exporter", _fake_exporter([]))
    a = json.loads(atlas(out_dir=str(tmp_path / "atlas"), ctx=object()))
    b = json.loads(atlas(out_dir=str(tmp_path / "atlas"), ctx=object()))
    assert a["url"] == b["url"], "one server per directory; re-export refreshes the files"


def test_exporter_failure_surfaces_its_stderr(store, tmp_path, monkeypatch):
    def failing(cmd):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Source changed during export: memory.db")
    monkeypatch.setattr(tools, "_run_exporter", failing)
    with pytest.raises(ToolError, match="Source changed during export"):
        atlas(out_dir=str(tmp_path / "atlas"), ctx=object())


def test_status_lists_running_servers(store, tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "_run_exporter", _fake_exporter([]))
    a = json.loads(atlas(out_dir=str(tmp_path / "atlas"), ctx=object()))
    status = json.loads(atlas(action="status", ctx=object()))
    assert any(s["url"] == a["url"] for s in status["servers"])


def test_rejects_unknown_action(store):
    with pytest.raises(ToolError, match="export"):
        atlas(action="sideways", ctx=object())
