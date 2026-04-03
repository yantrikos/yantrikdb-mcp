"""Smoke tests for the console entrypoint (subprocess — real install layout)."""

import subprocess
import sys


def test_version_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "yantrikdb_mcp", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "yantrikdb-mcp" in proc.stdout


def test_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "yantrikdb_mcp", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + proc.stderr
    assert "--transport" in out
    assert "YANTRIKDB_DB_PATH" in out
