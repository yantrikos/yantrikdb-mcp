"""`temporal(action="as_of", ...)` — the MCP surface for time-travel recall.

The engine's `recall_as_of` takes a unix float and NOTHING else; an ISO string
raises a bare `TypeError: argument 'as_of': must be real number, not str`. An
agent asked "what did we know on 2026-08-01" will reach for the date string
every single time, so this layer parses the forms an agent actually produces
and, on failure, returns an error that TEACHES the grammar instead of leaking
a type error the model cannot act on.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from yantrikdb_mcp._compat import ToolError
from yantrikdb_mcp.tools import _parse_as_of, temporal


# ── the parser ───────────────────────────────────────────────────────


def test_parses_iso_date_as_utc_midnight():
    ts = _parse_as_of("2026-08-01")
    assert datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") == "2026-08-01"


@pytest.mark.parametrize("s", ["2026-08-01T14:30:00Z", "2026-08-01T14:30:00+00:00",
                               "2026-08-01T14:30:00"])
def test_parses_iso_datetimes_including_bare_Z(s):
    """`fromisoformat` rejects a trailing Z on older Pythons, and a NAIVE
    stamp must be read as UTC — server-local time would make the same query
    mean different things on different hosts."""
    dt = datetime.fromtimestamp(_parse_as_of(s), timezone.utc)
    assert (dt.year, dt.month, dt.day, dt.hour) == (2026, 8, 1, 14)


@pytest.mark.parametrize("rel,secs", [("7d", 604800), ("24h", 86400),
                                      ("30m", 1800), ("90s", 90)])
def test_relative_ages_look_backwards(rel, secs):
    """"7d" means seven days AGO, not seven days from the epoch."""
    assert abs((time.time() - secs) - _parse_as_of(rel)) < 5


def test_accepts_unix_number_and_numeric_string():
    assert _parse_as_of(1785898500) == 1785898500.0
    assert _parse_as_of("1785898500") == 1785898500.0


def test_unparseable_input_names_every_accepted_form():
    """A rejection must teach the grammar — otherwise the agent's next attempt
    is another guess."""
    with pytest.raises(ToolError) as e:
        _parse_as_of("last tuesday")
    msg = str(e.value)
    for hint in ("2026-08-01", "unix", "7d"):
        assert hint in msg, f"error should mention {hint!r}; got: {msg}"


# ── the tool surface ─────────────────────────────────────────────────


class _Ctx:
    """Minimal stand-in for the MCP request context."""

    def __init__(self, db):
        self.request_context = type("R", (), {"lifespan_context": {"lazy": type("L", (), {"db": db})()}})()


@pytest.fixture
def ctx(tmp_path):
    import os

    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    db = load_engine(str(tmp_path / "asof.db"), model_name="bundled")
    yield _Ctx(db)
    try:
        db.close()
    except AttributeError:
        pass


def test_as_of_requires_query_and_as_of(ctx):
    with pytest.raises(ToolError, match="query"):
        temporal(action="as_of", as_of="2026-08-01", ctx=ctx)
    with pytest.raises(ToolError, match="as_of"):
        temporal(action="as_of", query="anything", ctx=ctx)


def test_rejects_unknown_action(ctx):
    with pytest.raises(ToolError, match="as_of"):
        temporal(action="sideways", ctx=ctx)


@pytest.mark.skipif(
    not hasattr(__import__("yantrikdb").YantrikDB, "recall_as_of"),
    reason="engine lacks recall_as_of — pre-v0.12",
)
def test_as_of_end_to_end_excludes_later_writes(ctx):
    db = ctx.request_context.lifespan_context["lazy"].db
    db.record("Tool-level probe: port is 8420", namespace="tl")
    time.sleep(1.1)
    cut = time.time()
    time.sleep(1.1)
    db.record("Tool-level probe: port is 9000", namespace="tl")

    out = json.loads(temporal(action="as_of", query="tool-level probe port",
                              as_of=str(cut), namespace="tl", ctx=ctx))
    blob = json.dumps(out["results"])
    assert "8420" in blob and "9000" not in blob, (
        "the tool must not surface memories recorded after as_of"
    )
    # The response states its own filtering so an empty/short result is not
    # misread as "nothing was ever known".
    assert "as_of" in out and "note" in out
    assert out["count"] == len(out["results"])
