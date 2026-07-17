"""HTTP-backend wiring tests for the server v0.8.27 read endpoints:
GET /v1/current (chain_head), /v1/session/digest, /v1/insights/gaps.

Fully offline — the requests.Session is replaced with a Mock, so no
network, no running cluster, no LLM. Asserts the request the backend
issues (path + params) AND the shape it hands back to the tool layer,
which is what would silently drift if the gateway contract changed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SRC = Path(__file__).parent.parent / "src" / "yantrikdb_mcp"
HTTP_BACKEND_PATH = SRC / "http_backend.py"


def _import_http_backend():
    spec = importlib.util.spec_from_file_location(
        "yantrikdb_mcp.http_backend", HTTP_BACKEND_PATH
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"http_backend.py has unimport-able dep: {e}")
    return module


def _fake_response(status_code: int, json_body=None):
    """A stand-in for requests.Response with just the surface _get /
    _get_optional touch: .status_code, .ok, .json(), .raise_for_status()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    resp.content = b"{}" if json_body is not None else b""
    resp.json.return_value = json_body if json_body is not None else {}

    def _raise():
        if status_code >= 400:
            import requests
            raise requests.HTTPError(f"{status_code}")

    resp.raise_for_status.side_effect = _raise
    return resp


def _backend_with_response(hb_module, response):
    """Build an HttpBackend whose leader is pre-set (so no discovery) and
    whose session.get returns `response`."""
    backend = hb_module.HttpBackend(server_urls=["http://node0:7438"], token="t")
    backend._leader = "http://node0:7438"
    backend._session = MagicMock()
    backend._session.get.return_value = response
    return backend


# ── GET /v1/current  (chain_head) ────────────────────────────────────


def test_chain_head_returns_record_on_found():
    hb = _import_http_backend()
    record = {
        "rid": "019f-abc", "text": "current value", "memory_type": "semantic",
        "importance": 0.9, "namespace": "proj.state", "certainty": 0.8,
        "domain": "work", "source": "user",
    }
    backend = _backend_with_response(
        hb, _fake_response(200, {"found": True, "namespace": "proj.state",
                                 "record": record})
    )

    head = backend.chain_head("proj.state")

    assert head == record, "chain_head must return the bare record dict (embedded parity)"
    # It hit GET /v1/current with the namespace param.
    call = backend._session.get.call_args
    assert call.args[0].endswith("/v1/current")
    assert call.kwargs["params"] == {"namespace": "proj.state"}


def test_chain_head_returns_none_on_404_empty_chain():
    hb = _import_http_backend()
    backend = _backend_with_response(
        hb, _fake_response(404, {"error": {"code": "generic",
                                           "message": "no chain head for namespace 'empty'"}})
    )

    head = backend.chain_head("empty")

    assert head is None, "an empty/unknown chain (404) is an EXPECTED None, not an error"


def test_chain_head_returns_none_when_found_false():
    hb = _import_http_backend()
    backend = _backend_with_response(hb, _fake_response(200, {"found": False}))
    assert backend.chain_head("ns") is None


def test_chain_head_requires_namespace():
    hb = _import_http_backend()
    backend = _backend_with_response(hb, _fake_response(200, {"found": False}))
    with pytest.raises(ValueError):
        backend.chain_head("")
    with pytest.raises(ValueError):
        backend.chain_head("   ")


# ── _get_optional: 404→None, but 500 still raises ────────────────────


def test_get_optional_reraises_on_500():
    hb = _import_http_backend()
    backend = _backend_with_response(hb, _fake_response(500))
    import requests
    with pytest.raises(requests.HTTPError):
        backend._get_optional("/v1/current", params={"namespace": "x"})


# ── GET /v1/session/digest ───────────────────────────────────────────


def test_session_digest_returns_object_and_builds_params():
    hb = _import_http_backend()
    digest_obj = {
        "narrative_head": {"rid": "019f-head", "snippet": "…"},
        "top_decisions": [], "open_conflicts": [], "open_conflict_count": 0,
        "pending_triggers": [], "pending_trigger_count": 0,
        "last_maintenance": None,
    }
    backend = _backend_with_response(hb, _fake_response(200, digest_obj))

    out = backend.session_digest(
        narrative_namespace="narr", scope="tenantA",
        include_gaps=True, max_gaps=3, max_decisions=8,
    )

    assert out == digest_obj
    assert isinstance(out, dict), "HTTP digest is a first-class object, not a JSON string"
    params = backend._session.get.call_args.kwargs["params"]
    assert params["namespace"] == "narr"
    assert params["scope"] == "tenantA"
    assert params["include_gaps"] == "true"
    assert params["max_gaps"] == 3
    assert params["max_decisions"] == 8


def test_session_digest_omits_optional_params_when_defaulted():
    hb = _import_http_backend()
    backend = _backend_with_response(hb, _fake_response(200, {}))
    backend.session_digest()  # all defaults, include_gaps False
    params = backend._session.get.call_args.kwargs["params"]
    assert "namespace" not in params
    assert "scope" not in params
    assert "include_gaps" not in params
    assert "max_gaps" not in params


# ── GET /v1/insights/gaps ────────────────────────────────────────────


def test_knowledge_gaps_returns_gaps_list():
    hb = _import_http_backend()
    gaps_payload = {
        "count": 2,
        "gaps": [
            {"query": "how do I rotate keys", "count": 5, "avg_top_score": 0.2},
            {"query": "cluster failover steps", "count": 3, "avg_top_score": 0.31},
        ],
    }
    backend = _backend_with_response(hb, _fake_response(200, gaps_payload))

    rows = backend.knowledge_gaps(min_count=2, max_avg_top_score=0.5, limit=10)

    assert rows == gaps_payload["gaps"], (
        "knowledge_gaps must return the bare list — the tool layer wraps it in "
        "{count, gaps} itself"
    )
    params = backend._session.get.call_args.kwargs["params"]
    assert params == {"min_count": 2, "max_avg_top_score": 0.5, "limit": 10}


def test_knowledge_gaps_forwards_namespace_when_set():
    hb = _import_http_backend()
    backend = _backend_with_response(hb, _fake_response(200, {"gaps": []}))
    backend.knowledge_gaps(namespace="proj")
    assert backend._session.get.call_args.kwargs["params"]["namespace"] == "proj"


def test_knowledge_gaps_missing_gaps_key_is_empty_list():
    hb = _import_http_backend()
    backend = _backend_with_response(hb, _fake_response(200, {}))
    assert backend.knowledge_gaps() == []
