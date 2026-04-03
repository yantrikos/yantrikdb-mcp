"""Unit tests for network transport bearer auth (no YantrikDB calls)."""

import asyncio
import json

import pytest

from yantrikdb_mcp.auth import BearerTokenMiddleware


def _http_scope(*, headers: list[tuple[bytes, bytes]] | None = None) -> dict:
    return {"type": "http", "headers": headers or []}


@pytest.fixture
def echo_app():
    """Minimal ASGI app that returns 200 + body."""

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"text/plain"]],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})

    return app


def test_bearer_401_when_missing(echo_app):
    mw = BearerTokenMiddleware(echo_app, api_key="secret")
    events: list[dict] = []

    async def run():
        async def send(e):
            events.append(e)

        async def receive():
            return {"type": "http.disconnect"}

        await mw(_http_scope(), receive, send)

    asyncio.run(run())

    assert events[0]["type"] == "http.response.start"
    assert events[0]["status"] == 401
    body = json.loads(events[1]["body"].decode())
    assert "error" in body


def test_bearer_401_when_wrong_token(echo_app):
    mw = BearerTokenMiddleware(echo_app, api_key="secret")
    events: list[dict] = []

    async def run():
        async def send(e):
            events.append(e)

        async def receive():
            return {"type": "http.disconnect"}

        scope = _http_scope(headers=[(b"authorization", b"Bearer wrong")])
        await mw(scope, receive, send)

    asyncio.run(run())
    assert events[0]["status"] == 401


def test_bearer_allows_valid_token(echo_app):
    mw = BearerTokenMiddleware(echo_app, api_key="secret")
    events: list[dict] = []

    async def run():
        async def send(e):
            events.append(e)

        async def receive():
            return {"type": "http.disconnect"}

        scope = _http_scope(headers=[(b"authorization", b"Bearer secret")])
        await mw(scope, receive, send)

    asyncio.run(run())
    assert events[0]["status"] == 200
    assert events[1]["body"] == b"ok"


def test_non_http_passthrough():
    """Lifespan and similar scopes should not require Authorization."""

    reached: list[str] = []

    async def app(scope, receive, send):
        reached.append(scope["type"])

    mw = BearerTokenMiddleware(app, api_key="secret")

    async def run():
        async def send(_):
            pass

        async def receive():
            return {"type": "http.disconnect"}

        await mw({"type": "lifespan"}, receive, send)

    asyncio.run(run())
    assert reached == ["lifespan"]
