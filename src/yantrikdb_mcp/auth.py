"""Bearer token authentication middleware for YantrikDB MCP network transports."""

import hmac
import json


class BearerTokenMiddleware:
    """Pure ASGI middleware for bearer token auth.

    Uses raw ASGI instead of BaseHTTPMiddleware to avoid issues with
    SSE streaming (long-lived connections that BaseHTTPMiddleware can't handle).

    Usage:
        YANTRIKDB_API_KEY=mysecret yantrikdb-mcp --transport sse

    Client connects with:
        Authorization: Bearer mysecret
    """

    def __init__(self, app, api_key: str):
        self.app = app
        self._api_key = api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Extract bearer token from headers
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()

        token = auth_header[7:] if auth_header.startswith("Bearer ") else ""

        if not hmac.compare_digest(token, self._api_key):
            body = json.dumps({"error": "Invalid or missing bearer token"}).encode()
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"www-authenticate", b"Bearer"],
                    [b"content-length", str(len(body)).encode()],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })
            return

        await self.app(scope, receive, send)
