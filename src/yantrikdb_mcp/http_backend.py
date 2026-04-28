"""HTTP backend — proxies MCP tool calls to a YantrikDB HTTP cluster.

When YANTRIKDB_SERVER_URL is set (e.g. "http://192.168.4.140:7438,http://192.168.4.141:7438"),
the MCP server forwards all operations to the cluster via HTTP instead of using
the embedded engine.  Leader auto-discovery handles Raft failovers transparently.

Not every embedded-backend method has a corresponding HTTP endpoint yet. The
methods that don't raise `RemoteUnsupportedError` with a clear hint, instead
of an opaque `AttributeError`. See yantrikdb-mcp issue #2 for context.
"""

import json
import logging
import time
from urllib.parse import urljoin

import requests

log = logging.getLogger("yantrikdb.mcp.http")


class RemoteUnsupportedError(NotImplementedError):
    """Raised when a method is not yet exposed over the HTTP transport.

    Using NotImplementedError as the base means tool handlers that catch it
    can render a friendly message; raw `except Exception` callers also see a
    sensible class name in tracebacks.
    """

    def __init__(self, method: str, hint: str = "") -> None:
        msg = (
            f"{method!r} is not supported when YANTRIKDB_SERVER_URL points at a "
            f"remote cluster — the embedded backend exposes it but the HTTP "
            f"transport does not yet have a corresponding endpoint."
        )
        if hint:
            msg = f"{msg}\nHint: {hint}"
        msg = (
            f"{msg}\nWorkaround: run yantrikdb-mcp in embedded mode "
            f"(unset YANTRIKDB_SERVER_URL) or open an issue at "
            f"https://github.com/yantrikos/yantrikdb-mcp/issues."
        )
        super().__init__(msg)
        self.method = method


class HttpBackend:
    """Drop-in replacement for the embedded YantrikDB engine that forwards to HTTP API.

    Implements the same method signatures that tools.py calls on the db object,
    so `_get_db(ctx)` can return this transparently.
    """

    def __init__(self, server_urls: list[str], token: str, timeout: int = 15):
        self._nodes = server_urls
        self._token = token
        self._leader: str | None = None
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    # ── leader discovery ────────────────────────────────────────────

    def _find_leader(self) -> str:
        """Query /v1/health on each node, return the one that accepts writes."""
        for node in self._nodes:
            try:
                r = self._session.get(f"{node}/v1/health", timeout=3)
                if r.ok:
                    data = r.json()
                    if data.get("cluster", {}).get("accepts_writes", False):
                        self._leader = node
                        return node
            except Exception:
                continue
        # No leader found — fall back to first node (might be single-node)
        # or the last known leader
        if self._leader:
            return self._leader
        self._leader = self._nodes[0]
        return self._leader

    def _url(self, path: str) -> str:
        """Get full URL for an API path, discovering leader if needed."""
        if not self._leader:
            self._find_leader()
        return f"{self._leader}{path}"

    def _post(self, path: str, body: dict, retries: int = 2) -> dict:
        """POST to the leader with auto-retry on 503 (leader failover)."""
        for attempt in range(retries + 1):
            try:
                r = self._session.post(self._url(path), json=body, timeout=self._timeout)
                if r.status_code == 503:
                    # Leader may have flipped — rediscover
                    log.warning("503 from %s, rediscovering leader...", self._leader)
                    self._leader = None
                    self._find_leader()
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.ConnectionError:
                self._leader = None
                self._find_leader()
                if attempt == retries:
                    raise
        return {}

    def _get(self, path: str, params: dict | None = None, retries: int = 2) -> dict:
        """GET from the leader with auto-retry on 503."""
        for attempt in range(retries + 1):
            try:
                r = self._session.get(self._url(path), params=params, timeout=self._timeout)
                if r.status_code == 503:
                    self._leader = None
                    self._find_leader()
                    continue
                r.raise_for_status()
                return r.json()
            except requests.exceptions.ConnectionError:
                self._leader = None
                self._find_leader()
                if attempt == retries:
                    raise
        return {}

    # ── methods called by tools.py ──────────────────────────────────

    def record(self, text: str, *, memory_type="semantic", importance=0.5,
               valence=0.0, metadata=None, namespace="default", certainty=0.8,
               domain="general", source="user", emotional_state=None, **_kw) -> str:
        body = {
            "text": text,
            "memory_type": memory_type,
            "importance": importance,
            "valence": valence,
            "metadata": metadata or {},
            "namespace": namespace,
            "certainty": certainty,
            "domain": domain,
            "source": source,
        }
        if emotional_state:
            body["emotional_state"] = emotional_state
        result = self._post("/v1/remember", body)
        return result.get("rid", "")

    def recall_with_response(self, *, query: str, top_k: int = 10,
                             memory_type=None, include_consolidated=False,
                             expand_entities=True, namespace=None,
                             domain=None, source=None) -> dict:
        body: dict = {"query": query, "top_k": top_k}
        if memory_type:
            body["memory_type"] = memory_type
        if namespace:
            body["namespace"] = namespace
        if domain:
            body["domain"] = domain
        if source:
            body["source"] = source
        result = self._post("/v1/recall", body)
        # Adapt server response to match embedded engine format
        items = []
        for r in result.get("results", []):
            items.append({
                "rid": r.get("rid", ""),
                "text": r.get("text", ""),
                "type": r.get("memory_type", "semantic"),
                "score": r.get("score", 0.0),
                "importance": r.get("importance", 0.5),
                "created_at": r.get("created_at", 0),
                "why_retrieved": r.get("why_retrieved", []),
            })
        return {
            "results": items,
            "confidence": result.get("confidence", 0.0) if "confidence" in result else (
                max((r["score"] for r in items), default=0.0)
            ),
            "hints": result.get("hints", []),
        }

    def forget(self, rid: str) -> bool:
        result = self._post("/v1/forget", {"rid": rid})
        return result.get("found", False)

    def correct(self, rid: str, new_text: str, *,
                new_importance=None, new_valence=None,
                correction_note=None) -> dict:
        body: dict = {"rid": rid, "new_text": new_text}
        if new_importance is not None:
            body["new_importance"] = new_importance
        if new_valence is not None:
            body["new_valence"] = new_valence
        if correction_note:
            body["correction_note"] = correction_note
        return self._post("/v1/correct", body)

    def think(self, config=None) -> "ThinkResult":
        body = {}
        if config:
            body["run_consolidation"] = getattr(config, "run_consolidation", True)
            body["run_conflicts"] = getattr(config, "run_conflicts", True)
            body["run_patterns"] = getattr(config, "run_patterns", True)
            body["run_personality"] = getattr(config, "run_personality", True)
        result = self._post("/v1/think", body)
        return _ThinkResult(result)

    def relate(self, src: str, dst: str, rel_type: str, weight: float = 1.0) -> str:
        body = {"src": src, "dst": dst, "rel_type": rel_type, "weight": weight}
        result = self._post("/v1/relate", body)
        return result.get("edge_id", "")

    def get_edges(self, entity: str) -> list:
        result = self._post("/v1/graph", {"action": "query", "entity": entity})
        edges = result.get("edges", [])
        return [_Edge(e) for e in edges]

    def get_conflicts(self, status=None, priority=None, entity=None,
                      conflict_type=None, limit=20) -> list:
        # Server may accept query-string filters; pass them through best-effort.
        params: dict = {}
        if status:
            params["status"] = status
        if priority:
            params["priority"] = priority
        if entity:
            params["entity"] = entity
        if conflict_type:
            params["conflict_type"] = conflict_type
        if limit:
            params["limit"] = limit
        result = self._get("/v1/conflicts", params=params or None)
        conflicts = result.get("conflicts", [])
        # Client-side filter belt-and-braces in case the server didn't honor a param
        if status:
            conflicts = [c for c in conflicts if c.get("status") == status]
        if priority:
            conflicts = [c for c in conflicts if c.get("priority") == priority]
        if entity:
            conflicts = [c for c in conflicts if c.get("entity") == entity]
        if conflict_type:
            conflicts = [c for c in conflicts if c.get("conflict_type") == conflict_type]
        # Return plain dicts — tools.py uses subscript access (c["..."]).
        # Embedded backend also returns dicts; HttpBackend used to wrap in
        # _Conflict, which broke `conflict(action="list")` with a
        # `'_Conflict' object is not subscriptable` error. (issue #2)
        return list(conflicts[:limit])

    def get_conflict(self, conflict_id: str):
        """Get a single conflict by id. Server has no /v1/conflicts/{id} GET
        today, so we filter the list response. Returns plain dict or None."""
        all_conf = self._get("/v1/conflicts").get("conflicts", [])
        for c in all_conf:
            if c.get("conflict_id") == conflict_id:
                return c
        return None

    def resolve_conflict(self, conflict_id: str, strategy: str,
                         winner_rid=None, new_text=None,
                         resolution_note=None, note=None) -> "_ResolveResult":
        body: dict = {"strategy": strategy}
        if winner_rid:
            body["winner_rid"] = winner_rid
        if new_text:
            body["new_text"] = new_text
        # Accept both kw spellings for cross-backend compat
        rn = resolution_note if resolution_note is not None else note
        if rn:
            body["resolution_note"] = rn
        # Server route is /v1/conflicts/{id}/resolve, NOT /v1/conflicts/resolve
        result = self._post(f"/v1/conflicts/{conflict_id}/resolve", body)
        return _ResolveResult(result)

    def dismiss_conflict(self, conflict_id: str, note=None) -> bool:
        body: dict = {"strategy": "dismiss"}
        if note:
            body["resolution_note"] = note
        self._post(f"/v1/conflicts/{conflict_id}/resolve", body)
        return True

    def reclassify_conflict(self, conflict_id: str, new_type: str,
                            note=None) -> "_ResolveResult":
        # No dedicated reclassify endpoint yet; closest path is resolve with a
        # custom strategy. Until the server exposes one, raise the clear
        # not-supported error so callers can degrade gracefully.
        raise self._not_supported(
            "reclassify_conflict",
            hint="Use embedded mode or file a feature request for "
                 "POST /v1/conflicts/{id}/reclassify."
        )

    def stats(self, namespace=None) -> "_Stats":
        result = self._get("/v1/stats")
        return _Stats(result)

    def get(self, rid: str):
        """Get a single memory by RID — returns None-like if not found."""
        # The server doesn't have a direct GET-by-RID endpoint;
        # recall with exact text isn't feasible. Return a stub.
        return None

    def get_beliefs_above(self, min_confidence: float) -> list:
        return []

    def get_patterns(self, pattern_type=None, status=None, limit=10) -> list:
        return []

    def recall_feedback(self, *, rid, feedback, query_text=None,
                        score_at_retrieval=None, rank_at_retrieval=None):
        pass  # Not exposed via HTTP API yet

    def recall_refine(self, *, original_query_embedding, refinement_text,
                      original_rids, top_k, namespace=None, domain=None, source=None):
        # Fall back to regular recall
        return self.recall_with_response(
            query=refinement_text, top_k=top_k,
            namespace=namespace, domain=domain, source=source,
        )

    def embed(self, text: str):
        return None  # Not needed for HTTP backend recall_refine fallback

    def get_personality(self, recompute=False) -> dict:
        result = self._get("/v1/personality")
        return result

    def session_end(self, session_id: str, summary: str | None = None) -> dict:
        """End an active session. Maps to DELETE /v1/sessions/{id}."""
        body: dict = {}
        if summary:
            body["summary"] = summary
        # _post supports POST; for DELETE we use the session directly
        url = self._url(f"/v1/sessions/{session_id}")
        r = self._session.delete(url, json=body, timeout=self._timeout)
        r.raise_for_status()
        return r.json() if r.content else {}

    # ── stubs for methods without HTTP endpoints today ──────────────
    # Each raises RemoteUnsupportedError with a clear pointer. This replaces
    # the previous behaviour of `AttributeError: 'HttpBackend' object has no
    # attribute '<method>'` (issue #2). Add real impls as endpoints land.

    @staticmethod
    def _not_supported(method: str, hint: str = "") -> RemoteUnsupportedError:
        return RemoteUnsupportedError(method, hint)

    def procedural_stats(self) -> dict:
        raise self._not_supported(
            "procedural_stats",
            hint="No /v1/stats sub-endpoint for procedural state today."
        )

    def archive(self, rid: str) -> bool:
        raise self._not_supported("archive")

    def hydrate(self, rid: str) -> bool:
        raise self._not_supported("hydrate")

    def list_memories(self, **kw) -> list:
        raise self._not_supported("list_memories",
                                  hint="Use recall_with_response with a broad query.")

    def derive_personality(self, **kw) -> dict:
        raise self._not_supported("derive_personality")

    def get_pending_triggers(self, **kw) -> list:
        raise self._not_supported("get_pending_triggers")

    def get_trigger_history(self, **kw) -> list:
        raise self._not_supported("get_trigger_history")

    def entity_profile(self, *args, **kw) -> dict:
        raise self._not_supported("entity_profile")

    def search_entities(self, *args, **kw) -> list:
        raise self._not_supported("search_entities")

    def relationship_depth(self, *args, **kw) -> int:
        raise self._not_supported("relationship_depth")

    def link_memory_entity(self, *args, **kw):
        raise self._not_supported("link_memory_entity")

    def backfill_memory_entities(self, *args, **kw):
        raise self._not_supported("backfill_memory_entities")

    def rebuild_graph_index(self, *args, **kw):
        raise self._not_supported("rebuild_graph_index")

    def rebuild_vec_index(self, *args, **kw):
        raise self._not_supported("rebuild_vec_index")

    def record_procedural(self, *args, **kw):
        raise self._not_supported("record_procedural")

    def reinforce_procedural(self, *args, **kw):
        raise self._not_supported("reinforce_procedural")

    def learn_category_members(self, *args, **kw):
        raise self._not_supported("learn_category_members")

    def reset_category_to_seed(self, *args, **kw):
        raise self._not_supported("reset_category_to_seed")

    def learned_weights(self, *args, **kw):
        raise self._not_supported("learned_weights")

    def session_history(self, *args, **kw) -> list:
        raise self._not_supported("session_history")

    def session_abandon_stale(self, *args, **kw):
        raise self._not_supported("session_abandon_stale")

    def active_session(self, *args, **kw):
        raise self._not_supported("active_session",
                                  hint="POST /v1/sessions to start a session; "
                                       "track session_id client-side.")

    def close(self):
        self._session.close()


# ── Adapter classes to match embedded engine return types ───────────

class _ThinkResult:
    def __init__(self, d: dict):
        self.triggers = d.get("triggers", [])
        self.consolidation_count = d.get("consolidations", 0)
        self.conflicts_found = d.get("conflicts_found", 0)
        self.patterns_new = d.get("patterns_new", 0)
        self.patterns_updated = d.get("patterns_updated", 0)
        self.duration_ms = d.get("duration_ms", 0)


class _Edge:
    def __init__(self, d: dict):
        self.edge_id = d.get("edge_id", "")
        self.src = d.get("src", "")
        self.dst = d.get("dst", "")
        self.rel_type = d.get("rel_type", "")
        self.weight = d.get("weight", 1.0)


class _Conflict:
    def __init__(self, d: dict):
        self.conflict_id = d.get("conflict_id", "")
        self.conflict_type = d.get("conflict_type", "")
        self.priority = d.get("priority", "")
        self.status = d.get("status", "")
        self.memory_a = d.get("memory_a", "")
        self.memory_b = d.get("memory_b", "")
        self.entity = d.get("entity", "")
        self.detection_reason = d.get("detection_reason", "")
        self.detected_at = d.get("detected_at", "")
        self.consolidation_status = d.get("status", "")


class _ResolveResult:
    def __init__(self, d: dict):
        self.winner_rid = d.get("winner_rid")
        self.loser_tombstoned = d.get("loser_tombstoned", False)
        self.new_memory_rid = d.get("new_memory_rid")


class _Stats:
    def __init__(self, d: dict):
        self.active_memories = d.get("active_memories", 0)
        self.consolidated_memories = d.get("consolidated_memories", 0)
        self.tombstoned_memories = d.get("tombstoned_memories", 0)
        self.archived_memories = d.get("archived_memories", 0)
        self.edges = d.get("edges", 0)
        self.entities = d.get("entities", 0)
        self.open_conflicts = d.get("open_conflicts", 0)
        self.resolved_conflicts = d.get("resolved_conflicts", 0)
        self.pending_triggers = d.get("pending_triggers", 0)
        self.active_patterns = d.get("active_patterns", 0)
        self.scoring_cache_entries = d.get("scoring_cache_entries", 0)
        self.vec_index_entries = d.get("vec_index_entries", 0)
