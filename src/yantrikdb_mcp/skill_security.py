"""Identity, gate-hardening, operational, and supply-chain controls
for the skill substrate. Schema validation lives in skill_validation;
content scanning lives in skill_content_scanner; everything else
(B/C/D/E/F/G classes from the v0.8.0 threat model) is here.

Design rule: env vars are read ONCE at module import (C2) to defeat
sub-process env spoofing and runtime-flip races. Operators who change
config must restart the MCP server. The frozen snapshot is exposed via
`config()` for the audit log to capture.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import socket
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# C2 — Lock env-derived config at startup
# ─────────────────────────────────────────────────────────────────────

_NS_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _read_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _read_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _read_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_iso(name: str) -> datetime | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        # Accept both with and without 'Z' suffix
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


class _Config:
    """Frozen at import time. Defeats env-var spoofing via sub-process
    inheritance (C2). To change config: restart the MCP server."""

    def __init__(self) -> None:
        # The gate itself (already used by tools.py)
        self.writes_enabled = _read_bool("YANTRIKDB_SKILLS_WRITE_ENABLED")
        # Outcome gate — split from writes_enabled in v0.8.1 (issue #8).
        # `outcome` calls cannot introduce new instructions (they only
        # append {succeeded, note≤500} against an already-validated
        # skill_id, with the same A1/A2/A4 content scan + D2 rate
        # limit applied), so their threat profile is meaningfully
        # different from `define`. Default TRUE so the feedback loop
        # works out of the box; operators can flip to false if they
        # want to lock the outcome substrate too.
        self.outcomes_enabled = _read_bool_default(
            "YANTRIKDB_OUTCOMES_WRITE_ENABLED", default=True
        )
        # C1 — time-bound gate (applies to both define + outcome —
        # forensic concerns are the same once writes are flowing)
        self.write_expires_at = _read_iso("YANTRIKDB_SKILLS_WRITE_EXPIRES_AT")
        # B1 — namespace allowlist
        ns_raw = _read_list("YANTRIKDB_SKILLS_ALLOWED_NAMESPACES")
        # Validate each entry against the same regex applies_to uses
        self.allowed_namespaces: tuple[str, ...] = tuple(
            n for n in ns_raw if _NS_RE.fullmatch(n)
        )
        # B3 — cross-origin replace
        self.allow_cross_origin_replace = _read_bool(
            "YANTRIKDB_SKILLS_ALLOW_CROSS_ORIGIN_REPLACE"
        )
        # D2 — rate limit (writes per minute per session, default 30)
        self.write_rate_per_min = _read_int("YANTRIKDB_SKILLS_WRITE_RATE", 30)
        # D1 — audit log path (None = no auditing)
        audit = os.environ.get("YANTRIKDB_SKILLS_AUDIT_LOG", "").strip()
        self.audit_log_path: Path | None = Path(audit) if audit else None
        # F — multi-tenant ack
        self.multitenant_ack = _read_bool("YANTRIKDB_SKILLS_MULTITENANT_ACK")
        # G — review queue for rule-type skills
        self.rule_requires_review = _read_bool_default(
            "YANTRIKDB_SKILLS_RULE_REQUIRES_REVIEW", default=True
        )
        # A3 toggle — exposed for content scanner (we already read it
        # there; mirrored here so the audit-log config snapshot is
        # complete)
        self.allow_urls = _read_bool("YANTRIKDB_SKILLS_ALLOW_URLS")
        # Disabled scanners (mirror)
        self.disabled_scanners = tuple(_read_list("YANTRIKDB_SKILLS_DISABLE_SCANNERS"))
        # Origin tag for B3 / E2
        self.author_origin = os.environ.get(
            "YANTRIKDB_SKILLS_AUTHOR_ORIGIN", "yantrikdb-mcp"
        )

    def snapshot(self) -> dict:
        """JSON-able snapshot for the audit log."""
        return {
            "writes_enabled": self.writes_enabled,
            "outcomes_enabled": self.outcomes_enabled,
            "write_expires_at": self.write_expires_at.isoformat() if self.write_expires_at else None,
            "allowed_namespaces": list(self.allowed_namespaces),
            "allow_cross_origin_replace": self.allow_cross_origin_replace,
            "write_rate_per_min": self.write_rate_per_min,
            "audit_log_path": str(self.audit_log_path) if self.audit_log_path else None,
            "multitenant_ack": self.multitenant_ack,
            "rule_requires_review": self.rule_requires_review,
            "allow_urls": self.allow_urls,
            "disabled_scanners": list(self.disabled_scanners),
            "author_origin": self.author_origin,
        }


def _read_bool_default(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


CONFIG = _Config()


def config() -> _Config:
    """Public accessor — returns the FROZEN config object. Reading
    `os.environ` after this point won't change MCP behavior."""
    return CONFIG


# ─────────────────────────────────────────────────────────────────────
# C1 — Time-bound gate
# ─────────────────────────────────────────────────────────────────────


def gate_open(action: str = "define") -> tuple[bool, str | None]:
    """Returns (is_open, reason_if_closed) for the given action.

    Two gates in v0.8.1+ (yantrikos/yantrikdb-mcp#8):
      - `define` (and `replace`): YANTRIKDB_SKILLS_WRITE_ENABLED, default FALSE
      - `outcome`: YANTRIKDB_OUTCOMES_WRITE_ENABLED, default TRUE

    Both share the C1 time-bound expiry — once it fires, all writes
    are refused regardless of which flag opened them.

    Default arg is "define" so existing callers (and tests) that don't
    pass action keep the v0.8.0 semantics.
    """
    if action == "outcome":
        if not CONFIG.outcomes_enabled:
            return False, (
                "YANTRIKDB_OUTCOMES_WRITE_ENABLED is false. Skill outcome "
                "tracking is disabled on this MCP server. Set the env var "
                "to true (or unset — it defaults to true) to enable the "
                "feedback loop."
            )
    else:
        # All other write actions (define, replace) use the stricter gate
        if not CONFIG.writes_enabled:
            return False, (
                "YANTRIKDB_SKILLS_WRITE_ENABLED is false. Skill writes are "
                "off by default; set the env var to true on the MCP server "
                "to enable agent-authored skills."
            )

    if CONFIG.write_expires_at is not None:
        now = datetime.now(timezone.utc)
        if now >= CONFIG.write_expires_at:
            return False, (
                f"YANTRIKDB_SKILLS_WRITE_EXPIRES_AT={CONFIG.write_expires_at.isoformat()} "
                f"has passed (now={now.isoformat()}). The time-bound gate "
                f"is closed — restart the MCP server with an updated "
                f"expiry to re-enable writes."
            )
    return True, None


# ─────────────────────────────────────────────────────────────────────
# B1 — Namespace allowlist
# ─────────────────────────────────────────────────────────────────────


def check_namespace_allowed(skill_id: str) -> None:
    """If the operator configured an allowlist, the skill_id's first
    dot-segment must be in it. Raise ValueError otherwise.

    If no allowlist is set, all namespaces are allowed (this is the
    backward-compatible default; the gate already restricts WHO can
    write at all)."""
    if not CONFIG.allowed_namespaces:
        return
    first_segment = skill_id.split(".", 1)[0]
    if first_segment not in CONFIG.allowed_namespaces:
        raise ValueError(
            f"[B1] skill_id namespace {first_segment!r} not in operator "
            f"allowlist {list(CONFIG.allowed_namespaces)}. Either pick a "
            f"skill_id under an allowed prefix, or have the operator "
            f"add this namespace to YANTRIKDB_SKILLS_ALLOWED_NAMESPACES."
        )


# ─────────────────────────────────────────────────────────────────────
# B2 — Author attribution (record WHO wrote this)
# ─────────────────────────────────────────────────────────────────────


def author_attribution(*, session_id: str | None = None) -> dict:
    """Generate the author-attribution sub-dict that goes into a
    skill's metadata at write time. Forensic trail for D1 audit log.

    Best-effort — none of these can be cryptographically verified, but
    any of them is better than nothing post-incident.
    """
    return {
        "session_id": session_id,
        "os_user": _safe(getpass.getuser),
        "hostname": _safe(socket.gethostname),
        "wall_clock_at_define": datetime.now(timezone.utc).isoformat(),
        "author_origin": CONFIG.author_origin,
        # Unique per-write nonce so audit-log grep can correlate "this
        # define event in the log" with "this metadata blob in the DB"
        "audit_nonce": uuid.uuid4().hex,
    }


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# B3 — Cross-origin replace guard
# ─────────────────────────────────────────────────────────────────────


def check_cross_origin_replace(existing_meta: dict | None, new_origin: str) -> None:
    """If replacing a skill written by a different origin, require an
    explicit operator ack. Prevents one consumer (e.g. this MCP) from
    silently overwriting another's (e.g. hermes-plugin's) skills."""
    if not existing_meta:
        return
    existing_origin = existing_meta.get("author_origin") or "unknown"
    if existing_origin == new_origin:
        return
    if CONFIG.allow_cross_origin_replace:
        return
    raise ValueError(
        f"[B3] refusing cross-origin replace: existing skill was written "
        f"by {existing_origin!r}, this MCP runs as {new_origin!r}. Set "
        f"YANTRIKDB_SKILLS_ALLOW_CROSS_ORIGIN_REPLACE=true on the MCP "
        f"server to allow this, or change skill_id to author a new "
        f"skill instead of replacing."
    )


# ─────────────────────────────────────────────────────────────────────
# B4 — Supersedes integrity
# ─────────────────────────────────────────────────────────────────────


def check_supersedes_integrity(
    supersedes_id: str | None,
    new_skill_id: str,
    superseded_meta: dict | None,
) -> None:
    """If a skill claims to supersede another:
      (a) the other must exist (meta is non-None)
      (b) both must be in the same first-namespace (no cross-namespace
          supersedure — a 'workflow.*' skill can't retire an 'auth.*' one)
    Raise ValueError otherwise.
    """
    if not supersedes_id:
        return
    if superseded_meta is None:
        raise ValueError(
            f"[B4] supersedes={supersedes_id!r} but no such skill exists "
            f"in the catalog. Refusing — operators have no way to verify "
            f"the intent if the predecessor isn't there to compare."
        )
    new_ns = new_skill_id.split(".", 1)[0]
    old_ns = supersedes_id.split(".", 1)[0]
    if new_ns != old_ns:
        raise ValueError(
            f"[B4] cross-namespace supersedure refused: "
            f"new skill in namespace {new_ns!r} cannot supersede "
            f"a skill in namespace {old_ns!r}. Supersedure is "
            f"scoped to one namespace by design."
        )


# ─────────────────────────────────────────────────────────────────────
# D2 — Rate limit (per-session token bucket)
# ─────────────────────────────────────────────────────────────────────


class _RateLimiter:
    """Sliding-window counter per session_id. Threadsafe. Eviction is
    lazy — entries older than the window get culled when accessed."""

    def __init__(self, per_minute: int) -> None:
        self._per_minute = max(1, per_minute)
        self._window_seconds = 60.0
        self._writes: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check_and_record(self, session_id: str) -> None:
        """Raise ValueError if the caller has exceeded per_minute writes
        in the last 60s. Otherwise record the new write and return."""
        now = time.time()
        cutoff = now - self._window_seconds
        with self._lock:
            q = self._writes[session_id]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._per_minute:
                oldest = q[0]
                retry_in = max(0.0, self._window_seconds - (now - oldest))
                raise ValueError(
                    f"[D2] rate limit exceeded: {len(q)} writes in the last "
                    f"60 seconds (limit {self._per_minute}). Retry in "
                    f"{retry_in:.1f}s, or raise YANTRIKDB_SKILLS_WRITE_RATE."
                )
            q.append(now)


_RATE_LIMITER = _RateLimiter(CONFIG.write_rate_per_min)


def check_rate_limit(session_id: str | None) -> None:
    _RATE_LIMITER.check_and_record(session_id or "default")


# ─────────────────────────────────────────────────────────────────────
# D1 — Audit log
# ─────────────────────────────────────────────────────────────────────


_AUDIT_LOCK = threading.Lock()


def audit_event(event: dict) -> None:
    """Append-only JSONL audit log. No-op if YANTRIKDB_SKILLS_AUDIT_LOG
    isn't configured. Errors are swallowed (audit failure should not
    block legitimate operations; the audit log is best-effort)."""
    if CONFIG.audit_log_path is None:
        return
    try:
        line = json.dumps(event, default=str, sort_keys=True)
        with _AUDIT_LOCK:
            CONFIG.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with CONFIG.audit_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Audit-log failures must not block the operation. Swallow
        # and rely on the next D4 (counter) for surface signal.
        pass


# ─────────────────────────────────────────────────────────────────────
# D4 — Counters (in-memory, exported via the stats tool)
# ─────────────────────────────────────────────────────────────────────


class _Counters:
    """Lightweight in-memory counters. The MCP server is generally
    long-lived enough that this gives operators a meaningful 'are we
    under attack right now' signal. Resets on restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.skill_defines_accepted = 0
        self.skill_defines_rejected = defaultdict(int)  # reason → count
        self.skill_outcomes_recorded = 0
        self.skill_outcomes_rejected = defaultdict(int)
        self.skill_pending_review = 0

    def accept_define(self) -> None:
        with self._lock:
            self.skill_defines_accepted += 1

    def reject_define(self, reason: str) -> None:
        with self._lock:
            self.skill_defines_rejected[reason] += 1

    def record_outcome(self) -> None:
        with self._lock:
            self.skill_outcomes_recorded += 1

    def reject_outcome(self, reason: str) -> None:
        with self._lock:
            self.skill_outcomes_rejected[reason] += 1

    def queue_for_review(self) -> None:
        with self._lock:
            self.skill_pending_review += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "skill_defines_accepted": self.skill_defines_accepted,
                "skill_defines_rejected": dict(self.skill_defines_rejected),
                "skill_outcomes_recorded": self.skill_outcomes_recorded,
                "skill_outcomes_rejected": dict(self.skill_outcomes_rejected),
                "skill_pending_review": self.skill_pending_review,
            }


COUNTERS = _Counters()


# ─────────────────────────────────────────────────────────────────────
# E1 — Content-addressable hash
# ─────────────────────────────────────────────────────────────────────


def body_sha256(body: str) -> str:
    """Hex digest of the body — stamped into metadata at write time,
    re-checked on reads to detect out-of-band tampering."""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def verify_body_hash(body: str, expected: str | None) -> bool:
    """Return True if expected is None (legacy skill from pre-E1 write)
    or matches the body. False otherwise — caller decides whether to
    surface or omit."""
    if not expected:
        return True
    return body_sha256(body) == expected


# ─────────────────────────────────────────────────────────────────────
# F — "Don't ship this configuration" startup checks
# ─────────────────────────────────────────────────────────────────────


def startup_safety_checks(
    *,
    is_cluster_mode: bool,
    db_actor_ids: list[str] | None = None,
) -> list[str]:
    """Return a list of WARNING strings about dangerous configurations.
    Empty list means we're safe. Caller logs these at WARNING level on
    init and includes them in the first audit event so operators see
    them in `journalctl -u yantrikdb-mcp`."""
    warnings: list[str] = []

    if CONFIG.writes_enabled:
        # F.1 — multi-tenant DB without ack
        if db_actor_ids and len(set(db_actor_ids)) > 1 and not CONFIG.multitenant_ack:
            warnings.append(
                f"[F.1] gate is on AND the DB shows writes from "
                f"{len(set(db_actor_ids))} different actor_ids. This is a "
                f"multi-tenant configuration — set "
                f"YANTRIKDB_SKILLS_MULTITENANT_ACK=true to explicitly "
                f"acknowledge or run separate databases per tenant."
            )

        # F.2 — cluster mode (HTTP) + write gate on
        if is_cluster_mode:
            warnings.append(
                "[F.2] gate is on AND running in HTTP cluster mode. "
                "The cluster must independently authenticate skill "
                "writes (the MCP server's gate alone is not sufficient). "
                "Verify your yantrikdb-server enforces write-auth on "
                "/v1/skills/*."
            )

        # F.3 — gate on without time-bound
        if CONFIG.write_expires_at is None:
            warnings.append(
                "[F.3] gate is on without an expiry "
                "(YANTRIKDB_SKILLS_WRITE_EXPIRES_AT not set). Consider "
                "time-bounding the write window — agents-authoring-skills "
                "is rarely a permanent state."
            )

        # F.4 — gate on without namespace allowlist
        if not CONFIG.allowed_namespaces:
            warnings.append(
                "[F.4] gate is on without a namespace allowlist "
                "(YANTRIKDB_SKILLS_ALLOWED_NAMESPACES unset). Agents can "
                "author into any namespace, including 'security.*' / "
                "'auth.*'. Consider scoping."
            )

        # F.5 — gate on without audit log
        if CONFIG.audit_log_path is None:
            warnings.append(
                "[F.5] gate is on without an audit log "
                "(YANTRIKDB_SKILLS_AUDIT_LOG unset). Skill writes will "
                "not be recorded for forensic review."
            )

    return warnings


# ─────────────────────────────────────────────────────────────────────
# G — Review queue routing
# ─────────────────────────────────────────────────────────────────────

PENDING_NAMESPACE = "skill_pending_review"


def should_route_to_review(skill_type: str, supersedes: str | None) -> bool:
    """Two cases route to the pending-review namespace instead of the
    live catalog:
      - rule-type skills (G primary rationale)
      - any supersedure across origins (will already have been blocked
        by B3 unless explicitly allowed; that goes through review too)
    """
    if CONFIG.rule_requires_review and skill_type == "rule":
        return True
    return False
