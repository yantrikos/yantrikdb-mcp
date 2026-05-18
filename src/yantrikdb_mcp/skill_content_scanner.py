"""Content-layer hardening for skill bodies (and outcome notes).

Implements defense-in-depth scanning that runs AFTER schema validation
and BEFORE the body lands in the substrate. Each scanner raises
`ValueError` with a precise reason on rejection; callers re-raise as
`ToolError` so the MCP client sees a 4xx-equivalent (no retry).

Scanner classes (per the v0.8.0 threat model):
  - A1 prompt-injection markers (OWASP LLM01)
  - A2 credential patterns (mirrors GitHub secret-scanning rule set;
       not exhaustive — first line of defense, not a guarantee)
  - A3 URL/IP exfil paths (toggleable via YANTRIKDB_SKILLS_ALLOW_URLS=true)
  - A4 unicode evasion (Format / Surrogate chars used to hide content
       from human reviewers; we strip Cf/Cs except whitelisted)
  - A5 encoded-payload heuristic (long base64/hex runs)
  - A6 implicit (markdown links handled via A3 — any http(s) URL)

False positives are real. Each scanner can be disabled individually via
`YANTRIKDB_SKILLS_DISABLE_SCANNERS="A2,A5"` for operators who hit them
on legitimate content. Disabling defaults to off; auditable in the
audit log.
"""

from __future__ import annotations

import os
import re
import unicodedata

# ─────────────────────────────────────────────────────────────────────
# A1 — Prompt-injection markers (OWASP LLM01)
# ─────────────────────────────────────────────────────────────────────
# Patterns chosen from documented prompt-injection corpora. Matched
# case-insensitively, anchored to word boundaries where punctuation
# would otherwise trick the regex. Not exhaustive — this is the first
# tripwire, not the only defense.
PROMPT_INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions?|prompts?|messages?|rules?)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|earlier|safety|system) (?:instructions?|prompts?|rules?|guidelines?)",
    r"forget (?:all |any |everything |the )?(?:previous|prior|above|earlier|you (?:were|are) told)",
    r"you (?:are|'re) (?:now|hereby) (?:an? )?(?:admin|root|sudo|developer|jailbroken|in (?:admin|dev|debug|root) mode|in god mode)",
    r"act as (?:if you (?:are|were) |an? )?(?:admin|root|sudo|dev|jailbroken|unrestricted|uncensored|dan)\b",
    r"new (?:instructions?|directives?|system prompt|persona|role) ?:",
    r"system (?:prompt|message) ?:",
    # Role-confusion: literal role markers placed inside body text by
    # an attacker trying to make a downstream agent treat the text as
    # a fresh system message. Match at start-of-line, after a newline,
    # OR after a sentence boundary (`. ` / `! ` / `? `). Followed by a
    # directive-shaped word so legit prose like "system: a multi-tenant
    # database..." doesn't trip the scanner.
    r"(?:^|\n|[.!?]\s+)(?:system|assistant|user|tool|function)\s*:\s+(?:you|new|now|the|do|please|i|ignore|disregard)",
    r"<\|im_start\|>system",
    r"\[INST\] ?(?:You are|System:)",
    r"override (?:safety|content|alignment|security) (?:filters?|guidelines?|policies?|controls?)",
    r"reveal (?:your |the )?(?:system prompt|instructions|api key|secret)",
    r"do anything now\b",
    r"bypass (?:all |any )?(?:filters?|guardrails?|restrictions?|safety|alignment)",
    r"pretend (?:you (?:are|'re|have)|to be) (?:dan|jailbroken|unrestricted|uncensored)",
]
_PROMPT_INJECTION_RE = re.compile(
    "|".join(PROMPT_INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE
)


# ─────────────────────────────────────────────────────────────────────
# A2 — Credential / secret patterns
# ─────────────────────────────────────────────────────────────────────
# Subset of GitHub secret-scanning patterns. Not all of them — the goal
# is to catch the common shapes that prove the body shouldn't be in a
# shared catalog, not to ship a complete IDS.
CREDENTIAL_PATTERNS = {
    "aws_access_key_id":         r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "aws_secret_access_key":     r"(?i)aws(?:.{0,20})?(?:secret|access).{0,20}?[\"'=:\s]([A-Za-z0-9/+=]{40})",
    "github_pat":                r"\bghp_[A-Za-z0-9]{36,}\b",
    "github_oauth":              r"\bgho_[A-Za-z0-9]{36,}\b",
    "github_app_token":          r"\bgh[su]_[A-Za-z0-9]{36,}\b",
    "github_fine_grained":       r"\bgithub_pat_[A-Za-z0-9_]{82,}\b",
    "slack_token":               r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
    "slack_webhook":             r"\bhttps://hooks\.slack\.com/services/T[A-Za-z0-9]+/B[A-Za-z0-9]+/[A-Za-z0-9]+\b",
    "stripe_secret":             r"\bsk_(?:live|test)_[A-Za-z0-9]{24,}\b",
    "stripe_restricted":         r"\brk_(?:live|test)_[A-Za-z0-9]{24,}\b",
    "google_api_key":            r"\bAIza[0-9A-Za-z_-]{35}\b",
    "openai_api_key":            r"\bsk-[A-Za-z0-9]{20}T3BlbkFJ[A-Za-z0-9]{20}\b",
    "anthropic_api_key":         r"\bsk-ant-[A-Za-z0-9_-]{32,}\b",
    "private_key_pem":           r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY(?: BLOCK)?-----",
    "ssh_authorized_key":        r"\bssh-(?:rsa|ed25519|dss|ecdsa) [A-Za-z0-9+/=]{40,}\b",
    "jwt_token":                 r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
    "generic_password_assign":   r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|token)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
    "basic_auth_url":            r"\bhttps?://[^/\s:]+:[^/\s:]{4,}@[^\s/]+",
    "discord_bot_token":         r"\b[MN][A-Za-z0-9]{23,25}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27,38}\b",
    "twilio_sid":                r"\bAC[0-9a-f]{32}\b",
    "twilio_auth_token":         r"(?i)\btwilio(?:.{0,20})?[\"'=:\s]([0-9a-f]{32})",
}
_CREDENTIAL_RES = {name: re.compile(pat) for name, pat in CREDENTIAL_PATTERNS.items()}


# ─────────────────────────────────────────────────────────────────────
# A3 — URLs (default deny, operator toggle)
# ─────────────────────────────────────────────────────────────────────
# Catches http(s) and ftp URLs anywhere in the body, plus bracketed
# markdown forms `[text](http://...)`. IP literals (v4) are caught
# separately because they trip the same exfil concern.
_URL_RE = re.compile(r"\b(?:https?|ftp)://[^\s)\]>]+", re.IGNORECASE)
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
)


# ─────────────────────────────────────────────────────────────────────
# A4 — Unicode evasion (Format / Surrogate categories)
# ─────────────────────────────────────────────────────────────────────
# Allowlist a few legitimate Cf chars (e.g. zero-width-joiner for
# emoji families, soft hyphen). Reject everything else in Cf/Cs/Cn.
_UNICODE_ALLOWED_CF = frozenset({
    "‍",  # ZWJ (emoji combinations)
    "­",  # soft hyphen
    "﻿",  # BOM — strip it but don't reject (common in copy-paste)
})


# ─────────────────────────────────────────────────────────────────────
# A5 — Encoded payload heuristic
# ─────────────────────────────────────────────────────────────────────
# Reject if body contains a contiguous run of base64-ish or hex-ish
# characters longer than the threshold. Threshold chosen to avoid
# false-positives on git hashes (40 chars), UUIDs (32-36), small
# SHA-256 hashes pasted as evidence. Tunable.
_ENCODED_PAYLOAD_THRESHOLD = 200
_BASE64_RUN_RE = re.compile(r"[A-Za-z0-9+/=]{%d,}" % _ENCODED_PAYLOAD_THRESHOLD)
_HEX_RUN_RE = re.compile(r"[0-9a-fA-F]{%d,}" % _ENCODED_PAYLOAD_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────
# Scanner-disable env knob
# ─────────────────────────────────────────────────────────────────────


def _disabled_scanners() -> frozenset[str]:
    """`YANTRIKDB_SKILLS_DISABLE_SCANNERS="A2,A5"` — operator escape
    hatch for known false-positives. Disabled scanners are logged in
    the audit trail by the caller (this module doesn't have its own
    logger to keep it pure-function)."""
    raw = os.environ.get("YANTRIKDB_SKILLS_DISABLE_SCANNERS", "")
    return frozenset(s.strip().upper() for s in raw.split(",") if s.strip())


def _urls_allowed() -> bool:
    v = os.environ.get("YANTRIKDB_SKILLS_ALLOW_URLS", "").strip().lower()
    return v in ("1", "true", "yes", "on")


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def scan_body(body: str, *, is_outcome_note: bool = False) -> None:
    """Run all enabled scanners over `body`. Raise `ValueError` on
    first rejection. The caller is responsible for re-raising as
    `ToolError` and for audit-logging the scanner decision.

    For outcome notes, the same scanners apply (D3) but the body is
    typically shorter — that's fine, the regexes scale.
    """
    disabled = _disabled_scanners()

    # A1 — prompt-injection markers
    if "A1" not in disabled:
        m = _PROMPT_INJECTION_RE.search(body)
        if m:
            raise ValueError(
                f"[A1] body contains a prompt-injection marker "
                f"({m.group(0)[:40]!r}). If this is legitimate content "
                f"that hit a false positive, disable via "
                f"YANTRIKDB_SKILLS_DISABLE_SCANNERS=A1 (audited)."
            )

    # A2 — credential patterns
    if "A2" not in disabled:
        for name, regex in _CREDENTIAL_RES.items():
            if regex.search(body):
                raise ValueError(
                    f"[A2] body appears to contain a credential ({name}). "
                    f"Never store secrets in shared skills. If this is a "
                    f"false positive disable via "
                    f"YANTRIKDB_SKILLS_DISABLE_SCANNERS=A2 (audited)."
                )

    # A3 — URLs / IPv4 literals (off by default for outcome notes too)
    if "A3" not in disabled and not _urls_allowed():
        m = _URL_RE.search(body) or _IPV4_RE.search(body)
        if m:
            raise ValueError(
                f"[A3] body contains a URL or IP literal ({m.group(0)[:60]!r}). "
                f"Skills with embedded URLs are an exfil path for future "
                f"agents. Set YANTRIKDB_SKILLS_ALLOW_URLS=true if your "
                f"workflow legitimately needs them (audited)."
            )

    # A4 — unicode evasion
    if "A4" not in disabled:
        for ch in body:
            cat = unicodedata.category(ch)
            if cat in ("Cf", "Cs", "Cn") and ch not in _UNICODE_ALLOWED_CF:
                raise ValueError(
                    f"[A4] body contains a non-printing unicode char "
                    f"(U+{ord(ch):04X}, category {cat}). These are commonly "
                    f"used to hide content from human reviewers. Strip "
                    f"the char before retrying or disable via "
                    f"YANTRIKDB_SKILLS_DISABLE_SCANNERS=A4."
                )

    # A5 — long encoded-payload runs
    if "A5" not in disabled:
        # Skip A5 entirely for outcome notes — they're short by D3 and
        # the threshold rarely triggers anyway.
        if not is_outcome_note:
            m = _BASE64_RUN_RE.search(body) or _HEX_RUN_RE.search(body)
            if m:
                raise ValueError(
                    f"[A5] body contains a {len(m.group(0))}-char run of "
                    f"base64/hex characters — likely an encoded payload. "
                    f"If this is legitimate (e.g. a config example), split "
                    f"into smaller fragments or disable via "
                    f"YANTRIKDB_SKILLS_DISABLE_SCANNERS=A5."
                )


def scanner_report(body: str) -> dict:
    """Return which scanners flagged the body (without raising). Used
    by the audit log so we can record "rejected by [A1, A2]" rather
    than just first-match."""
    flags = {}
    for code, fn in (
        ("A1", lambda b: _PROMPT_INJECTION_RE.search(b)),
        ("A2", lambda b: any(r.search(b) for r in _CREDENTIAL_RES.values())),
        ("A3", lambda b: not _urls_allowed() and (_URL_RE.search(b) or _IPV4_RE.search(b))),
        ("A4", lambda b: any(
            unicodedata.category(c) in ("Cf", "Cs", "Cn") and c not in _UNICODE_ALLOWED_CF
            for c in b
        )),
        ("A5", lambda b: _BASE64_RUN_RE.search(b) or _HEX_RUN_RE.search(b)),
    ):
        try:
            flags[code] = bool(fn(body))
        except Exception:
            flags[code] = False
    return flags
