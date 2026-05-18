"""Client-side schema validation for the skill substrate.

Mirrors the validation that yantrikdb-server applies at `/v1/skills/define`
and that yantrikdb-hermes-plugin applies in its `embedded.py`. Lifted from
hermes-plugin v0.3.0+ so the load-bearing `applies_to` regex (hyphen-vs-
underscore drift) stays consistent across every consumer that writes to
`skill_substrate`.

These functions raise `ValueError` so the caller can decide how to surface
them — in `tools.py` they get re-raised as `ToolError` (MCP 4xx-equivalent,
no retry).
"""

from __future__ import annotations

import re

# Lowercase, dot-separated segments: e.g. workflow.git.commit_clean
SKILL_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")

# Lowercase + digits + underscores ONLY — no hyphens, no dots, no spaces.
# Load-bearing: there's a regression test on hermes-plugin pinning this
# exact shape because hyphen drift broke the substrate previously.
APPLIES_TO_RE = re.compile(r"^[a-z][a-z0-9_]*$")

SKILL_TYPES = frozenset({"procedure", "reference", "lesson", "pattern", "rule"})

SKILL_BODY_MIN = 50
SKILL_BODY_MAX = 5000
SKILL_ID_MIN = 4
SKILL_ID_MAX = 200
APPLIES_TO_MAX = 10

SKILL_NAMESPACE = "skill_substrate"
OUTCOME_NAMESPACE = "outcome_substrate"


def validate_skill_define_args(
    skill_id: str,
    body: str,
    skill_type: str,
    applies_to: list[str],
) -> None:
    """Raise `ValueError` if any field violates the schema.

    Match `yantrikdb-server`'s `/v1/skills/define` validation and
    `yantrikdb-hermes-plugin`'s `embedded.validate_skill_define_args` so
    the substrate stays consistent across consumers.
    """
    # skill_id
    if not isinstance(skill_id, str):
        raise ValueError("skill_id must be a string")
    if not (SKILL_ID_MIN <= len(skill_id) <= SKILL_ID_MAX):
        raise ValueError(
            f"skill_id length must be {SKILL_ID_MIN}..{SKILL_ID_MAX} chars; got {len(skill_id)}"
        )
    if not SKILL_ID_RE.fullmatch(skill_id):
        raise ValueError(
            f"skill_id {skill_id!r} must match {SKILL_ID_RE.pattern} "
            "(lowercase, dot-separated segments; e.g. 'workflow.git.commit_clean')"
        )

    # body
    if not isinstance(body, str):
        raise ValueError("body must be a string")
    if not (SKILL_BODY_MIN <= len(body) <= SKILL_BODY_MAX):
        raise ValueError(
            f"body length must be {SKILL_BODY_MIN}..{SKILL_BODY_MAX} chars; got {len(body)}"
        )

    # skill_type
    if skill_type not in SKILL_TYPES:
        raise ValueError(
            f"skill_type {skill_type!r} not in {sorted(SKILL_TYPES)}"
        )

    # applies_to — load-bearing regex
    if not isinstance(applies_to, list) or not applies_to:
        raise ValueError("applies_to must be a non-empty list of identifiers")
    if len(applies_to) > APPLIES_TO_MAX:
        raise ValueError(
            f"applies_to may contain at most {APPLIES_TO_MAX} entries; got {len(applies_to)}"
        )
    for entry in applies_to:
        if not isinstance(entry, str) or not APPLIES_TO_RE.fullmatch(entry):
            raise ValueError(
                f"applies_to entry {entry!r} must match {APPLIES_TO_RE.pattern} "
                "(lowercase + digits + underscores ONLY — no hyphens, no dots, no spaces)"
            )


def validate_skill_id(skill_id: str) -> None:
    """Lightweight check for `surface/get/outcome` paths — full schema
    isn't relevant when we're just referencing an existing skill."""
    if not isinstance(skill_id, str):
        raise ValueError("skill_id must be a string")
    if not (SKILL_ID_MIN <= len(skill_id) <= SKILL_ID_MAX):
        raise ValueError(
            f"skill_id length must be {SKILL_ID_MIN}..{SKILL_ID_MAX} chars; got {len(skill_id)}"
        )
    if not SKILL_ID_RE.fullmatch(skill_id):
        raise ValueError(
            f"skill_id {skill_id!r} must match {SKILL_ID_RE.pattern}"
        )
