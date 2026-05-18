"""Unit tests for the skill substrate validator.

Mirrors yantrikdb-hermes-plugin's schema test discipline — including the
load-bearing 'no hyphens in applies_to' regression case.
"""

from __future__ import annotations

import pytest

from yantrikdb_mcp.skill_validation import (
    APPLIES_TO_MAX,
    SKILL_BODY_MAX,
    SKILL_BODY_MIN,
    SKILL_ID_MAX,
    SKILL_ID_MIN,
    SKILL_TYPES,
    validate_skill_define_args,
    validate_skill_id,
)


# ─────────────────────────────────────────────────────────────────────
# validate_skill_define_args — happy path
# ─────────────────────────────────────────────────────────────────────


def test_define_minimal_valid():
    validate_skill_define_args(
        skill_id="workflow.git.commit_clean",
        body="x" * 100,
        skill_type="procedure",
        applies_to=["git"],
    )


def test_define_max_applies_to():
    validate_skill_define_args(
        skill_id="foo.bar",
        body="x" * 60,
        skill_type="rule",
        applies_to=[f"a{i}" for i in range(APPLIES_TO_MAX)],
    )


@pytest.mark.parametrize("kind", sorted(SKILL_TYPES))
def test_define_all_skill_types_accepted(kind: str):
    validate_skill_define_args("foo.bar", "x" * 60, kind, ["foo"])


# ─────────────────────────────────────────────────────────────────────
# validate_skill_define_args — skill_id rejections
# ─────────────────────────────────────────────────────────────────────


def test_define_skill_id_must_be_dotted():
    """No dot → reject. yantrikdb-server / hermes-plugin demand at least
    one segment after the leading lower-id."""
    with pytest.raises(ValueError, match="skill_id"):
        validate_skill_define_args("flat_id", "x" * 60, "procedure", ["foo"])


def test_define_skill_id_no_uppercase():
    with pytest.raises(ValueError, match="skill_id"):
        validate_skill_define_args("Foo.bar", "x" * 60, "procedure", ["foo"])


def test_define_skill_id_no_hyphens():
    with pytest.raises(ValueError, match="skill_id"):
        validate_skill_define_args("foo-bar.baz", "x" * 60, "procedure", ["foo"])


def test_define_skill_id_too_short():
    with pytest.raises(ValueError, match=f"{SKILL_ID_MIN}"):
        validate_skill_define_args("a.b", "x" * 60, "procedure", ["foo"])


def test_define_skill_id_too_long():
    with pytest.raises(ValueError, match=f"{SKILL_ID_MAX}"):
        validate_skill_define_args(
            "a." + "x" * SKILL_ID_MAX, "x" * 60, "procedure", ["foo"]
        )


# ─────────────────────────────────────────────────────────────────────
# validate_skill_define_args — body rejections
# ─────────────────────────────────────────────────────────────────────


def test_define_body_too_short():
    with pytest.raises(ValueError, match="body length"):
        validate_skill_define_args(
            "foo.bar", "x" * (SKILL_BODY_MIN - 1), "procedure", ["foo"]
        )


def test_define_body_too_long():
    with pytest.raises(ValueError, match="body length"):
        validate_skill_define_args(
            "foo.bar", "x" * (SKILL_BODY_MAX + 1), "procedure", ["foo"]
        )


def test_define_body_not_string():
    with pytest.raises(ValueError, match="body must be a string"):
        validate_skill_define_args("foo.bar", 12345, "procedure", ["foo"])  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────
# validate_skill_define_args — skill_type rejections
# ─────────────────────────────────────────────────────────────────────


def test_define_unknown_skill_type():
    with pytest.raises(ValueError, match="skill_type"):
        validate_skill_define_args("foo.bar", "x" * 60, "lore", ["foo"])


# ─────────────────────────────────────────────────────────────────────
# validate_skill_define_args — applies_to rejections
# ─────────────────────────────────────────────────────────────────────


def test_define_applies_to_must_be_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        validate_skill_define_args("foo.bar", "x" * 60, "procedure", [])


def test_define_applies_to_must_be_list():
    with pytest.raises(ValueError, match="non-empty"):
        validate_skill_define_args("foo.bar", "x" * 60, "procedure", "git")  # type: ignore[arg-type]


def test_define_applies_to_no_hyphens_LOAD_BEARING():
    """Load-bearing regression — hermes-plugin pins this exact rejection.
    A hyphen in `applies_to` breaks cross-consumer substrate consistency
    (server side validates it; we mirror server-side here)."""
    with pytest.raises(ValueError, match="applies_to entry"):
        validate_skill_define_args(
            "foo.bar", "x" * 60, "procedure", ["git-commit"]
        )


def test_define_applies_to_no_dots():
    with pytest.raises(ValueError, match="applies_to entry"):
        validate_skill_define_args(
            "foo.bar", "x" * 60, "procedure", ["git.commit"]
        )


def test_define_applies_to_no_spaces():
    with pytest.raises(ValueError, match="applies_to entry"):
        validate_skill_define_args(
            "foo.bar", "x" * 60, "procedure", ["git commit"]
        )


def test_define_applies_to_over_max():
    with pytest.raises(ValueError, match=f"{APPLIES_TO_MAX}"):
        validate_skill_define_args(
            "foo.bar", "x" * 60, "procedure",
            [f"x{i}" for i in range(APPLIES_TO_MAX + 1)],
        )


# ─────────────────────────────────────────────────────────────────────
# validate_skill_id (lightweight check)
# ─────────────────────────────────────────────────────────────────────


def test_validate_skill_id_passes_dotted():
    validate_skill_id("foo.bar.baz")


def test_validate_skill_id_rejects_flat():
    with pytest.raises(ValueError):
        validate_skill_id("flat")


def test_validate_skill_id_rejects_non_string():
    with pytest.raises(ValueError):
        validate_skill_id(42)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────
# Write gate — _skill_writes_enabled
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("val,expected", [
    (False, False),
    (True, True),
])
def test_skill_writes_gate(monkeypatch, val, expected):
    """C2: config is frozen at import. Tests poke the frozen CONFIG
    directly rather than mutating env, mirroring how the gate behaves
    in a real running MCP server (config changes require restart)."""
    from yantrikdb_mcp import skill_security
    from yantrikdb_mcp.tools import _skill_writes_enabled

    monkeypatch.setattr(skill_security.CONFIG, "writes_enabled", val)
    # Also clear any time-bound expiry so this test only exercises the bool
    monkeypatch.setattr(skill_security.CONFIG, "write_expires_at", None)
    assert _skill_writes_enabled() is expected


def test_skill_writes_gate_env_parse_truthy(monkeypatch):
    """Confirm the env-var parser accepts the documented truthy values
    when a fresh _Config() is constructed (i.e. simulating a startup)."""
    from yantrikdb_mcp import skill_security

    for val in ("1", "true", "TRUE", "yes", "on", " true "):
        monkeypatch.setenv("YANTRIKDB_SKILLS_WRITE_ENABLED", val)
        cfg = skill_security._Config()
        assert cfg.writes_enabled is True, f"failed for {val!r}"


def test_skill_writes_gate_env_parse_falsy(monkeypatch):
    """Confirm falsy / missing / garbage values keep the gate closed."""
    from yantrikdb_mcp import skill_security

    monkeypatch.delenv("YANTRIKDB_SKILLS_WRITE_ENABLED", raising=False)
    assert skill_security._Config().writes_enabled is False

    for val in ("", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("YANTRIKDB_SKILLS_WRITE_ENABLED", val)
        cfg = skill_security._Config()
        assert cfg.writes_enabled is False, f"unexpected open for {val!r}"
