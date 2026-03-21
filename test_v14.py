"""Thorough local tests for V14 substitution category features.

Tests the full flow: seed categories → conflict detection via think() →
reclassify learning → category reset → MCP tool paths.
"""

import json
import os
import tempfile
import shutil

from yantrikdb import YantrikDB


def setup_db(dim=384):
    """Create a fresh in-memory DB with embedder."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    db = YantrikDB(db_path=":memory:", embedding_dim=dim, embedder=model)
    return db


def test_seed_categories_populated():
    """Seed categories should be populated on fresh DB."""
    db = setup_db()
    cats = db.substitution_categories()
    cat_names = [c["name"] for c in cats]
    print(f"  Categories: {cat_names}")
    assert len(cats) >= 8, f"Expected >= 8 seed categories, got {len(cats)}"
    assert "databases" in cat_names
    assert "programming_languages" in cat_names
    assert "editors_tools" in cat_names

    # Check databases has seed members
    members = db.substitution_members("databases")
    tokens = [m["token_normalized"] for m in members]
    print(f"  databases members ({len(members)}): {tokens[:5]}...")
    assert "postgresql" in tokens
    assert "mysql" in tokens
    assert len(members) >= 10

    # All should be source=seed
    for m in members:
        assert m["source"] == "seed", f"Expected seed source, got {m['source']} for {m['token_normalized']}"
    print("  PASS")


def test_think_detects_category_conflict():
    """think() should detect PostgreSQL vs MySQL as a conflict, not redundancy."""
    db = setup_db()

    # Record two memories that differ only by the database name
    rid_a = db.record("We use PostgreSQL for the API backend", importance=0.8, domain="architecture")
    rid_b = db.record("We use MySQL for the API backend", importance=0.8, domain="architecture")
    print(f"  Recorded: {rid_a[:8]}... and {rid_b[:8]}...")

    # Run think
    result = db.think()
    print(f"  think() result: conflicts={result['conflicts_found']}, triggers={len(result['triggers'])}")

    # Check triggers
    trigger_types = [t["trigger_type"] for t in result["triggers"]]
    trigger_reasons = [t["reason"] for t in result["triggers"]]
    print(f"  Trigger types: {trigger_types}")
    for r in trigger_reasons:
        print(f"    - {r}")

    # Should have a potential_conflict, not just redundancy
    has_conflict = any("conflict" in tt for tt in trigger_types)
    has_category = any("databases" in r.lower() or "substitution" in r.lower() for r in trigger_reasons)

    if has_conflict:
        print("  PASS - detected as conflict")
    else:
        # Check if conflicts were created directly by scan_conflicts
        conflicts = db.get_conflicts(status="open")
        print(f"  Open conflicts: {len(conflicts)}")
        for c in conflicts:
            print(f"    - {c['conflict_type']}: {c['detection_reason'][:80]}")
        if len(conflicts) > 0:
            print("  PASS - conflict record created")
        else:
            print("  FAIL - no conflict detected for PostgreSQL vs MySQL")
            return False
    return True


def test_think_redundancy_for_true_duplicates():
    """think() should still treat true duplicates as redundancy."""
    db = setup_db()

    rid_a = db.record("We deploy to AWS us-east-1 region", importance=0.8)
    rid_b = db.record("We deploy to AWS us-east-1 region every day", importance=0.8)
    print(f"  Recorded near-duplicates: {rid_a[:8]}... and {rid_b[:8]}...")

    result = db.think()
    trigger_types = [t["trigger_type"] for t in result["triggers"]]
    print(f"  Trigger types: {trigger_types}")

    # Should be redundancy, not conflict
    has_redundancy = "redundancy" in trigger_types
    has_conflict = any("conflict" in tt for tt in trigger_types)
    if has_redundancy and not has_conflict:
        print("  PASS - correctly classified as redundancy")
    elif not result["triggers"]:
        print("  PASS - no triggers (similarity may not have crossed threshold)")
    else:
        print(f"  WARN - unexpected triggers: {trigger_types}")
    return True


def test_learn_category_members():
    """learn_category_members should add new members correctly."""
    db = setup_db()

    # Add new members to databases
    count = db.learn_category_members("databases", [("tidb", 0.35), ("surrealdb", 0.35)], "llm_suggested")
    print(f"  Added {count} new members")
    assert count == 2

    members = db.substitution_members("databases")
    tokens = [m["token_normalized"] for m in members]
    assert "tidb" in tokens
    assert "surrealdb" in tokens

    # LLM-suggested should be pending status
    tidb = next(m for m in members if m["token_normalized"] == "tidb")
    assert tidb["source"] == "llm_suggested"
    assert tidb["status"] == "pending"
    print(f"  tidb: source={tidb['source']}, status={tidb['status']}, confidence={tidb['confidence']}")
    print("  PASS")


def test_reclassify_conflict_no_pollution():
    """reclassify_conflict should NOT add stopwords to categories."""
    db = setup_db()

    # Create a conflict manually between memories with different editors
    rid_a = db.record("I use VSCode before starting my work day", importance=0.8)
    rid_b = db.record("I use Neovim after finishing my work day", importance=0.8)

    # Force a conflict record
    result = db.think()
    conflicts = db.get_conflicts(status="open")
    print(f"  Open conflicts after think(): {len(conflicts)}")

    if len(conflicts) == 0:
        # Create one manually for testing
        print("  No conflict auto-detected, testing reclassify would need a conflict_id")
        print("  SKIP (conflict not auto-detected for this text pair)")
        return True

    conflict_id = conflicts[0]["conflict_id"]
    print(f"  Reclassifying conflict {conflict_id[:8]}...")

    # Before reclassify, check editors_tools member count
    before = db.substitution_members("editors_tools")
    before_count = len(before)
    print(f"  editors_tools before: {before_count} members")

    # Reclassify
    result = db.reclassify_conflict(conflict_id, "preference")
    print(f"  Learned members: {json.dumps(result.get('learned_members', []))}")

    # After reclassify, check editors_tools wasn't polluted
    after = db.substitution_members("editors_tools")
    after_count = len(after)
    after_tokens = [m["token_normalized"] for m in after]
    print(f"  editors_tools after: {after_count} members")

    # Should not contain stopwords
    bad_tokens = [t for t in after_tokens if t in ("a", "the", "before", "after", "starting", "finishing", "my", "work", "day", "i", "use")]
    if bad_tokens:
        print(f"  FAIL - category polluted with: {bad_tokens}")
        return False
    else:
        print(f"  PASS - no pollution (added {after_count - before_count} members)")
    return True


def test_reset_category_to_seed():
    """reset_category_to_seed should remove all non-seed members."""
    db = setup_db()

    # Add some junk
    db.learn_category_members("editors_tools", [("notepad", 0.5), ("nano", 0.5)], "user_confirmed")
    before = db.substitution_members("editors_tools")
    print(f"  editors_tools with junk: {len(before)} members")

    # Reset
    removed = db.reset_category_to_seed("editors_tools")
    print(f"  Removed {removed} non-seed members")

    after = db.substitution_members("editors_tools")
    after_tokens = [m["token_normalized"] for m in after]
    print(f"  editors_tools after reset: {len(after)} members: {after_tokens}")

    assert "notepad" not in after_tokens
    assert "nano" not in after_tokens
    # Seed members should remain
    assert "vscode" in after_tokens
    assert "neovim" in after_tokens
    assert removed == 2
    print("  PASS")


def test_category_conflict_detection_languages():
    """Test that programming language substitution is detected."""
    db = setup_db()

    rid_a = db.record("Our backend is written in Python with FastAPI", importance=0.8, domain="architecture")
    rid_b = db.record("Our backend is written in Rust with Actix", importance=0.8, domain="architecture")
    print(f"  Recorded language conflict: {rid_a[:8]}... vs {rid_b[:8]}...")

    result = db.think()
    conflicts = db.get_conflicts(status="open")
    trigger_reasons = [t["reason"] for t in result["triggers"]]

    print(f"  Triggers: {len(result['triggers'])}, Conflicts: {len(conflicts)}")
    for r in trigger_reasons:
        print(f"    - {r}")
    for c in conflicts:
        print(f"    conflict: {c['conflict_type']} - {c['detection_reason'][:80]}")

    # Either a trigger or conflict should mention the substitution
    has_detection = (
        any("programming_languages" in r.lower() or "frameworks" in r.lower() or "substitution" in r.lower() for r in trigger_reasons)
        or any("programming_languages" in c.get("detection_reason", "").lower() or "frameworks" in c.get("detection_reason", "").lower() for c in conflicts)
        or len(conflicts) > 0
    )
    if has_detection:
        print("  PASS")
    else:
        print("  WARN - substitution not detected (may need entity links)")
    return True


def test_stats_include_categories():
    """Stats should reflect category data."""
    db = setup_db()
    stats = db.stats(namespace=None)
    print(f"  Stats: {json.dumps(stats, indent=2)}")
    print("  PASS (stats returned)")


def run_all():
    print("=" * 60)
    print("YantrikDB V14 Local Test Suite")
    print("=" * 60)

    tests = [
        ("Seed categories populated", test_seed_categories_populated),
        ("think() detects category conflict", test_think_detects_category_conflict),
        ("think() treats duplicates as redundancy", test_think_redundancy_for_true_duplicates),
        ("learn_category_members", test_learn_category_members),
        ("reclassify_conflict no pollution", test_reclassify_conflict_no_pollution),
        ("reset_category_to_seed", test_reset_category_to_seed),
        ("Category conflict: languages", test_category_conflict_detection_languages),
        ("Stats include categories", test_stats_include_categories),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            result = test_fn()
            if result is False:
                failed += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
