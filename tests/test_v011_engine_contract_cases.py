"""v0.11 engine behavioral-contract cases — the PACK substrate.

Same GATING RULE as the v0.10 suite: every case gates on a FEATURE PROBE
(hasattr / signature / raised-type), never a version parse. On a pre-v0.11
engine these SKIP with a reason naming the missing surface; on v0.11 they RUN
and become the regression wall that lets pyproject widen the pin to <0.12.0.

Packs are signed, portable memory bundles. The behaviors pinned here were
learned empirically against yantrikdb 0.11.3 (not guessed):

  1. typed pack errors      — 3 classes, all RuntimeError subclasses
  2. keypair / pubkey       — generate_pack_keypair() -> (sk, pk); pubkey_of(sk)==pk
  3. seal -> sign -> manifest — sign flips manifest.signed True + sets pubkey
  4. install -> recall -> uninstall — pack rows recallable; uninstall zero-residue
  5. mount idempotency      — double-mount raises PackAlreadyMounted
  6. publisher trust        — trust/list/untrust round-trips
  7. mounted trust + tier   — mounted pack carries trust label + local-first
                              tier_multiplier < 1.0 (user-corrections-win at rank)
  8. embedder-identity gate — cross-embedder mount raises PackEmbedderMismatch
"""
from __future__ import annotations

import os
import tempfile

import pytest

import yantrikdb
from yantrikdb import YantrikDB


# ── feature probes (not version parses) ──────────────────────────────

HAS_PACKS = hasattr(YantrikDB, "install_pack") and hasattr(YantrikDB, "seal_pack")
HAS_PACK_ERRORS = hasattr(yantrikdb, "PackAlreadyMounted")

pytestmark = pytest.mark.skipif(
    not HAS_PACKS, reason="engine lacks the pack substrate — pre-v0.11 surface"
)


@pytest.fixture(scope="module")
def eng():
    """A bundled-embedder engine factory + a shared source db with two rows
    sealed into a signed pack. Returns (make_db, packfile, secret, pubkey)."""
    os.environ.setdefault("YANTRIKDB_EMBEDDER", "bundled")
    from yantrikdb_mcp.embedder import load_engine

    def make_db(name):
        return load_engine(os.path.join(tempfile.mkdtemp(), name), model_name="bundled")

    src = make_db("pack_src.db")
    src.record("Pack export: the deploy window opens at 04:00 UTC", namespace="packns")
    src.record("Pack export: the rate limit is 500 rps", namespace="packns")
    sk, pk = src.generate_pack_keypair()
    ident = src.embedder_identity()
    packfile = os.path.join(tempfile.mkdtemp(), "contract.ypack")
    src.seal_pack(
        packfile, "contract-pack", "1.0.0", "contract-origin", namespace="packns",
        embedder_name=ident["name"], embedder_digest=ident["digest"], embedder_dim=ident["dim"],
    )
    src.sign_pack(packfile, sk)
    yield make_db, packfile, sk, pk
    try:
        src.close()
    except AttributeError:
        pass


# ── 1. typed pack errors ─────────────────────────────────────────────


def test_pack_errors_are_runtime_subclasses():
    """The three pack errors live on the package root and subclass
    RuntimeError, so legacy `except RuntimeError` handlers keep catching them."""
    if not HAS_PACK_ERRORS:
        pytest.skip("engine lacks typed pack errors — pre-v0.11 surface")
    for name in ("PackAlreadyMounted", "PackEmbedderMismatch", "PackSignatureInvalid"):
        cls = getattr(yantrikdb, name, None)
        assert cls is not None, f"yantrikdb.{name} missing from package root"
        assert issubclass(cls, RuntimeError), f"{name} must subclass RuntimeError"


# ── 2. keypair / pubkey ──────────────────────────────────────────────


def test_keypair_and_pubkey_derivation(eng):
    make_db, _packfile, sk, pk = eng
    d = make_db("kp.db")
    sk2, pk2 = d.generate_pack_keypair()
    assert isinstance(sk2, str) and isinstance(pk2, str) and sk2 != pk2
    # pubkey_of is a pure derivation from the secret — deterministic.
    assert d.pubkey_of(sk) == pk, "pubkey_of(secret) must recover the public key"


# ── 3. seal -> sign -> manifest ──────────────────────────────────────


def test_seal_sign_manifest_roundtrip(eng):
    make_db, packfile, _sk, pk = eng
    d = make_db("man.db")
    man = d.read_pack_manifest(packfile)
    assert man["signed"] is True, "a signed pack's manifest must report signed=True"
    assert man["publisher_pubkey"] == pk, "manifest must carry the signer's pubkey"
    assert man["pack_id"] == "contract-origin@1.0.0"
    assert man["content_digest"].startswith("blake3:"), "content digest anchors integrity"
    assert man["corpus_rows"] == 2


# ── 4. install -> recall -> uninstall (zero-residue) ─────────────────


def test_install_recall_uninstall_zero_residue(eng):
    make_db, packfile, _sk, pk = eng
    d = make_db("con.db")
    d.trust_publisher(pk, label="contract")

    pack_id = d.install_pack(packfile)
    assert pack_id == "contract-origin@1.0.0"

    rows = d.recall(query="deploy window opens", top_k=5, namespace="packns")
    texts = " ".join(r.get("text", "") for r in (rows or []) if isinstance(r, dict))
    assert "04:00 UTC" in texts, "mounted pack rows must be recallable"

    before = d.stats().get("active_memories")
    assert d.uninstall_pack(pack_id), "uninstall must report success"
    after = d.stats().get("active_memories")
    assert after == before, (
        f"uninstall must be zero-residue: active_memories {before}->{after}. "
        "A pack you cannot cleanly remove is a pack you cannot trust to install."
    )


# ── 5. mount idempotency ─────────────────────────────────────────────


def test_double_mount_raises_already_mounted(eng):
    if not HAS_PACK_ERRORS:
        pytest.skip("engine lacks typed pack errors — pre-v0.11 surface")
    make_db, packfile, _sk, pk = eng
    d = make_db("mnt.db")
    d.trust_publisher(pk, label="contract")
    d.install_pack(packfile)  # install auto-mounts
    with pytest.raises(yantrikdb.PackAlreadyMounted):
        d.mount_pack(packfile)


# ── 6. publisher trust round-trip ────────────────────────────────────


def test_trust_publisher_roundtrip(eng):
    make_db, _packfile, _sk, pk = eng
    d = make_db("trust.db")
    d.trust_publisher(pk, label="acme")
    listed = {t["pubkey"]: t.get("label") for t in d.trusted_publishers()}
    assert listed.get(pk) == "acme", "trusted publisher must list with its label"
    d.untrust_publisher(pk)
    assert pk not in {t["pubkey"] for t in d.trusted_publishers()}, "untrust must remove"


# ── 7. mounted pack carries trust + local-first tier ─────────────────


def test_mounted_pack_is_downweighted_vs_local(eng):
    """A mounted pack's rows must rank BELOW equally-relevant local memories:
    the mounted entry exposes tier_multiplier < 1.0. This is the ranking-layer
    half of user-corrections-win — imported knowledge never outranks what the
    user recorded locally."""
    make_db, packfile, _sk, pk = eng
    d = make_db("tier.db")
    d.trust_publisher(pk, label="contract")
    d.install_pack(packfile)
    mounted = d.mounted_packs()
    assert mounted, "installed pack must appear in mounted_packs"
    entry = mounted[0]
    assert entry["trust"] == "signed", "a signed+trusted pack must report trust='signed'"
    assert 0.0 < entry["tier_multiplier"] < 1.0, (
        "pack memories must be down-weighted vs local (local-first ranking)"
    )


# ── 8. embedder-identity gate ────────────────────────────────────────


def test_cross_embedder_mount_is_refused(eng):
    """embedder_identity() is the cross-machine safety anchor: a pack sealed
    against a different embedding space cannot silently poison recall — mount
    raises PackEmbedderMismatch rather than returning plausible-but-meaningless
    hits."""
    if not HAS_PACK_ERRORS:
        pytest.skip("engine lacks typed pack errors — pre-v0.11 surface")
    make_db, _packfile, _sk, _pk = eng
    src = make_db("mm_src.db")
    src.record("mismatch probe row", namespace="mm")
    bad = os.path.join(tempfile.mkdtemp(), "bad.ypack")
    src.seal_pack(
        bad, "bad", "1.0.0", "acme", namespace="mm",
        embedder_name="totally-different", embedder_digest="blake3:deadbeef", embedder_dim=999,
    )
    d = make_db("mm_con.db")
    with pytest.raises(yantrikdb.PackEmbedderMismatch):
        d.mount_pack(bad)


def test_embedder_identity_is_adopted_lazily(eng):
    """Identity is None on a virgin db and adopts on first use — so a freshly
    created store carries no embedding fingerprint to seal into a pack until it
    has actually embedded something. adopt_embedder_identity() forces it."""
    make_db, _packfile, _sk, _pk = eng
    d = make_db("ident.db")
    assert d.embedder_identity() is None, "virgin db has no embedder identity yet"
    d.adopt_embedder_identity()
    ident = d.embedder_identity()
    assert set(("name", "digest", "dim")).issubset(ident.keys())
    assert ident["digest"].startswith("blake3:"), "embedder identity is digest-anchored"
    assert isinstance(ident["dim"], int) and ident["dim"] > 0
