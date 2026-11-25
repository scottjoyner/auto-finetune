"""Ephemeral signing keys, private temp-dir SQLite only; no prod or exports."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from src import lfm3_release_receipt_verifier as verifier
from src import lfm3_release_replay_ledger as ledger
from src import model_export

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 29, 5, 0, 0, tzinfo=timezone.utc)
KID = "test-only-no-prod-key"


@pytest.fixture
def keys():
    private = Ed25519PrivateKey.generate()
    return private, {KID: private.public_key()}


@pytest.fixture
def artifact_request():
    sha = lambda label: hashlib.sha256(label.encode()).hexdigest()
    return {
        "base_model_repo": "LiquidAI/LFM3-1.7B",
        "base_model_revision": "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        "base_model_sha256": sha("synthetic-weights"),
        "adapter_sha256": sha("synthetic-adapter"),
        "training_code_sha256": sha("synthetic-source"),
        "heldout_dataset_sha256": sha("synthetic-holdout"),
        "evaluation_receipt_sha256": sha("synthetic-scores"),
        "output_format": "gguf", "quantization": "q8_0",
        "output_name": "test-model.gguf",
    }


def private_dir(tmp_path):
    folder = tmp_path / "private-ledger"
    folder.mkdir(mode=0o700)
    folder.chmod(0o700)
    return folder


def signed(private, artifact_request, release_id="1" * 32):
    manifest = dict(artifact_request)
    manifest.update({
        "release_id": release_id,
        "issued_at": "2026-10-29T04:59:00Z",
        "expires_at": "2026-10-29T05:59:00Z",
    })
    signature = private.sign(verifier.signing_bytes(KID, manifest))
    return json.dumps({
        "schema": verifier.SCHEMA, "key_id": KID,
        "manifest": manifest,
        "signature_b64": base64.b64encode(signature).decode(),
    }, sort_keys=True, separators=(",", ":")).encode()


def claim(raw, keys, artifact_request, folder, **overrides):
    opts = {
        "trusted_public_keys": keys,
        "expected_request": artifact_request,
        "ledger_directory": folder,
        "now_utc": NOW,
    }
    opts.update(overrides)
    return ledger.claim_verified_release_evidence(raw, **opts)


def test_signature_valid_and_claim_durable_across_calls_and_restart(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    raw = signed(priv, artifact_request)
    first = claim(raw, trust, artifact_request, folder)
    assert first.status == "local_claim_recorded_non_authorizing"
    assert first.signature_verified
    assert first.claim_committed and first.local_replay_claimed
    assert not first.fleet_wide_replay_protected
    assert not first.export_permitted and not first.promotion_permitted
    again = claim(raw, trust, artifact_request, folder)
    assert again.status == "blocked"
    assert again.reason == "release_id_already_claimed"
    conn = sqlite3.connect(folder / ledger.DB_NAME)
    assert conn.execute("SELECT count(*) FROM release_claims").fetchone() == (1,)
    assert conn.execute("SELECT release_id FROM release_claims").fetchone()[0] == "1" * 32
    conn.close()


def test_existing_db_not_recreated_or_overwritten(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    assert claim(signed(priv, artifact_request), trust, artifact_request, folder).claim_committed
    db = folder / ledger.DB_NAME
    first_bytes = db.read_bytes()
    assert claim(signed(priv, artifact_request), trust, artifact_request, folder).status == "blocked"
    assert db.read_bytes() == first_bytes


def test_atomic_simultaneous_threads_only_one_winner(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    raw = signed(priv, artifact_request)
    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(lambda _: claim(raw, trust, artifact_request, folder), range(12)))
    assert sum(r.claim_committed for r in results) == 1
    assert sum(r.reason == "release_id_already_claimed" for r in results) == 11
    assert all(not r.export_permitted and not r.promotion_permitted for r in results)


def test_distinct_release_ids_can_be_recorded_without_approval(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    results = [claim(signed(priv, artifact_request, f"{i:032x}"), trust, artifact_request, folder)
               for i in range(1, 15)]
    assert all(r.claim_committed for r in results)
    assert all(not r.export_permitted for r in results)
    conn = sqlite3.connect(folder / ledger.DB_NAME)
    assert conn.execute("SELECT count(*) FROM release_claims").fetchone()[0] == 14
    conn.close()


def test_no_default_trusted_keys_or_ledger_path(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    raw = signed(priv, artifact_request)
    assert not claim(raw, {}, artifact_request, folder).claim_committed
    assert not (folder / ledger.DB_NAME).exists()
    missing = tmp_path / "not-created"
    assert claim(raw, trust, artifact_request, missing).status == "blocked"
    assert not missing.exists()


def test_signature_forgery_never_creates_db(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    raw = signed(Ed25519PrivateKey.generate(), artifact_request)
    result = claim(raw, trust, artifact_request, folder)
    assert not result.signature_verified and not result.claim_committed
    assert not (folder / ledger.DB_NAME).exists()


@pytest.mark.parametrize("field", ["adapter_sha256", "output_format",
                                     "evaluation_receipt_sha256", "output_name"])
def test_mismatched_request_never_creates_db(tmp_path, keys, artifact_request, field):
    priv, trust = keys
    folder = private_dir(tmp_path)
    raw = signed(priv, artifact_request)
    tampered = dict(artifact_request)
    if field.endswith("_sha256"):
        tampered[field] = "e" * 64
    elif field == "output_format":
        tampered["output_format"] = "onnx"
        tampered["quantization"] = "none"
    else:
        tampered[field] = "other.gguf"
    result = claim(raw, trust, tampered, folder)
    assert not result.claim_committed
    assert not (folder / ledger.DB_NAME).exists()


def test_bad_directory_permissions_refused(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    folder.chmod(0o755)
    result = claim(signed(priv, artifact_request), trust, artifact_request, folder)
    assert result.status == "blocked"
    assert not (folder / ledger.DB_NAME).exists()


def test_symlink_directory_refused(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(folder, target_is_directory=True)
    result = claim(signed(priv, artifact_request), trust, artifact_request, alias)
    assert result.status == "blocked"
    assert not (folder / ledger.DB_NAME).exists()


def test_relative_ledger_location_refused(tmp_path, keys, artifact_request):
    priv, trust = keys
    result = claim(signed(priv, artifact_request), trust, artifact_request, Path("relative/ledger"))
    assert result.status == "blocked"
    assert not (tmp_path / "relative").exists()


def test_symlink_db_file_refused_and_target_not_modified(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    target = tmp_path / "target"
    target.write_bytes(b"PRESERVE")
    (folder / ledger.DB_NAME).symlink_to(target)
    result = claim(signed(priv, artifact_request), trust, artifact_request, folder)
    assert result.status == "blocked"
    assert target.read_bytes() == b"PRESERVE"


def test_hardlinked_db_file_refused(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    source = tmp_path / "source"
    source.write_bytes(b"NO_WRITE")
    os.link(source, folder / ledger.DB_NAME)
    result = claim(signed(priv, artifact_request), trust, artifact_request, folder)
    assert result.status == "blocked"
    assert source.read_bytes() == b"NO_WRITE"


def test_directory_as_db_file_refused(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    (folder / ledger.DB_NAME).mkdir()
    result = claim(signed(priv, artifact_request), trust, artifact_request, folder)
    assert result.status == "blocked"


def test_corrupt_sqlite_file_fails_closed_and_is_preserved(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    db = folder / ledger.DB_NAME
    db.write_bytes(b"NOT SQLITE")
    result = claim(signed(priv, artifact_request), trust, artifact_request, folder)
    assert result.status == "blocked"
    assert db.read_bytes() == b"NOT SQLITE"


def test_incompatible_sqlite_schema_fails_closed(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    db = folder / ledger.DB_NAME
    with sqlite3.connect(db) as c:
        c.execute("CREATE TABLE release_claims (release_id TEXT PRIMARY KEY)")
    result = claim(signed(priv, artifact_request), trust, artifact_request, folder)
    assert result.status == "blocked"
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM release_claims").fetchone()[0] == 0


def test_replay_id_unique_even_if_signature_and_timestamp_differ(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    first = signed(priv, artifact_request)
    assert claim(first, trust, artifact_request, folder).claim_committed
    different = signed(priv, artifact_request)
    assert claim(different, trust, artifact_request, folder).reason == "release_id_already_claimed"


def test_ledger_database_mode_private(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    assert claim(signed(priv, artifact_request), trust, artifact_request, folder).claim_committed
    assert ((folder / ledger.DB_NAME).stat().st_mode & 0o777) == 0o600


def test_public_receipt_redacted_and_immutable(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    raw = signed(priv, artifact_request)
    out = claim(raw, trust, artifact_request, folder)
    plain = json.dumps(out.as_public_dict())
    assert str(folder) not in plain
    assert "LiquidAI" not in plain
    assert artifact_request["adapter_sha256"] not in plain
    assert "1" * 32 not in plain
    assert not out.export_permitted
    with pytest.raises(AttributeError):
        out.export_permitted = True


@pytest.mark.parametrize("invalid", [None, 8, "notbytes", b"", b"\xff", b"{}"])
def test_invalid_evidence_never_initializes_ledger(tmp_path, keys, artifact_request, invalid):
    priv, trust = keys
    folder = private_dir(tmp_path)
    result = claim(invalid, trust, artifact_request, folder)
    assert not result.claim_committed
    assert not (folder / ledger.DB_NAME).exists()


def test_export_still_blocked_after_replay_claim(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    result = claim(signed(priv, artifact_request), trust, artifact_request, folder)
    assert result.claim_committed
    output = tmp_path / "model.gguf"
    assert not model_export.export_gguf("synthetic", str(output)).success
    assert not model_export.export_merged("synthetic", str(output)).success
    assert not model_export.export_onnx("synthetic", str(output)).success
    assert not output.exists()


def test_inprogress_uncommitted_transaction_recovers(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    # Create schema with a different ID.
    assert claim(signed(priv, artifact_request, "a" * 32), trust, artifact_request, folder).claim_committed
    script = """
import os,sqlite3,sys
c=sqlite3.connect(sys.argv[1],isolation_level=None)
c.execute("PRAGMA journal_mode=DELETE")
c.execute("PRAGMA synchronous=FULL")
c.execute("BEGIN IMMEDIATE")
c.execute("INSERT INTO release_claims VALUES (?,?,?)", ("b"*32,"c"*64,"synthetic"))
os._exit(99)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(folder / ledger.DB_NAME)],
        capture_output=True, timeout=8,
    )
    assert child.returncode == 99
    # SQLite rolls back the uncommitted insert, not a false replay denial.
    r = claim(signed(priv, artifact_request, "b" * 32), trust, artifact_request, folder)
    assert r.claim_committed
    with sqlite3.connect(folder / ledger.DB_NAME) as c:
        assert c.execute("SELECT COUNT(*) FROM release_claims").fetchone()[0] == 2


def test_interprocess_simultaneous_writers_one_winner(tmp_path, keys, artifact_request):
    priv, trust = keys
    folder = private_dir(tmp_path)
    raw = signed(priv, artifact_request)
    pub = trust[KID].public_bytes(Encoding.Raw, PublicFormat.Raw)
    # Each child sees only an ephemeral public key, never a signing secret.
    script = """
import base64,json,sys
from datetime import datetime,timezone
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from src.lfm3_release_replay_ledger import claim_verified_release_evidence
raw=base64.b64decode(sys.argv[1])
artifact_request=json.loads(base64.b64decode(sys.argv[2]))
public=Ed25519PublicKey.from_public_bytes(base64.b64decode(sys.argv[3]))
result=claim_verified_release_evidence(raw,trusted_public_keys={"test-only-no-prod-key":public},
    expected_request=artifact_request,ledger_directory=sys.argv[4],
    now_utc=datetime(2026,10,29,5,0,0,tzinfo=timezone.utc),busy_timeout_ms=4000)
print(result.status)
"""
    args = [
        sys.executable, "-c", script,
        base64.b64encode(raw).decode(),
        base64.b64encode(json.dumps(artifact_request).encode()).decode(),
        base64.b64encode(pub).decode(),
        str(folder),
    ]
    children = [
        subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
        for _ in range(6)
    ]
    results = []
    for child in children:
        stdout, stderr = child.communicate(timeout=12)
        assert child.returncode == 0, stderr
        results.append(stdout.strip())
    assert results.count("local_claim_recorded_non_authorizing") == 1
    assert results.count("blocked") == 5
