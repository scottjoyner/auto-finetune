"""External trust-store verification tests use ephemeral, never-exported keys."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from src.lfm3_sensorgraft_receipt import (
    SIGNED_SCHEMA,
    draft_custody,
    signed_message,
)
from src.lfm3_sensorgraft_verify import verify_files

FOLDER = Path(__file__).resolve().parents[1] / "research/evidence/lfm3-real-lora-20261029"


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path):
    ed = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    ser = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    report = json.loads((FOLDER / "lfm3-real-lora-3seed-20step-equal-lr-20261029.json").read_text())
    review = json.loads((FOLDER / "paired-review.json").read_text())
    payload = draft_custody(report, review)
    key = ed.Ed25519PrivateKey.generate()
    key_public = key.public_key().public_bytes(
        encoding=ser.Encoding.Raw, format=ser.PublicFormat.Raw
    )
    envelope = {
        "schema": SIGNED_SCHEMA,
        "payload": payload,
        "key_id": "ephemeral-test",
        "signature_base64": base64.b64encode(
            key.sign(signed_message("ephemeral-test", payload))
        ).decode(),
    }
    paths = {
        k: tmp_path / f"{k}.json" for k in ("report", "review", "envelope", "trust", "revocations")
    }
    put(paths["report"], report)
    put(paths["review"], review)
    put(paths["envelope"], envelope)
    trust_sha = put(
        paths["trust"],
        {
            "schema": "sensorgraft.ed25519-trust.v1",
            "keys": {"ephemeral-test": base64.b64encode(key_public).decode()},
        },
    )
    rev_sha = put(
        paths["revocations"],
        {
            "schema": "sensorgraft.ed25519-revocations.v1",
            "revoked_key_ids": [],
            "revoked_receipt_sha256": [],
        },
    )
    return paths, trust_sha, rev_sha


def check(setup, trust_pin=None, rev_pin=None):
    paths, t, r = setup
    return verify_files(
        paths["report"],
        paths["review"],
        paths["envelope"],
        paths["trust"],
        paths["revocations"],
        trust_sha256=t if trust_pin is None else trust_pin,
        revocations_sha256=r if rev_pin is None else rev_pin,
    )


def test_good_ephemeral_verified_with_out_of_band_pins(setup):
    result = check(setup)
    assert result["status"] == "authenticated_receipt"
    assert result["independent_trust_pin_verified"] is True
    assert result["independent_revocation_pin_verified"] is True
    assert result["revocation_freshness_verified"] is False
    assert result["weight_promotion_permitted"] is False


def test_wrong_external_trust_digest_blocks(setup):
    assert check(setup, trust_pin="f" * 64)["status"] == "blocked"


def test_unpinned_trust_store_blocks(setup):
    assert check(setup, trust_pin="unpinned")["status"] == "blocked"


def test_wrong_revocations_digest_blocks(setup):
    assert check(setup, rev_pin="0" * 64)["status"] == "blocked"


def test_revoked_key_blocks_with_new_pin(setup):
    paths, t, _ = setup
    r = put(
        paths["revocations"],
        {
            "schema": "sensorgraft.ed25519-revocations.v1",
            "revoked_key_ids": ["ephemeral-test"],
            "revoked_receipt_sha256": [],
        },
    )
    assert check((paths, t, r))["reason"] == "revoked_signer"


def test_invalid_public_key_length_blocks_with_new_pin(setup):
    paths, _, r = setup
    t = put(
        paths["trust"],
        {
            "schema": "sensorgraft.ed25519-trust.v1",
            "keys": {"ephemeral-test": base64.b64encode(b"bad").decode()},
        },
    )
    assert check((paths, t, r))["reason"] == "invalid_trusted_public_key_length"


def test_external_revocations_schema_fail_closed(setup):
    paths, t, _ = setup
    r = put(
        paths["revocations"],
        {
            "schema": "unexpected",
            "revoked_key_ids": [],
            "revoked_receipt_sha256": [],
        },
    )
    assert check((paths, t, r))["reason"] == "invalid_external_trust_or_revocation_schema"


def test_revocation_list_duplicate_blocks(setup):
    paths, t, _ = setup
    r = put(
        paths["revocations"],
        {
            "schema": "sensorgraft.ed25519-revocations.v1",
            "revoked_key_ids": ["x", "x"],
            "revoked_receipt_sha256": [],
        },
    )
    assert check((paths, t, r))["reason"] == "invalid_revocation_entries"


def test_signer_claim_in_envelope_cannot_reassign_signature(setup):
    paths, t, r = setup
    envelope = json.loads(paths["envelope"].read_text())
    envelope["key_id"] = "different"
    put(paths["envelope"], envelope)
    assert check((paths, t, r))["status"] == "blocked"


def test_tampered_report_data_blocks(setup):
    paths, t, r = setup
    report = json.loads(paths["report"].read_text())
    report["arms"][0]["events"][0]["grad_norm"] += 0.1
    put(paths["report"], report)
    assert check((paths, t, r))["status"] == "blocked"


def test_untrusted_signer_when_missing_from_trust_store(setup):
    paths, _, r = setup
    t = put(
        paths["trust"],
        {
            "schema": "sensorgraft.ed25519-trust.v1",
            "keys": {"another": base64.b64encode(bytes(range(32))).decode()},
        },
    )
    assert check((paths, t, r))["reason"] == "signer_not_independently_trusted"