"""Research receipt custody checks; no production signer keys are used."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_sensorgraft_receipt import (
    SIGNED_SCHEMA,
    _read_json,
    canonical,
    digest,
    draft_custody,
    signed_message,
    verify_signed_receipt,
)

FOLDER = Path(__file__).resolve().parents[1] / "research/evidence/lfm3-real-lora-20261029"


@pytest.fixture(scope="module")
def evidence():
    report = json.loads((FOLDER / "lfm3-real-lora-3seed-20step-equal-lr-20261029.json").read_text())
    decision = json.loads((FOLDER / "paired-review.json").read_text())
    return report, decision


@pytest.fixture
def draft(evidence):
    return draft_custody(*evidence)


@pytest.fixture
def signer():
    ed = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    private = ed.Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    return private, public


def signed(draft, signer):
    private, public = signer
    return {
        "schema": SIGNED_SCHEMA,
        "payload": draft,
        "key_id": "research-test-only",
        "signature_base64": base64.b64encode(
            private.sign(signed_message("research-test-only", draft))
        ).decode("ascii"),
    }, {"research-test-only": public}


def verified(envelope, evidence, trusted, *, revoked_keys=None, revoked_receipts=None):
    return verify_signed_receipt(
        envelope,
        *evidence,
        trusted_public_keys=trusted,
        revoked_key_ids=revoked_keys if revoked_keys is not None else set(),
        revoked_receipt_sha256=revoked_receipts if revoked_receipts is not None else set(),
    )


def test_draft_binds_every_record_and_defaults_to_unattested(evidence, draft):
    report, decision = evidence
    assert draft["schema"] == "sensorgraft.lfm3.experiment-custody.v1"
    assert draft["provenance"] == "producer-claimed-unattested"
    assert draft["report_sha256"] == digest(report)
    assert draft["review_sha256"] == digest(decision)
    assert draft["model_weights_sha256"] == report["base_sha256"]
    assert len(draft["arm_proofs"]) == 6
    assert sum(x["step_count"] for x in draft["arm_proofs"]) == 120
    assert not draft["sensor_graft_ingested"]
    assert not draft["weight_promotion_permitted"]
    assert draft == json.loads((FOLDER / "unsigned-sensorgraft-custody.json").read_text())


def test_canonical_form_deterministic(evidence, draft):
    assert canonical(draft) == canonical(dict(reversed(list(draft.items()))))
    assert len(digest(draft)) == 64


def test_valid_ephemeral_ed25519_research_signature(evidence, draft, signer):
    envelope, trusted = signed(draft, signer)
    result = verified(envelope, evidence, trusted)
    assert result["status"] == "authenticated_receipt"
    assert result["authenticated"] is True
    assert result["data_origin_independently_verified"] is False
    assert result["sensor_graft_ingested"] is False
    assert result["weight_promotion_permitted"] is False


def test_untrusted_signer_is_rejected(evidence, draft, signer):
    envelope, _ = signed(draft, signer)
    assert verified(envelope, evidence, {})["reason"] == "signer_not_independently_trusted"


def test_revoked_key_is_rejected(evidence, draft, signer):
    envelope, trusted = signed(draft, signer)
    result = verified(envelope, evidence, trusted, revoked_keys={"research-test-only"})
    assert result["reason"] == "revoked_signer"


def test_revoked_receipt_is_rejected(evidence, draft, signer):
    envelope, trusted = signed(draft, signer)
    result = verified(envelope, evidence, trusted, revoked_receipts={digest(draft)})
    assert result["reason"] == "revoked_experiment_receipt"


def test_wrong_public_key_is_rejected(evidence, draft, signer):
    envelope, _ = signed(draft, signer)
    another = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    ser = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    other = (
        another.Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(encoding=ser.Encoding.Raw, format=ser.PublicFormat.Raw)
    )
    assert verified(envelope, evidence, {"research-test-only": other})["status"] == "blocked"


def test_unsigned_receipt_cannot_be_authenticated(evidence, draft):
    assert verified(draft, evidence, {})["reason"] == "invalid_signed_envelope"


def test_missing_revocation_view_refused(evidence, draft, signer):
    envelope, trusted = signed(draft, signer)
    result = verify_signed_receipt(
        envelope,
        *evidence,
        trusted_public_keys=trusted,
        revoked_key_ids=None,
        revoked_receipt_sha256=set(),
    )
    assert result["reason"] == "independent_trust_or_revocations_missing"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["payload"].update({"weight_promotion_permitted": True}),
        lambda x: x["payload"].update({"sensor_graft_ingested": True}),
        lambda x: x["payload"]["arm_proofs"][0].update({"step_count": 21}),
        lambda x: x["payload"]["arm_proofs"][0].update({"events_sha256": "0" * 64}),
        lambda x: x["payload"].update({"extra": "override"}),
        lambda x: x.update({"key_id": "unrecognized-key"}),
        lambda x: x.update({"signature_base64": "@@@@"}),
        lambda x: x.update({"signature_base64": base64.b64encode(b"bad").decode()}),
        lambda x: x.update({"schema": "unsigned"}),
    ],
)
def test_signed_receipt_tampering_blocks(evidence, draft, signer, mutation):
    envelope, trusted = signed(draft, signer)
    mutation(envelope)
    result = verified(envelope, evidence, trusted)
    assert result["status"] == "blocked"
    assert result["weight_promotion_permitted"] is False


def test_training_result_tamper_blocks_even_with_untouched_signature(evidence, draft, signer):
    envelope, trusted = signed(draft, signer)
    report, decision = copy.deepcopy(evidence)
    report["arms"][0]["events"][0]["grad_norm"] += 0.1
    assert verified(envelope, (report, decision), trusted)["status"] == "blocked"


def test_review_tamper_is_refused(evidence):
    report, decision = copy.deepcopy(evidence)
    decision["weight_promotion_permitted"] = True
    with pytest.raises(ValueError, match="review_not_recomputed"):
        draft_custody(report, decision)


def test_promotion_claim_in_report_is_refused(evidence):
    report, decision = copy.deepcopy(evidence)
    report["weight_promotion_permitted"] = True
    with pytest.raises(ValueError, match="report_review_blocked"):
        draft_custody(report, decision)


def test_receipt_chain_requires_valid_previous_hash(evidence):
    report, decision = evidence
    prev = "a" * 64
    d = draft_custody(report, decision, previous_receipt_sha256=prev)
    assert d["previous_receipt_sha256"] == prev
    with pytest.raises(ValueError, match="invalid_previous"):
        draft_custody(report, decision, previous_receipt_sha256="not-a-hash")


def test_duplicate_json_fields_are_rejected(tmp_path):
    f = tmp_path / "duplicate.json"
    f.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="duplicate_json_key"):
        _read_json(f, 100)


def test_symlink_json_source_rejected(tmp_path):
    real = tmp_path / "actual.json"
    real.write_text("{}")
    fake = tmp_path / "fake.json"
    fake.symlink_to(real)
    with pytest.raises(ValueError, match="missing_symlink"):
        _read_json(fake, 100)


def test_large_input_rejected(tmp_path):
    f = tmp_path / "too-large.json"
    f.write_text("a" * 128)
    with pytest.raises(ValueError, match="oversized"):
        _read_json(f, 100)


def test_relabel_same_public_key_is_not_authenticated(evidence, draft, signer):
    envelope, trusted = signed(draft, signer)
    envelope["key_id"] = "another-label"
    trusted["another-label"] = trusted["research-test-only"]
    assert verified(envelope, evidence, trusted)["status"] == "blocked"


def test_nonfinite_raw_json_rejected(tmp_path):
    source = tmp_path / "nonfinite.json"
    source.write_text('{"measurement": NaN}')
    with pytest.raises(ValueError, match="nonfinite_json_constant"):
        _read_json(source, 100)