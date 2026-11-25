"""Ephemeral test keys only; signed evidence is NEVER a release authority."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from src import lfm3_release_receipt_verifier as verifier
from src import model_export

NOW = datetime(2026, 10, 29, 5, 0, 0, tzinfo=timezone.utc)
KID = "research-test-only"


@pytest.fixture
def private():
    # Generated afresh in RAM for this test; never used as a production key
    # or committed/serialized to a file.
    return Ed25519PrivateKey.generate()


@pytest.fixture
def artifact_request():
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    return {
        "base_model_repo": "LiquidAI/LFM3-1.7B",
        "base_model_revision": "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd",
        "base_model_sha256": sha("synthetic-model-hash"),
        "adapter_sha256": sha("synthetic-adapter-hash"),
        "training_code_sha256": sha("synthetic-trainer-hash"),
        "heldout_dataset_sha256": sha("synthetic-holdout-hash"),
        "evaluation_receipt_sha256": sha("synthetic-evaluation-receipt"),
        "output_format": "gguf",
        "quantization": "q8_0",
        "output_name": "lfm3-research-q8_0.gguf",
    }


def manifest(artifact_request, **changes):
    data = dict(artifact_request)
    data.update({
        "release_id": "1" * 32,
        "issued_at": "2026-10-29T04:55:00Z",
        "expires_at": "2026-10-29T06:00:00Z",
    })
    data.update(changes)
    return data


def envelope(private, artifact_request, key_id=KID, manifest_override=None):
    man = manifest_override if manifest_override is not None else manifest(artifact_request)
    signed = private.sign(verifier.signing_bytes(key_id, man))
    return {
        "schema": verifier.SCHEMA,
        "key_id": key_id,
        "manifest": man,
        "signature_b64": base64.b64encode(signed).decode("ascii"),
    }


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def checked(private, artifact_request, value, **kwargs):
    opts = {
        "trusted_public_keys": {KID: private.public_key()},
        "expected_request": artifact_request,
        "previously_seen_ids": frozenset(),
        "now_utc": NOW,
    }
    opts.update(kwargs)
    return verifier.verify_release_evidence(
        encoded(value) if not isinstance(value, bytes) else value, **opts
    )


def test_ephemeral_valid_signature_is_not_release_authorization(private, artifact_request):
    r = checked(private, artifact_request, envelope(private, artifact_request))
    assert r.status == "signature_verified_non_authorizing"
    assert r.signature_verified is True
    assert r.request_bound is True
    assert r.release_id_checked_against_snapshot is True
    assert r.export_permitted is False
    assert r.promotion_permitted is False
    assert r.source_attested is False
    assert r.heldout_quality_attested is False
    assert r.durable_replay_protection is False


@pytest.mark.parametrize("field", sorted(verifier.REQUEST_FIELDS))
def test_mismatched_expected_artifact_or_export_policy_is_denied(private, artifact_request, field):
    other = dict(artifact_request)
    if field.endswith("_sha256"):
        other[field] = hashlib.sha256(b"other").hexdigest()
    elif field == "base_model_revision":
        other[field] = "f" * 40
    elif field == "base_model_repo":
        other[field] = "Other/Other-model"
    elif field == "output_format":
        other[field], other["quantization"] = "onnx", "none"
    elif field == "quantization":
        other[field] = "f16"
    elif field == "output_name":
        other[field] = "other.gguf"
    check = checked(private, artifact_request, envelope(private, other))
    assert check.status == "blocked"
    assert check.export_permitted is False


@pytest.mark.parametrize("field,value", [
    ("output_name", "../escape.gguf"),
    ("output_name", "nested/file.gguf"),
    ("output_name", "hi..bye.gguf"),
    ("output_name", "bad\\name.gguf"),
    ("base_model_repo", "../LFM3"),
    ("base_model_repo", "My..repo/LFM3"),
    ("base_model_revision", "main"),
    ("base_model_sha256", "x" * 64),
    ("evaluation_receipt_sha256", "0" * 63),
    ("output_format", "unsafe"),
    ("quantization", "q1_0"),
])
def test_malformed_signed_policy_rejected(private, artifact_request, field, value):
    man = manifest(artifact_request, **{field: value})
    result = checked(private, artifact_request, envelope(private, artifact_request, manifest_override=man))
    assert result.status == "blocked"
    assert not result.export_permitted


def test_forged_signature_by_untrusted_signer_is_rejected(private, artifact_request):
    attacker = Ed25519PrivateKey.generate()
    result = checked(private, artifact_request, envelope(attacker, artifact_request))
    assert result.reason == "invalid_signature"


def test_signed_key_id_substitution_invalidates_signature(private, artifact_request):
    second = Ed25519PrivateKey.generate()
    value = envelope(private, artifact_request)
    value["key_id"] = "other-trusted"
    result = checked(private, artifact_request, value, trusted_public_keys={
        KID: private.public_key(), "other-trusted": second.public_key()
    })
    assert result.reason == "invalid_signature"


def test_untrusted_signer_not_accepted_from_envelope(private, artifact_request):
    val = envelope(private, artifact_request)
    result = checked(private, artifact_request, val, trusted_public_keys={
        "unrelated": private.public_key()
    })
    assert result.reason == "unrecognized_key_id"


def test_no_provisioned_trust_root_is_blocked(private, artifact_request):
    result = checked(private, artifact_request, envelope(private, artifact_request),
                     trusted_public_keys={})
    assert result.reason == "trusted_keys_not_provisioned"


def test_missing_replay_snapshot_is_blocked(private, artifact_request):
    result = checked(private, artifact_request, envelope(private, artifact_request),
                     previously_seen_ids=None)
    assert result.reason == "replay_snapshot_not_provisioned"


def test_previously_consumed_release_id_is_rejected(private, artifact_request):
    val = envelope(private, artifact_request)
    result = checked(private, artifact_request, val,
                     previously_seen_ids={val["manifest"]["release_id"]})
    assert result.reason == "previously_seen_release_id"


def test_revoked_key_is_blocked(private, artifact_request):
    result = checked(private, artifact_request, envelope(private, artifact_request),
                     revoked_key_ids={KID})
    assert result.reason == "key_revoked"


def test_non_ed25519_key_type_is_refused(private, artifact_request):
    result = checked(private, artifact_request, envelope(private, artifact_request),
                     trusted_public_keys={KID: b"untrusted-key-data"})
    assert result.reason == "invalid_trusted_key_type"


@pytest.mark.parametrize("edits,expected", [
    ({"issued_at": "2026-10-29T05:07:00Z"}, "receipt_not_yet_valid"),
    ({"expires_at": "2026-10-29T04:59:59Z"}, "receipt_expired"),
    ({"expires_at": "2026-10-31T05:00:00Z"}, "invalid_receipt_validity_window"),
    ({"issued_at": "2026-10-30T04:55:00Z", "expires_at": "2026-10-30T06:00:00Z"}, "receipt_not_yet_valid"),
    ({"issued_at": "2026-10-29 04:55:00"}, "invalid_utc_timestamp"),
    ({"expires_at": "2026-10-29T06:00:00+00:00"}, "invalid_utc_timestamp"),
    ({"expires_at": "not-a-timestamp"}, "invalid_utc_timestamp"),
    ({"expires_at": "2026-10-29T04:50:00Z"}, "invalid_receipt_validity_window"),
])
def test_time_restrictions(private, artifact_request, edits, expected):
    man = manifest(artifact_request, **edits)
    result = checked(private, artifact_request, envelope(private, artifact_request, manifest_override=man))
    assert result.reason == expected


def test_expiry_boundary_is_closed(private, artifact_request):
    man = manifest(artifact_request, expires_at="2026-10-29T05:00:00Z")
    result = checked(private, artifact_request, envelope(private, artifact_request, manifest_override=man))
    assert result.reason == "receipt_expired"


def test_output_format_onnx_is_validly_signed_but_never_authorized(private, artifact_request):
    req = dict(artifact_request, output_format="onnx", quantization="none",
               output_name="model.onnx")
    result = checked(private, req, envelope(private, req), expected_request=req)
    assert result.signature_verified is True
    assert result.export_permitted is False


def test_output_format_merged_validly_signed_but_never_authorized(private, artifact_request):
    req = dict(artifact_request, output_format="merged", quantization="none",
               output_name="merged-checkpoint")
    result = checked(private, req, envelope(private, req), expected_request=req)
    assert result.signature_verified and not result.export_permitted


def test_manifest_signature_tampering_is_rejected(private, artifact_request):
    val = envelope(private, artifact_request)
    val["manifest"]["expires_at"] = "2026-10-29T06:00:01Z"
    result = checked(private, artifact_request, val)
    assert result.reason == "invalid_signature"


def test_untrusted_new_envelope_fields_rejected(private, artifact_request):
    val = envelope(private, artifact_request)
    val["embedded_public_key"] = "ATTACKER_CONTROLLED"
    assert checked(private, artifact_request, val).reason == "invalid_envelope_fields"


def test_untrusted_new_manifest_fields_rejected(private, artifact_request):
    val = envelope(private, artifact_request)
    val["manifest"]["allow_export"] = True
    assert checked(private, artifact_request, val).reason == "invalid_manifest_fields"


def test_boolean_manifest_rejected(private, artifact_request):
    val = envelope(private, artifact_request)
    val["manifest"]["base_model_sha256"] = True
    assert checked(private, artifact_request, val).reason == "invalid_manifest_types"


def test_invalid_signature_encoding(private, artifact_request):
    val = envelope(private, artifact_request)
    val["signature_b64"] = "***"
    assert checked(private, artifact_request, val).reason == "invalid_signature_encoding"


def test_invalid_signature_length(private, artifact_request):
    val = envelope(private, artifact_request)
    val["signature_b64"] = base64.b64encode(b"short").decode()
    assert checked(private, artifact_request, val).reason == "invalid_signature_length"


@pytest.mark.parametrize("invalid", [b"", b"\xff", b"{broken", b"null", b"[]"])
def test_invalid_envelope_syntax_fails_closed(private, artifact_request, invalid):
    assert checked(private, artifact_request, invalid).status == "blocked"


def test_json_duplicate_top_level_key_rejected(private, artifact_request):
    val = envelope(private, artifact_request)
    raw = encoded(val).replace(b'"schema":', b'"schema":"evil","schema":')
    assert checked(private, artifact_request, raw).reason == "invalid_or_duplicate_json"


def test_json_duplicate_nested_key_rejected(private, artifact_request):
    val = envelope(private, artifact_request)
    raw = encoded(val).replace(b'"release_id":', b'"release_id":"0","release_id":')
    assert checked(private, artifact_request, raw).reason == "invalid_or_duplicate_json"


def test_json_nan_literal_rejected(private, artifact_request):
    val = envelope(private, artifact_request)
    raw = encoded(val).replace(b'"release_id":"11111111111111111111111111111111"',
                                b'"release_id":NaN')
    assert checked(private, artifact_request, raw).reason == "invalid_or_duplicate_json"


def test_maximum_envelope_size_is_enforced(private, artifact_request):
    val = encoded(envelope(private, artifact_request))
    assert len(val) < verifier.MAX_ENVELOPE_BYTES
    assert checked(private, artifact_request, val + b" " * 8192).reason == "invalid_receipt_size"


def test_wrong_schema_rejected(private, artifact_request):
    val = envelope(private, artifact_request)
    val["schema"] = "unauthorized.version2"
    assert checked(private, artifact_request, val).reason == "unsupported_receipt_schema"


def test_public_output_never_echoes_model_or_ids(private, artifact_request):
    val = envelope(private, artifact_request)
    data = checked(private, artifact_request, val).as_public_dict()
    plain = json.dumps(data)
    assert "LiquidAI" not in plain
    assert KID not in plain
    assert artifact_request["adapter_sha256"] not in plain
    assert val["manifest"]["release_id"] not in plain
    assert val["signature_b64"] not in plain
    assert "export_permitted" in data and not data["export_permitted"]


def test_no_signing_key_is_provisioned_in_runtime_module():
    source = verifier.__file__
    with open(source, encoding="utf-8") as f:
        text = f.read()
    assert "Ed25519PrivateKey" not in text
    assert "PRIVATE KEY-----" not in text
    assert "load_pem_private_key" not in text


def test_no_signature_ever_unlocks_existing_exporter(private, artifact_request, tmp_path):
    verified = checked(private, artifact_request, envelope(private, artifact_request))
    assert verified.signature_verified and not verified.export_permitted
    path = str(tmp_path / "no-write")
    assert model_export.export_merged("private", path).success is False
    assert model_export.export_gguf("private", path).success is False
    assert model_export.export_onnx("private", path).success is False
    assert not (tmp_path / "no-write").exists()


def test_public_result_type_immutable(private, artifact_request):
    result = checked(private, artifact_request, envelope(private, artifact_request))
    with pytest.raises(AttributeError):
        result.export_permitted = True
