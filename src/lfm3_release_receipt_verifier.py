"""Read-only Ed25519 evidence checking; never authorizes model release.

There is no provisioned production trust root, durable replay ledger, review
authority, or atomic artifact publisher. Signature proof alone is insufficient
for export or promotion.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Any, Mapping

SCHEMA = "auto-finetune.release-evidence.v1"
DOMAIN = b"auto-finetune/release-evidence/ed25519/v1\n"
MAX_ENVELOPE_BYTES = 8192
MAX_VALIDITY = timedelta(hours=24)
FUTURE_SKEW = timedelta(minutes=5)
RELEASE_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
KEY_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
REPO_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}/[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
REQUEST_FIELDS = frozenset({
    "base_model_repo", "base_model_revision", "base_model_sha256",
    "adapter_sha256", "training_code_sha256", "heldout_dataset_sha256",
    "evaluation_receipt_sha256", "output_format", "quantization", "output_name",
})
MANIFEST_FIELDS = REQUEST_FIELDS | frozenset({"release_id", "issued_at", "expires_at"})
ENVELOPE_FIELDS = frozenset({"schema", "key_id", "manifest", "signature_b64"})


@dataclass(frozen=True)
class ReceiptCheck:
    status: str
    reason: str
    signature_verified: bool = False
    request_bound: bool = False
    release_id_checked_against_snapshot: bool = False
    source_attested: bool = False
    heldout_quality_attested: bool = False
    durable_replay_protection: bool = False
    export_permitted: bool = False
    promotion_permitted: bool = False

    def as_public_dict(self) -> dict[str, Any]:
        """Exclude all private paths, models, raw data, signatures or IDs."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class _InvalidReceipt(ValueError):
    pass


def _unique_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise _InvalidReceipt("duplicate_json_key")
        obj[key] = value
    return obj


def _no_constant(_value):
    raise _InvalidReceipt("nonfinite_json_constant")


def _canonical(obj: Mapping[str, Any]) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def signing_bytes(key_id: str, manifest: Mapping[str, Any]) -> bytes:
    """Deterministic public test vector; never signs or creates keys."""
    return DOMAIN + _canonical({
        "schema": SCHEMA, "key_id": key_id, "manifest": manifest,
    })


def _check_request(request: Any) -> None:
    if type(request) is not dict or request.keys() != REQUEST_FIELDS:
        raise _InvalidReceipt("invalid_expected_request_shape")
    for key, value in request.items():
        if type(value) is not str:
            raise _InvalidReceipt("invalid_expected_request_types")
        if key.endswith("_sha256") and not HASH_PATTERN.fullmatch(value):
            raise _InvalidReceipt("invalid_hash_binding")
    if not REPO_PATTERN.fullmatch(request["base_model_repo"]):
        raise _InvalidReceipt("invalid_model_repository")
    if ".." in request["base_model_repo"]:
        raise _InvalidReceipt("unsafe_model_repository")
    if not REVISION_PATTERN.fullmatch(request["base_model_revision"]):
        raise _InvalidReceipt("invalid_model_revision")
    name = request["output_name"]
    if not NAME_PATTERN.fullmatch(name) or ".." in name:
        raise _InvalidReceipt("unsafe_output_name")
    fmt, quant = request["output_format"], request["quantization"]
    if fmt not in ("merged", "gguf", "onnx"):
        raise _InvalidReceipt("unsupported_output_format")
    if (fmt == "gguf" and quant not in ("q8_0", "q4_k_m", "f16")) or (
        fmt != "gguf" and quant != "none"
    ):
        raise _InvalidReceipt("invalid_quantization_binding")


def _utc(text: Any) -> datetime:
    if type(text) is not str or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", text
    ):
        raise _InvalidReceipt("invalid_utc_timestamp")
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise _InvalidReceipt("invalid_utc_timestamp") from exc


def verify_release_evidence(
    envelope_bytes: bytes,
    *,
    trusted_public_keys: Mapping[str, Any],
    expected_request: dict[str, str],
    previously_seen_ids: frozenset[str] | set[str] | None = None,
    revoked_key_ids: frozenset[str] | set[str] = frozenset(),
    now_utc: datetime | None = None,
) -> ReceiptCheck:
    """Check a signed proposal but NEVER return export/promotion authority.

    Trusted keys and the expected request must come from an independent,
    privileged source, not the supplied envelope. The replay snapshot is
    read-only and not transactionally claimed. Optional clock injection is
    intended for unit tests; production requires a trusted UTC clock.
    """
    def denied(reason: str) -> ReceiptCheck:
        return ReceiptCheck(status="blocked", reason=reason)

    if type(envelope_bytes) is not bytes or not 0 < len(envelope_bytes) <= MAX_ENVELOPE_BYTES:
        return denied("invalid_receipt_size")
    if not isinstance(trusted_public_keys, Mapping) or not trusted_public_keys:
        return denied("trusted_keys_not_provisioned")
    if previously_seen_ids is None:
        return denied("replay_snapshot_not_provisioned")
    if type(previously_seen_ids) not in (set, frozenset) or type(revoked_key_ids) not in (set, frozenset):
        return denied("invalid_trust_context")
    try:
        _check_request(expected_request)
    except _InvalidReceipt as exc:
        return denied(str(exc))
    try:
        envelope = json.loads(
            envelope_bytes.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_no_constant,
        )
    except (UnicodeError, ValueError, TypeError, OverflowError):
        return denied("invalid_or_duplicate_json")

    if type(envelope) is not dict or envelope.keys() != ENVELOPE_FIELDS:
        return denied("invalid_envelope_fields")
    if envelope["schema"] != SCHEMA:
        return denied("unsupported_receipt_schema")
    kid, manifest, sig_b64 = (
        envelope["key_id"], envelope["manifest"], envelope["signature_b64"]
    )
    if type(kid) is not str or not KEY_ID_PATTERN.fullmatch(kid):
        return denied("invalid_key_id")
    if kid in revoked_key_ids:
        return denied("key_revoked")
    if kid not in trusted_public_keys:
        return denied("unrecognized_key_id")
    if type(manifest) is not dict or manifest.keys() != MANIFEST_FIELDS:
        return denied("invalid_manifest_fields")
    if any(type(value) is not str for value in manifest.values()):
        return denied("invalid_manifest_types")
    rid = manifest["release_id"]
    if not RELEASE_ID_PATTERN.fullmatch(rid):
        return denied("invalid_release_id")
    if rid in previously_seen_ids:
        return denied("previously_seen_release_id")
    request = {key: manifest[key] for key in REQUEST_FIELDS}
    try:
        _check_request(request)
        issued, expires = _utc(manifest["issued_at"]), _utc(manifest["expires_at"])
    except _InvalidReceipt as exc:
        return denied(str(exc))
    if request != expected_request:
        return denied("signed_request_mismatch")
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None:
        return denied("invalid_trusted_clock")
    now_utc = now_utc.astimezone(timezone.utc)
    if not (issued < expires <= issued + MAX_VALIDITY):
        return denied("invalid_receipt_validity_window")
    if issued > now_utc + FUTURE_SKEW:
        return denied("receipt_not_yet_valid")
    if now_utc >= expires:
        return denied("receipt_expired")
    if type(sig_b64) is not str or len(sig_b64) > 100:
        return denied("invalid_signature_encoding")
    try:
        signature = base64.b64decode(sig_b64, validate=True)
    except (ValueError, binascii.Error):
        return denied("invalid_signature_encoding")
    if len(signature) != 64:
        return denied("invalid_signature_length")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return denied("ed25519_verifier_unavailable")
    key = trusted_public_keys[kid]
    if not isinstance(key, Ed25519PublicKey):
        return denied("invalid_trusted_key_type")
    try:
        key.verify(signature, signing_bytes(kid, manifest))
    except (InvalidSignature, ValueError, TypeError):
        return denied("invalid_signature")

    # The supplied trust map does not prove operator authorization, signed
    # scores do not prove heldout independence, and a read-only used-ID
    # snapshot is not atomic replay protection.
    return ReceiptCheck(
        status="signature_verified_non_authorizing",
        reason="independent_release_approval_and_atomic_replay_gate_absent",
        signature_verified=True,
        request_bound=True,
        release_id_checked_against_snapshot=True,
        source_attested=False,
        heldout_quality_attested=False,
        durable_replay_protection=False,
        export_permitted=False,
        promotion_permitted=False,
    )
