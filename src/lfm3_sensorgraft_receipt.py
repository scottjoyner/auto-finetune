"""Offline SensorGraft LFM3 research receipts; never promote model weights.

An unsigned hash manifest is NOT an authenticated receipt. Ed25519 verification
requires a trusted key installed out-of-band and an explicit revocation view.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
from pathlib import Path

from src.lfm3_real_lora_review import review

SCHEMA = "sensorgraft.lfm3.experiment-custody.v1"
SIGNED_SCHEMA = "sensorgraft.lfm3.ed25519-receipt.v1"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
KEY_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signed_message(key_id, payload):
    if not isinstance(key_id, str) or not KEY_ID.fullmatch(key_id):
        raise ValueError("invalid_signer_identity")
    return (
        b"sensorgraft.lfm3.ed25519-receipt.v1\x00"
        + key_id.encode("ascii")
        + b"\x00"
        + canonical(payload)
    )


def _blocked(reason):
    return {
        "schema": "sensorgraft.lfm3.experiment-custody-verdict.v1",
        "status": "blocked",
        "reason": reason,
        "authenticated": False,
        "weight_promotion_permitted": False,
    }


def draft_custody(report, existing_review, *, previous_receipt_sha256=None):
    """Create an UNSIGNED cross-checkable digest envelope, never an attestation."""
    try:
        verdict = review(report)
        if verdict.get("status") != "review_required":
            raise ValueError("report_review_blocked")
        if existing_review != verdict:
            raise ValueError("review_not_recomputed_or_mismatch")
        if previous_receipt_sha256 is not None and (
            not isinstance(previous_receipt_sha256, str)
            or not HEX64.fullmatch(previous_receipt_sha256)
        ):
            raise ValueError("invalid_previous_receipt_hash")
        if len(report["arms"]) > 6:
            raise ValueError("too_many_arms")
        arm_proofs = []
        for arm in report["arms"]:
            events = arm["events"]
            if not 1 <= len(events) <= 30:
                raise ValueError("unbounded_telemetry")
            arm_proofs.append(
                {
                    "run_id": arm["run_id"],
                    "seed": arm["seed"],
                    "optimizer": arm["optimizer"],
                    "initial_adapter_sha256": arm["initial_adapter_sha256"],
                    "final_adapter_sha256": arm["final_adapter_sha256"],
                    "train_sha256": arm["train_sha256"],
                    "heldout_sha256": arm["heldout_sha256"],
                    "events_sha256": digest(events),
                    "step_count": len(events),
                }
            )
        arm_proofs.sort(key=lambda x: (x["seed"], x["optimizer"]))
        if len({r["run_id"] for r in arm_proofs}) != len(arm_proofs):
            raise ValueError("duplicate_run_ids")
        payload = {
            "schema": SCHEMA,
            "provenance": "producer-claimed-unattested",
            "model_revision": report["base_revision"],
            "model_weights_sha256": report["base_sha256"],
            "report_sha256": digest(report),
            "review_sha256": digest(existing_review),
            "previous_receipt_sha256": previous_receipt_sha256,
            "source": "auto-finetune-real-lora-synthetic-only",
            "sensor_graft_ingested": False,
            "semantic_retrieval_verified": False,
            "causal_self_improvement_verified": False,
            "arm_proofs": arm_proofs,
            "weight_promotion_permitted": False,
        }
        return payload
    except (KeyError, ValueError, TypeError, OverflowError) as exc:
        raise ValueError(
            "untrusted_experiment_receipt:" + type(exc).__name__ + ":" + str(exc)
        ) from exc


def _check_payload(payload, report, existing_review):
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "provenance",
        "model_revision",
        "model_weights_sha256",
        "report_sha256",
        "review_sha256",
        "previous_receipt_sha256",
        "source",
        "sensor_graft_ingested",
        "semantic_retrieval_verified",
        "causal_self_improvement_verified",
        "arm_proofs",
        "weight_promotion_permitted",
    }:
        raise ValueError("invalid_payload_fields")
    expected = draft_custody(
        report,
        existing_review,
        previous_receipt_sha256=payload.get("previous_receipt_sha256"),
    )
    if canonical(payload) != canonical(expected):
        raise ValueError("experiment_hash_or_policy_mismatch")
    return canonical(payload)


def verify_signed_receipt(
    envelope,
    report,
    existing_review,
    *,
    trusted_public_keys,
    revoked_key_ids,
    revoked_receipt_sha256,
):
    """Authenticate only with an externally pinned Ed25519 public key and revocations.

    Trust of the signer does NOT independently prove underlying real-world
    experiment claims or authorize optimizer changes.
    """
    try:
        if (
            not isinstance(envelope, dict)
            or set(envelope)
            != {
                "schema",
                "payload",
                "key_id",
                "signature_base64",
            }
            or envelope["schema"] != SIGNED_SCHEMA
        ):
            return _blocked("invalid_signed_envelope")
        key_id = envelope["key_id"]
        if not isinstance(key_id, str) or not KEY_ID.fullmatch(key_id):
            return _blocked("invalid_signer_identity")
        if (
            not isinstance(trusted_public_keys, dict)
            or type(revoked_key_ids) not in (set, frozenset)
            or type(revoked_receipt_sha256) not in (set, frozenset)
        ):
            return _blocked("independent_trust_or_revocations_missing")
        if key_id in revoked_key_ids:
            return _blocked("revoked_signer")
        key_bytes = trusted_public_keys.get(key_id)
        if type(key_bytes) is not bytes or len(key_bytes) != 32:
            return _blocked("signer_not_independently_trusted")
        data = _check_payload(envelope["payload"], report, existing_review)
        receipt_sha = hashlib.sha256(data).hexdigest()
        if receipt_sha in revoked_receipt_sha256:
            return _blocked("revoked_experiment_receipt")
        signed = envelope["signature_base64"]
        if not isinstance(signed, str) or len(signed) > 128:
            return _blocked("malformed_signature")
        signature = base64.b64decode(signed, validate=True)
        if len(signature) != 64:
            return _blocked("invalid_signature_length")
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError:
            return _blocked("ed25519_verifier_unavailable")
        Ed25519PublicKey.from_public_bytes(key_bytes).verify(
            signature, signed_message(key_id, envelope["payload"])
        )
        return {
            "schema": "sensorgraft.lfm3.experiment-custody-verdict.v1",
            "status": "authenticated_receipt",
            "authenticated": True,
            "signer_key_id": key_id,
            "receipt_sha256": receipt_sha,
            "report_sha256": envelope["payload"]["report_sha256"],
            "model_weights_sha256": envelope["payload"]["model_weights_sha256"],
            "data_origin_independently_verified": False,
            "sensor_graft_ingested": False,
            "weight_promotion_permitted": False,
        }
    except (ValueError, TypeError, KeyError, OverflowError, binascii.Error) as exc:
        return _blocked("invalid_custody:" + type(exc).__name__)
    except Exception as exc:
        # Includes cryptography InvalidSignature; fail closed without leaking key data.
        return _blocked("signature_not_verified:" + type(exc).__name__)


def _pairs_unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate_json_key")
        obj[key] = value
    return obj


def _read_json(path, limit):
    p = Path(path)
    if not p.is_file() or p.is_symlink() or p.stat().st_size > limit:
        raise ValueError("missing_symlink_or_oversized_input")
    return json.loads(
        p.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json_constant")),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = _read_json(args.report, 300_000)
        existing_review = _read_json(args.review, 16_000)
        payload = draft_custody(report, existing_review)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True, indent=2, allow_nan=False)
        print(
            json.dumps(
                {
                    "status": "unattested_hash_manifest_only",
                    "receipt_sha256": digest(payload),
                    "output": str(args.output),
                    "weight_promotion_permitted": False,
                }
            )
        )
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit("fail_closed_receipt: " + type(exc).__name__) from exc


if __name__ == "__main__":
    main()