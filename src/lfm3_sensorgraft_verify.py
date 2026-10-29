"""Verify independently pinned LFM3 experiment receipts, never issue signatures.

Separate trust/revocation snapshots must be supplied from outside the report,
with SHA-256 pins obtained out-of-band. No network or database writes.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import re
from pathlib import Path

from src.lfm3_sensorgraft_receipt import _blocked, _read_json, verify_signed_receipt

HEX64 = re.compile(r"[a-f0-9]{64}\Z")
KEY_ID = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")


def _read_pinned(path, expected, limit):
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        raise ValueError("missing_independent_file_pin")
    p = Path(path)
    if not p.is_file() or p.is_symlink() or p.stat().st_size > limit:
        raise ValueError("pinned_file_missing_unsafe_or_oversized")
    if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        raise ValueError("independent_file_pin_mismatch")
    return _read_json(p, limit)


def verify_files(
    report_file,
    review_file,
    envelope_file,
    trust_file,
    revocations_file,
    *,
    trust_sha256,
    revocations_sha256,
):
    try:
        report = _read_json(report_file, 300_000)
        review = _read_json(review_file, 16_000)
        envelope = _read_json(envelope_file, 64_000)
        trust = _read_pinned(trust_file, trust_sha256, 16_000)
        revocations = _read_pinned(revocations_file, revocations_sha256, 16_000)
        if (
            not isinstance(trust, dict)
            or set(trust) != {"schema", "keys"}
            or trust["schema"] != "sensorgraft.ed25519-trust.v1"
            or not isinstance(trust["keys"], dict)
            or not 1 <= len(trust["keys"]) <= 32
            or not isinstance(revocations, dict)
            or set(revocations)
            != {
                "schema",
                "revoked_key_ids",
                "revoked_receipt_sha256",
            }
            or revocations["schema"] != "sensorgraft.ed25519-revocations.v1"
        ):
            return _blocked("invalid_external_trust_or_revocation_schema")
        public = {}
        for name, value in trust["keys"].items():
            if (
                not isinstance(name, str)
                or not KEY_ID.fullmatch(name)
                or not isinstance(value, str)
            ):
                return _blocked("invalid_trusted_key_descriptor")
            decoded = base64.b64decode(value, validate=True)
            if len(decoded) != 32:
                return _blocked("invalid_trusted_public_key_length")
            public[name] = decoded
        revoked_ids = revocations["revoked_key_ids"]
        revoked_receipts = revocations["revoked_receipt_sha256"]
        if (
            not isinstance(revoked_ids, list)
            or not isinstance(revoked_receipts, list)
            or len(revoked_ids) > 1024
            or len(revoked_receipts) > 1024
            or any(not isinstance(k, str) or not KEY_ID.fullmatch(k) for k in revoked_ids)
            or any(not isinstance(h, str) or not HEX64.fullmatch(h) for h in revoked_receipts)
            or len(set(revoked_ids)) != len(revoked_ids)
            or len(set(revoked_receipts)) != len(revoked_receipts)
        ):
            return _blocked("invalid_revocation_entries")
        result = verify_signed_receipt(
            envelope,
            report,
            review,
            trusted_public_keys=public,
            revoked_key_ids=set(revoked_ids),
            revoked_receipt_sha256=set(revoked_receipts),
        )
        if result["status"] == "authenticated_receipt":
            result["independent_trust_pin_verified"] = True
            result["independent_revocation_pin_verified"] = True
            result["revocation_freshness_verified"] = False
        return result
    except (OSError, ValueError, TypeError, KeyError, binascii.Error) as exc:
        return _blocked("invalid_external_verification_inputs:" + type(exc).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--signed-envelope", type=Path, required=True)
    parser.add_argument("--trust-store", type=Path, required=True)
    parser.add_argument("--trust-sha256", required=True)
    parser.add_argument("--revocations", type=Path, required=True)
    parser.add_argument("--revocations-sha256", required=True)
    args = parser.parse_args()
    result = verify_files(
        args.report,
        args.review,
        args.signed_envelope,
        args.trust_store,
        args.revocations,
        trust_sha256=args.trust_sha256,
        revocations_sha256=args.revocations_sha256,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    if result["status"] != "authenticated_receipt":
        raise SystemExit(78)


if __name__ == "__main__":
    main()