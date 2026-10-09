"""Receiver-owned HMAC precommit receipt, for independent research witnessing.

A different node (x1) runs this script via restricted authenticated SSH.
The HMAC key NEVER crosses to the K2 producer (Xwing). A receipt is durably
fsynced before ACK. Store only run ID, batch digest, timing and a MAC:
no prompts, tokens, direction vectors, raw tensors, or model data.

This is a research audit record, not a trusted clock / HSM and not enough on
its own to certify prompts were disjoint or actual model heldout quality.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import time


def is_hex(value: str, size: int = 64):
    return isinstance(value, str) and len(value) == size and all(
        c in "0123456789abcdef" for c in value)


def canonical(record):
    return json.dumps(record, separators=(",", ":"), sort_keys=True)


def receiver_key(root: Path) -> bytes:
    path = root / "receiver-owned.key"
    st = path.stat()
    if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
        raise PermissionError("receiver key needs mode 600")
    raw = path.read_bytes()
    if not 32 <= len(raw) <= 64:
        raise ValueError("invalid receiver key length")
    return raw


def prepare_receiver(root: Path) -> None:
    """Explicit, one-time receiver-local setup; never ran on Xwing."""
    root = root.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077:
        raise PermissionError("receiver root requires mode 700")
    key_file = root / "receiver-owned.key"
    if not key_file.exists():
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(os.urandom(32))
            stream.flush()
            os.fsync(stream.fileno())
    receiver_key(root)


def receive(root: Path, run_id: str, batch_sha: str) -> dict:
    """Hash-chain receipts under exclusive file lock, bounded 64 batches."""
    if not is_hex(run_id, 32) or not is_hex(batch_sha):
        raise ValueError("invalid run or batch digest")
    root = root.resolve(strict=True)
    if root.stat().st_mode & 0o077:
        raise PermissionError("receiver root must be private")
    key = receiver_key(root)
    filepath = root / (run_id + ".receipt.jsonl")
    fd = os.open(filepath, os.O_RDWR | os.O_CREAT
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "r+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        if stream.seek(0, os.SEEK_END) > 32768:
            raise ValueError("receiver ledger size cap")
        stream.seek(0)
        lines = stream.readlines()
        if len(lines) >= 64:
            raise ValueError("receipt batch cap")
        prior = "0" * 64
        seen = set()
        for line in lines:
            rec = json.loads(line)
            if rec["run_id"] != run_id:
                raise ValueError("receipt mixed-run violation")
            check = dict(rec)
            signed = check.pop("receiver_hmac_sha256")
            if not hmac.compare_digest(
                hmac.new(key, canonical(check).encode(), hashlib.sha256).hexdigest(),
                signed
            ):
                raise ValueError("receiver receipt MAC mismatch")
            if rec["previous_receipt_hmac_sha256"] != prior:
                raise ValueError("receiver receipt chain break")
            if rec["batch_sha256"] in seen:
                raise ValueError("duplicated PRE batch receipt")
            seen.add(rec["batch_sha256"])
            prior = signed
        if batch_sha in seen:
            raise ValueError("duplicate receipt request")
        payload = {
            "schema": "auto-finetune.dust-k2-independent-receipt.v1",
            "run_id": run_id,
            "batch_sha256": batch_sha,
            "batch_index": len(lines),
            "receiver_utc_ns": time.time_ns(),
            "previous_receipt_hmac_sha256": prior,
        }
        payload["receiver_hmac_sha256"] = hmac.new(
            key, canonical(payload).encode(), hashlib.sha256).hexdigest()
        stream.seek(0, os.SEEK_END)
        stream.write(canonical(payload) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return payload


def verify(root: Path, run_id: str) -> dict:
    if not is_hex(run_id, 32):
        raise ValueError("invalid run ID")
    root = root.resolve(strict=True)
    key = receiver_key(root)
    data = (root / (run_id + ".receipt.jsonl")).read_text()
    prior = "0" * 64
    seen = set()
    count = 0
    for line in data.splitlines():
        row = json.loads(line)
        signed = row.pop("receiver_hmac_sha256")
        if (row["run_id"] != run_id or row["batch_index"] != count
                or row["previous_receipt_hmac_sha256"] != prior
                or row["batch_sha256"] in seen):
            raise ValueError("bad independent receipt sequence")
        if not hmac.compare_digest(
            hmac.new(key, canonical(row).encode(), hashlib.sha256).hexdigest(),
            signed
        ):
            raise ValueError("invalid receiver HMAC")
        seen.add(row["batch_sha256"])
        prior = signed
        count += 1
    return {"schema": "dust-receiver-verified.v1",
            "signed_batch_receipts": count,
            "unique_prebatch_sha256": len(seen),
            "receiver_hmac_chain_verified": True,
            "producer_has_receiver_key": False,
            "privacy": "only SHA digests and local timing",
            "independent_custody_for_precommit": True,
            "authorizes_real_classifier_training": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    options = p.add_mutually_exclusive_group(required=True)
    options.add_argument("--setup", action="store_true")
    options.add_argument("--receive", action="store_true")
    options.add_argument("--verify", action="store_true")
    p.add_argument("--run-id")
    p.add_argument("--batch-sha256")
    args = p.parse_args(argv)
    if args.setup:
        prepare_receiver(args.root)
        print("RECEIVER_KEY_PROVISIONED_LOCAL_ONLY")
    elif args.receive:
        print(canonical(receive(args.root, args.run_id, args.batch_sha256)))
    else:
        print(canonical(verify(args.root, args.run_id)))


if __name__ == "__main__":
    main()
