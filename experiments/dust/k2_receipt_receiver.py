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


def verify_event_join(root: Path, run_id: str, events_file: Path,
                      derived_file: Path | None = None,
                      expected_model_sha: str | None = None) -> dict:
    """Receiver-side join of independently signed PRE receipts and source log.

    The caller fetches the source event log over an authorized SSH connection
    into receiver-owned custody. Producer can never edit the receiver's
    HMAC-signed receipts. This does not prove model-loss quality or split
    correctness, only source event/receiver receipt agreement and sequencing.
    """
    verified = verify(root, run_id)
    receipts = [
        json.loads(line) for line in (
            root / (run_id + ".receipt.jsonl")
        ).read_text().splitlines()
    ]
    if events_file.stat().st_size > 1024 * 1024:
        raise ValueError("event join file exceeds size cap")
    prior = "0" * 64
    pre = {}
    all_pre = {}
    all_post = {}
    receipt_seen = False
    awaiting = []
    receipt_count = 0
    post_count = 0
    for line in events_file.read_text().splitlines():
        event = json.loads(line)
        event_sha = event.pop("event_sha256", None)
        if not is_hex(event_sha):
            raise ValueError("invalid producer event digest")
        if hashlib.sha256(canonical(event).encode()).hexdigest() != event_sha:
            raise ValueError("producer event content hash mismatch")
        if event.get("previous_event_sha256") != prior:
            raise ValueError("producer event chain broken")
        prior = event_sha
        if event.get("schema") != "auto-finetune.dust-k2-direction-events.v1":
            raise ValueError("wrong producer schema")
        phase = event.get("phase")
        if phase == "PRE":
            if receipt_seen or len(pre) >= 4:
                raise ValueError("unexpected PRE phase or batch length")
            index = event["candidate_index"]
            if index != post_count + len(pre) or index in pre:
                raise ValueError("duplicate/out-of-order PRE")
            all_pre[index] = dict(event)
            pre[index] = {
                "event_sha256": event_sha,
                "producer_pre_monotonic_ns": event["local_monotonic_ns"],
            }
        elif phase == "RECEIPT":
            if receipt_seen or not pre or receipt_count >= len(receipts):
                raise ValueError("RECEIPT without PRE batch")
            batch_digest = hashlib.sha256(
                "|".join(pre[x]["event_sha256"] for x in sorted(pre)).encode()
            ).hexdigest()
            independent = receipts[receipt_count]
            if (
                event["batch_sha256"] != batch_digest
                or independent["batch_sha256"] != batch_digest
                or independent["receiver_hmac_sha256"]
                    != event["receiver_hmac_sha256"]
                or independent["receiver_utc_ns"] != event["receiver_utc_ns"]
                or independent["batch_index"] != event["batch_index"]
                or independent["batch_index"] != receipt_count
            ):
                raise ValueError("cross-node PRE receipt mismatch")
            receipt_seen = True
            awaiting = list(sorted(pre))
        elif phase == "POST":
            if not receipt_seen or not awaiting:
                raise ValueError("post before independent receipt")
            index = event["candidate_index"]
            if index != awaiting[0] or index not in pre:
                raise ValueError("out-of-order POST")
            if event["pre_event_sha256"] != pre[index]["event_sha256"]:
                raise ValueError("POST references wrong PRE")
            if event["local_monotonic_ns"] <= pre[index]["producer_pre_monotonic_ns"]:
                raise ValueError("POST timestamp predates PRE")
            all_post[index] = dict(event)
            awaiting.pop(0)
            post_count += 1
            if not awaiting:
                pre = {}
                receipt_seen = False
                receipt_count += 1
        else:
            raise ValueError("unexpected producer event phase")
    if pre or awaiting or receipt_seen or receipt_count != len(receipts):
        raise ValueError("incomplete cross-node producer receipt join")
    derived_verified = False
    positive_labels = None
    derived_sha = None
    if derived_file is not None:
        if not is_hex(expected_model_sha):
            raise ValueError("expected pinned model SHA required for label join")
        if derived_file.stat().st_size > 1024 * 1024:
            raise ValueError("derived label file exceeds bounded size")
        observations = [json.loads(line) for line in
                        derived_file.read_text().splitlines()]
        if len(observations) != post_count:
            raise ValueError("derived label count mismatch")
        expected_fields = {
            "schema", "episode_hmac_sha256", "model_revision_sha256",
            "candidate_index", "features_pre_probe", "sigma",
            "loss_clean", "loss_plus", "loss_minus",
            "probe_time_order_attested",
        }
        positives = 0
        for index, row in enumerate(observations):
            p = all_pre[index]
            q = all_post[index]
            if (
                set(row) != expected_fields
                or row["schema"] != "auto-finetune.dust-predictive-probe.v1"
                or row["candidate_index"] != index
                or row["model_revision_sha256"] != expected_model_sha
                or row["episode_hmac_sha256"] != p["episode_hmac_sha256"]
                or row["features_pre_probe"] != p["features_pre_probe"]
                or row["sigma"] != p["sigma"]
                or row["loss_clean"] != p["clean_pre_probe"]
                or row["loss_plus"] != q["loss_plus"]
                or row["loss_minus"] != q["loss_minus"]
                or row["probe_time_order_attested"] is not True
            ):
                raise ValueError("derived label differs from PRE/POST events")
            positives += int(row["loss_plus"] < row["loss_clean"])
        derived_verified = True
        positive_labels = positives
        derived_sha = hashlib.sha256(derived_file.read_bytes()).hexdigest()
    return {
        **verified,
        "derived_label_join_verified": derived_verified,
        "derived_label_file_sha256": derived_sha,
        "derived_positive_directions": positive_labels,
        "producer_pre_post_chain_join_verified": True,
        "all_post_scores_follow_receiver_ack": True,
        "source_event_count": post_count * 2 + receipt_count,
        "completed_direction_count": post_count,
        "expected_receipts": receipt_count,
        "classifier_training_authorized": False,
        "audit_scope": "cryptographic receipt and sequence only; no heldout/model-quality attestation",
        "joined_producer_events_sha256":
            hashlib.sha256(events_file.read_bytes()).hexdigest(),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    options = p.add_mutually_exclusive_group(required=True)
    options.add_argument("--setup", action="store_true")
    options.add_argument("--receive", action="store_true")
    options.add_argument("--verify", action="store_true")
    p.add_argument("--run-id")
    p.add_argument("--batch-sha256")
    p.add_argument("--join-events", type=Path)
    p.add_argument("--join-derived", type=Path)
    p.add_argument("--expected-model-sha")
    args = p.parse_args(argv)
    if args.setup:
        prepare_receiver(args.root)
        print("RECEIVER_KEY_PROVISIONED_LOCAL_ONLY")
    elif args.receive:
        print(canonical(receive(args.root, args.run_id, args.batch_sha256)))
    else:
        if args.join_events is not None:
            print(canonical(verify_event_join(
                args.root, args.run_id, args.join_events,
                args.join_derived, args.expected_model_sha)))
        else:
            print(canonical(verify(args.root, args.run_id)))


if __name__ == "__main__":
    main()
