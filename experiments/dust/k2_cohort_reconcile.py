"""Independent x1 cohort reconciliation: receipt, scalar label, cluster custody.

Requires custody copies transferred from Xwing into a mode-700 x1 directory.
The HMAC signing key remains exclusively with the x1 receiver. This runner
never loads the model or reads prompt text. It refuses any missing, altered,
incomplete, replayed or cross-split cluster record.

PASS authorizes *read-only dataset inspection only*. Classifier training stays
HOLD until enough independent prompts and suitable real feature histories.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from .k2_receipt_receiver import verify_event_join

SCHEMA = "auto-finetune.dust-k2-reconciled-cohort.v1"
EXPECTED_MODEL_SHA = "6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconcile(root: Path, intake: Path, producer_summary: Path):
    if not root.is_dir() or not intake.is_dir():
        raise ValueError("missing receiver and intake directories")
    for folder in (root, intake):
        if folder.is_symlink() or folder.stat().st_mode & 0o077:
            raise PermissionError("receiver and intake must be private mode 700")
    report = json.loads(producer_summary.read_text(encoding="utf-8"))
    if report.get("schema") != "auto-finetune.dust-k2-bounded-cohort-collection.v1":
        raise ValueError("wrong producer collection schema")
    samples = report.get("source_samples")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 8:
        raise ValueError("cohort count invalid")
    if (report.get("classifier_training_authorized") is not False
            or report.get("direction_rows") != len(samples) * 8
            or len({r["near_duplicate_cluster_sha256"] for r in samples})
                != len(samples)):
        raise ValueError("original cohort violates source isolation")
    grouped = Counter()
    observations = 0
    receipts = 0
    positives = 0
    fingerprints = []
    seen = set()
    for item in samples:
        run_id = item["receiver_run_id"]
        if run_id in seen or len(run_id) != 32 or any(
                ch not in "0123456789abcdef" for ch in run_id):
            raise ValueError("duplicated/invalid receiver run ID")
        seen.add(run_id)
        if item["split"] not in ("train","validation","test"):
            raise ValueError("unexpected cohort split")
        if item["source_row_count"] != 8:
            raise ValueError("unexpected direction population")
        paths = {
            kind: intake / (run_id + "." + ext)
            for kind, ext in (
                ("events", "events.jsonl"),
                ("derived", "derived.jsonl"),
                ("summary", "summary.json"),
            )
        }
        for kind, path in paths.items():
            if (path.is_symlink() or not path.is_file()
                    or path.stat().st_mode & 0o077):
                raise PermissionError("untrusted or world-readable intake file")
            if file_sha(path) != item[kind + "_sha256"]:
                raise ValueError("copied " + kind + " file digest mismatch")
        summary = json.loads(paths["summary"].read_text())
        if (
            summary.get("population") != 8
            or summary.get("optimizer_updates") != 0
            or summary.get("backward_calls") != 0
            or summary.get("base_weights_unchanged") is not True
            or summary.get("adapter_weights_unchanged") is not True
            or summary.get("source_group_schema")
                != "masked-prompt-prefix-v2"
            or summary.get("model_revision_sha256") != EXPECTED_MODEL_SHA
            or summary.get("source_episode_hmac_sha256")
                != item["episode_hmac_sha256"]
            or summary.get("preflight_manifest_sha256")
                != report["preflight_sha256"]
            or summary.get("preflight_group_split") != item["split"]
            or summary["derived_evidence"]["rows"] != 8
            or summary["derived_evidence"]["receiver_precommit_receipts"] != 2
        ):
            raise ValueError("producer summary differs from pinned cohort")
        joined = verify_event_join(
            root, run_id, paths["events"], paths["derived"],
            EXPECTED_MODEL_SHA)
        if (
            joined["expected_receipts"] != 2
            or joined["completed_direction_count"] != 8
            or joined["derived_label_join_verified"] is not True
            or joined["all_post_scores_follow_receiver_ack"] is not True
        ):
            raise ValueError("independent receiver-label join failed")
        grouped[item["split"]] += 1
        receipts += joined["signed_batch_receipts"]
        observations += joined["completed_direction_count"]
        positives += joined["derived_positive_directions"]
        fingerprints.append({
            "receiver_run_id": run_id,
            "split": item["split"],
            "near_duplicate_cluster_sha256":
                item["near_duplicate_cluster_sha256"],
            "events_sha256": joined["joined_producer_events_sha256"],
            "derived_sha256": joined["derived_label_file_sha256"],
        })
    if dict(grouped) != report["split_counts"]:
        raise ValueError("source split counts drifted")
    return {
        "schema": SCHEMA,
        "result": "PASS_RECEIVER_CUSTODY_ONLY",
        "source_episode_count": len(samples),
        "distinct_near_duplicate_clusters": len(samples),
        "split_counts": dict(sorted(grouped.items())),
        "independent_receiver_hmac_receipts_verified": receipts,
        "direction_labels_verified": observations,
        "beneficial_plus_directions": positives,
        "preflight_sha256": report["preflight_sha256"],
        "producer_summary_sha256": file_sha(producer_summary),
        "evidence": fingerprints,
        "any_prompt_text_in_report": False,
        "independent_semantic_duplicate_screening": False,
        "classifier_training_authorized": False,
        "optimizer_authority": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--receiver-root", type=Path, required=True)
    p.add_argument("--intake-directory", type=Path, required=True)
    p.add_argument("--producer-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.verify_only:
        p.error("--verify-only required")
    if args.output.exists() or args.output.is_symlink():
        p.error("cannot overwrite existing independent report")
    os.umask(0o077)
    outcome = reconcile(args.receiver_root, args.intake_directory,
                        args.producer_summary)
    with args.output.open("x", encoding="utf-8") as result:
        result.write(json.dumps(outcome, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        key: value for key, value in outcome.items()
        if key != "evidence"
    }, sort_keys=True))


if __name__ == "__main__":
    main()
