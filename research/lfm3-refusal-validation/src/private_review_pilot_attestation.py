"""Fail-closed private pilot route attestation against the frozen source panel.

Checks exact balanced case selection, three A/B/C responses per case, stable
blinded ordering and original journal integrity. No grades or private IDs print.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from src.private_review_pilot import (
    PILOT_NAME, SOURCE_EVIDENCE, N_PER_REVIEWER, N_PER_BUCKET, BUCKETS,
    load_route, select_cases,
)
from src.reviewer_batches import source_hash
from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.xstest_450_eval import ARM_ORDER, read_verified_dataset
from src.xstest_progress_audit import audit, snapshot_complete_lines


def expected_route_ids(chosen, mapping):
    """Reconstruct exactly 32 original prompts times three model responses."""
    if set(chosen) != set(BUCKETS) or any(
            len(ids) != N_PER_BUCKET or len(ids) != len(set(ids))
            for ids in chosen.values()):
        raise ValueError("pilot strata are not exactly balanced")
    all_cases = [case for group in BUCKETS for case in chosen[group]]
    if len(set(all_cases)) != 32:
        raise ValueError("pilot must have 32 unique prompts")
    reverse = {}
    for blinded_id, data in mapping.items():
        if not isinstance(data, dict):
            raise ValueError("invalid private mapping entry")
        key = (data.get("source_id"), data.get("arm"))
        if key in reverse:
            raise ValueError("duplicate source/arm identity")
        reverse[key] = blinded_id
    if len(reverse) != 1350:
        raise ValueError("private mapping must cover 1350 paired arm/case outputs")
    selected = [reverse[(case, arm)] for case in all_cases for arm in ARM_ORDER]
    if len(selected) != N_PER_REVIEWER or len(set(selected)) != N_PER_REVIEWER:
        raise ValueError("route must be exactly 96 distinct blinded records")
    return sorted(selected, key=lambda identity: hashlib.sha256(
        (PILOT_NAME+":order:"+identity).encode("utf-8")).hexdigest())
def validate_routing_order(entries, expected, manifest_ids):
    if len(manifest_ids) != 1350 or len(set(manifest_ids)) != 1350:
        raise ValueError("bad original reviewer batch roster")
    if len(entries) != N_PER_REVIEWER or len(expected) != N_PER_REVIEWER:
        raise ValueError("wrong pilot entry count")
    positions = {key: (i // 90 + 1, i % 90 + 1)
                 for i, key in enumerate(manifest_ids)}
    for actual, key in zip(entries, expected):
        if (actual["review_id"] != key or
                (actual["batch"], actual["index"]) != positions[key]):
            raise ValueError("pilot route differs from deterministic source selection")


def attest_route_with_entries(results_dir, packet_dir, batches_dir, route_path):
    """Return aggregate attestation and its private, verified route snapshot.

    The entries stay in-process and must never be logged or published.
    """
    from src.reviewer_handoff_audit import audit as custody_audit
    results_dir, packet_dir, batches_dir = map(
        Path, (results_dir, packet_dir, batches_dir))
    custody = custody_audit(packet_dir, batches_dir)
    if (not custody["original_source_columns_intact"] or
            not custody["reviewer_batch_privacy_permissions_verified"]):
        raise ValueError("original reviewer packet custody failed")
    entries, private_route = load_route(
        route_path, packet_dir, batches_dir, include_manifest=True)
    progress = audit(results_dir)
    if (progress["recorded_generations"] != 1350 or
            progress["journal_integrity"] != "passed" or
            progress["ignored_unterminated_tail"] or
            progress["checkpoint_sha256"] != checked_arm_paths(ARMS)):
        raise ValueError("frozen generation journal or checkpoint identity invalid")
    rows, partial, digest = snapshot_complete_lines(
        results_dir / "private_generations.jsonl")
    if partial or digest != progress["journal_complete_lines_sha256"]:
        raise ValueError("journal snapshot drifted during attestation")
    evidence = json.loads(SOURCE_EVIDENCE.read_text(encoding="utf-8"))
    if (evidence["private_journal_sha256"] != digest or
            evidence["checkpoint_sha256"] != progress["checkpoint_sha256"] or
            evidence["benchmark_sha256"] != progress["benchmark_sha256"]):
        raise ValueError("response diversity audit and source do not agree")
    if (set(private_route) != {
            "study", "n_prompts", "entries_per_reviewer", "source_journal_sha256",
            "review_batch_manifest_sha256", "private_model_key_sha256", "route",
            "human_labels_present_in_route", "model_identity_present_in_route"
        } or private_route["source_journal_sha256"] != digest or
            private_route["private_model_key_sha256"] !=
            source_hash(packet_dir / "private_model_key.json")):
        raise ValueError("private route provenance no longer matches the frozen source")
    chosen = select_cases(rows, read_verified_dataset(), evidence)
    mapping = json.loads((packet_dir / "private_model_key.json").read_text())
    expected = expected_route_ids(chosen, mapping)
    manifest = json.loads((batches_dir / "private_batch_manifest.json").read_text())
    validate_routing_order(entries, expected, manifest["review_ids"])
    report = {
        "evidence_class": "private-source-blind route attestation only",
        "pilot_route_reconstructed_exactly": True,
        "n_balanced_source_buckets": len(BUCKETS),
        "source_prompts": 32,
        "blinded_response_targets": N_PER_REVIEWER,
        "source_journal_integrity": "passed",
        "reviewer_batch_custody": "passed",
        "human_labels_added": False,
        "behavioral_safety_validated": False,
        "model_release_approved": False,
    }
    return report, entries


def attest_route(results_dir, packet_dir, batches_dir, route_path):
    """Public, aggregate-only attestation: never publish the private route."""
    report, _ = attest_route_with_entries(
        results_dir, packet_dir, batches_dir, route_path)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--packet-dir", type=Path, required=True)
    p.add_argument("--batches-dir", type=Path, required=True)
    p.add_argument("--route", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(attest_route(
        a.results_dir, a.packet_dir, a.batches_dir, a.route), indent=2))


if __name__ == "__main__":
    main()
