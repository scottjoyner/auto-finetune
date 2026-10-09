"""Fail-closed x1 cohort ledger reconciliation contract, with synthetic files."""
from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from experiments.dust.k2_cohort_reconcile import (
    EXPECTED_MODEL_SHA, reconcile,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(root, *, shared_cluster=False):
    custody = root / "receiver"
    custody.mkdir(mode=0o700)
    intake = root / "intake"
    intake.mkdir(mode=0o700)
    rows = []
    for index in range(2):
        run = f"{index + 1:032x}"
        paths = {}
        for kind, suffix in (("events", "events.jsonl"),
                             ("derived", "derived.jsonl"),
                             ("summary", "summary.json")):
            path = intake / (run + "." + suffix)
            paths[kind] = path
            path.write_text("{}\n")
            path.chmod(0o600)
        summary_doc = {
            "population": 8, "optimizer_updates": 0, "backward_calls": 0,
            "base_weights_unchanged": True,
            "adapter_weights_unchanged": True,
            "source_group_schema": "masked-prompt-prefix-v2",
            "model_revision_sha256": EXPECTED_MODEL_SHA,
            "source_episode_hmac_sha256": "a" * 64 if index == 0 else "b" * 64,
            "preflight_manifest_sha256": "e" * 64,
            "preflight_group_split": "train" if index == 0 else "test",
            "derived_evidence": {"rows": 8, "receiver_precommit_receipts": 2},
        }
        paths["summary"].write_text(json.dumps(summary_doc))
        data = {
            "receiver_run_id": run,
            "source_row_count": 8,
            "split": "train" if index == 0 else "test",
            "episode_hmac_sha256": summary_doc["source_episode_hmac_sha256"],
            "near_duplicate_cluster_sha256":
                "c" * 64 if shared_cluster or index == 0 else "d" * 64,
        }
        data.update({k + "_sha256": sha(v) for k, v in paths.items()})
        rows.append(data)
    cohort = {
        "schema": "auto-finetune.dust-k2-bounded-cohort-collection.v1",
        "source_samples": rows, "direction_rows": 16,
        "classifier_training_authorized": False,
        "split_counts": {"train": 1, "test": 1, "validation": 0},
        "preflight_sha256": "e" * 64,
    }
    manifest = intake / "producer-cohort-summary.json"
    manifest.write_text(json.dumps(cohort))
    return custody, intake, manifest


class CohortReceiptTests(unittest.TestCase):
    @staticmethod
    def receipt_mock(*args, **kwargs):
        return {
            "expected_receipts": 2,
            "signed_batch_receipts": 2,
            "completed_direction_count": 8,
            "derived_positive_directions": 3,
            "derived_label_join_verified": True,
            "all_post_scores_follow_receiver_ack": True,
            "joined_producer_events_sha256": "e" * 64,
            "derived_label_file_sha256": "f" * 64,
        }

    def test_reconciliation_success_is_not_independent_key_custody(self):
        with tempfile.TemporaryDirectory() as tmp:
            custody, intake, manifest = make_fixture(pathlib.Path(tmp))
            with patch(
                "experiments.dust.k2_cohort_reconcile.verify_event_join",
                self.receipt_mock
            ):
                report = reconcile(custody, intake, manifest)
            self.assertEqual(report["direction_labels_verified"], 16)
            self.assertEqual(report["independent_receiver_hmac_receipts_verified"], 4)
            self.assertEqual(report["distinct_near_duplicate_clusters"], 2)
            self.assertEqual(report["beneficial_plus_directions"], 6)
            self.assertFalse(report["classifier_training_authorized"])
            self.assertFalse(report["producer_cannot_read_receiver_key"])
            self.assertIn("UNPROVEN", report["result"])

    def test_rejects_copy_tampering_before_receiver_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            custody, intake, manifest = make_fixture(pathlib.Path(tmp))
            target = intake / ("1".zfill(32) + ".derived.jsonl")
            target.write_text('{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                reconcile(custody, intake, manifest)

    def test_quarantines_repeated_near_duplicate_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            custody, intake, manifest = make_fixture(
                pathlib.Path(tmp), shared_cluster=True)
            with self.assertRaisesRegex(ValueError, "source isolation"):
                reconcile(custody, intake, manifest)

    def test_rejects_world_readable_private_intake(self):
        with tempfile.TemporaryDirectory() as tmp:
            custody, intake, manifest = make_fixture(pathlib.Path(tmp))
            intake.chmod(0o755)
            with self.assertRaises(PermissionError):
                reconcile(custody, intake, manifest)


if __name__ == "__main__":
    unittest.main()
