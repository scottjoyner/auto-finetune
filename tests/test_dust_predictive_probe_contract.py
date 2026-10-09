"""Probe ingestion gate: only keyed, pre-probe and schema-verified evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dust import predictive_probe_contract as contract


def valid_record(episode="a" * 64, candidate=0):
    return {
        "schema": contract.SCHEMA,
        "episode_hmac_sha256": episode,
        "model_revision_sha256": "b" * 64,
        "candidate_index": candidate,
        "features_pre_probe": [0.1] * 8,
        "sigma": 0.25,
        "loss_clean": 1.0,
        "loss_plus": 0.9,
        "loss_minus": 1.1,
        "probe_time_order_attested": True,
    }


class ProbeContractTests(unittest.TestCase):
    def test_label_requires_completed_forward_evidence(self):
        item = contract.validate_record(valid_record())
        self.assertEqual(item["label"], 1)
        self.assertAlmostEqual(item["gain"], .1)
        self.assertAlmostEqual(item["antithetic_slope"], -.4)
        bad = valid_record()
        bad["loss_plus"] = 1.01
        self.assertEqual(contract.validate_record(bad)["label"], 0)

    def test_reject_raw_tokens_extra_fields_and_late_features(self):
        for key, value in (
            ("tokens", [1, 2, 3]),
            ("prompt", "private text"),
            ("candidate_vector", [1.0]),
            ("loss_after_epoch", 0.5),
        ):
            row = valid_record()
            row[key] = value
            with self.assertRaisesRegex(ValueError, "unsafe"):
                contract.validate_record(row)
        row = valid_record()
        row["probe_time_order_attested"] = False
        with self.assertRaisesRegex(ValueError, "attestation"):
            contract.validate_record(row)
        row = valid_record()
        del row["loss_minus"]
        with self.assertRaisesRegex(ValueError, "unsafe"):
            contract.validate_record(row)

    def test_nonfinite_invalid_shapes_sigma_and_digest_rejected(self):
        for key, value in (("sigma", 0.0),
                           ("loss_plus", float("nan")),
                           ("features_pre_probe", [0.1]),
                           ("episode_hmac_sha256", "not-a-prompt-hmac")):
            row = valid_record()
            row[key] = value
            with self.assertRaises(ValueError):
                contract.validate_record(row)

    def test_episode_split_is_deterministic_and_group_stable(self):
        episode = "0" * 64
        split = contract.partition(episode)
        self.assertIn(split, ("train", "validation", "test"))
        self.assertEqual(contract.partition(episode), split)
        records = [contract.validate_record(valid_record(episode, i))
                   for i in range(8)]
        self.assertEqual({r["split"] for r in records}, {split})

    def test_bounded_jsonl_validate_only_and_no_feature_output(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "derived.jsonl"
            rows = [valid_record("a" * 64, 0),
                    valid_record("a" * 64, 1),
                    valid_record("c" * 64, 0)]
            path.write_text("".join(json.dumps(x) + "\n" for x in rows))
            report = contract.validate_jsonl(path)
            self.assertEqual(report["rows"], 3)
            self.assertEqual(report["episodes"], 2)
            self.assertFalse(report["classification_training_authorized"])
            self.assertFalse(report["pre_probe_attestation_independently_verified"])
            self.assertNotIn("features", report)
            self.assertNotIn("tokens", report)
            self.assertEqual(len(report["data_sha256"]), 64)

    def test_duplicates_mixed_revisions_empty_and_long_lines_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "derived.jsonl"
            good = valid_record()
            path.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                contract.validate_jsonl(path)
            other = valid_record("c" * 64)
            other["model_revision_sha256"] = "d" * 64
            path.write_text(json.dumps(good) + "\n" + json.dumps(other) + "\n")
            with self.assertRaisesRegex(ValueError, "mixed model revisions"):
                contract.validate_jsonl(path)
            path.write_text("")
            with self.assertRaisesRegex(ValueError, "empty probe"):
                contract.validate_jsonl(path)
            path.write_text("x" * 5000 + "\n")
            with self.assertRaisesRegex(ValueError, "oversized"):
                contract.validate_jsonl(path)


if __name__ == "__main__":
    unittest.main()
