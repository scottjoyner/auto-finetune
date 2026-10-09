"""Bounded full-auxiliary lexical screening: integrity and no overclaim."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import unittest

from experiments.dust.k2_cohort_preflight import SCHEMA
from experiments.dust.k2_cross_corpus_audit import (
    MAX_EXTENDED_AUXILIARY_PAIR_CAP, lexical_matches, summarize_screen)
from experiments.dust.k2_crosscorpus_veto import bind_quarantine_overlay


class ExtendedAuxiliaryTests(unittest.TestCase):
    def setUp(self):
        self.prompt = "How do we keep the same user prompt isolated from training"
        self.rows = [
            {"sample_index": 0, "normalized_prompt": self.prompt,
             "episode_hmac_sha256": "a" * 64},
            {"sample_index": 1,
             "normalized_prompt": "Rare aurora colors in a desert valley at dawn",
             "episode_hmac_sha256": "b" * 64},
        ]
        self.manifest = {
            "schema": SCHEMA,
            "source_dataset_sha256": "c" * 64,
            "model_config_sha256": "d" * 64,
            "candidates": [
                {
                    "sample_index": i,
                    "near_duplicate_cluster_sha256": ("e" if i == 0 else "f") * 64,
                    "episode_hmac_sha256": row["episode_hmac_sha256"],
                    "group_split": "train" if i == 0 else "test",
                    "disposition": "ELIGIBLE",
                } for i, row in enumerate(self.rows)
            ],
        }

    def report(self, count=800, available=None):
        auxiliary = [f"Unrelated corpus phrase {i:05d}" for i in range(count)]
        auxiliary[-1] = self.prompt
        return summarize_screen(
            self.rows, self.manifest,
            [("auxiliary.jsonl", auxiliary, "1" * 64,
              count if available is None else available)],
            expected_source_sha256="c" * 64, auxiliary_pair_cap=16384,
        )

    def test_extended_cap_catches_overlap_beyond_legacy_first_512(self):
        old = summarize_screen(
            self.rows, self.manifest,
            [("auxiliary.jsonl",
              [f"Unrelated corpus phrase {i:05d}" for i in range(512)],
              "1" * 64, 800)],
            expected_source_sha256="c" * 64,
        )
        self.assertEqual(old["cross_corpus_matched_source_indices"], 0)
        self.assertFalse(old["complete_auxiliary_coverage"])
        expanded = self.report()
        self.assertEqual(expanded["cross_corpus_matched_source_indices"], 1)
        self.assertEqual(expanded["quarantine_sample_indices"], [0])
        self.assertEqual(expanded["auxiliary_pair_cap"], 16384)
        self.assertTrue(expanded["complete_auxiliary_coverage"])
        self.assertFalse(expanded["semantic_paraphrase_screened"])
        self.assertFalse(expanded["source_independence_certified"])
        self.assertFalse(expanded["classifier_training_authorized"])

    def test_rejects_fake_cap_and_full_coverage(self):
        with self.assertRaisesRegex(ValueError, "bounded"):
            summarize_screen(
                self.rows, self.manifest,
                [("a.jsonl", [self.prompt], "1" * 64, 1)],
                expected_source_sha256="c" * 64,
                auxiliary_pair_cap=MAX_EXTENDED_AUXILIARY_PAIR_CAP + 1,
            )
        with self.assertRaisesRegex(ValueError, "bounded"):
            summarize_screen(
                self.rows, self.manifest,
                [("a.jsonl", [self.prompt], "1" * 64, 1)],
                expected_source_sha256="c" * 64,
                auxiliary_pair_cap=True,
            )

    def test_sha_bound_extended_report_enforces_truncation_and_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            report = self.report(count=513, available=11037)
            cohort_sha = hashlib.sha256(json.dumps(
                self.manifest, sort_keys=True).encode()).hexdigest()
            report["source_preflight_sha256"] = cohort_sha
            report["source_model_config_sha256"] = self.manifest[
                "model_config_sha256"]
            def store():
                path.write_text(json.dumps(report, sort_keys=True))
                path.chmod(0o600)
                return hashlib.sha256(path.read_bytes()).hexdigest()
            sha = store()
            overlay = bind_quarantine_overlay(
                self.manifest, cohort_sha, path, sha)
            self.assertFalse(overlay["lexical_scan_complete_for_supplied_corpora"])
            self.assertEqual(overlay["truncated_auxiliary_corpora"], 1)
            self.assertEqual(overlay["quarantined_clusters"], {"e" * 64})
            report["auxiliary_pair_cap"] = 512
            with self.assertRaisesRegex(ValueError, "truncation"):
                bind_quarantine_overlay(
                    self.manifest, cohort_sha, path, store())
            report["auxiliary_pair_cap"] = 16384
            report["aux_corpora"][0]["truncated"] = False
            with self.assertRaisesRegex(ValueError, "truncation"):
                bind_quarantine_overlay(
                    self.manifest, cohort_sha, path, store())

    def test_quick_ratio_pruning_preserves_full_frozen_threshold(self):
        import difflib
        source = self.rows
        target = [
            self.prompt,
            self.prompt + "!",
            "Entirely different details of cloud operations",
            "Rare aurora colors in a desert valley at dawn!",
        ]
        expected = {}
        for record in source:
            best = max(difflib.SequenceMatcher(
                None, record["normalized_prompt"], other,
                autojunk=False).ratio() for other in target)
            if best >= .85:
                expected[record["sample_index"]] = best
        actual = lexical_matches(source, target)
        self.assertEqual(set(actual), set(expected))
        for index in expected:
            self.assertAlmostEqual(
                actual[index]["lexical_similarity_lower_bound"],
                expected[index], delta=.0001)


if __name__ == "__main__":
    unittest.main()
