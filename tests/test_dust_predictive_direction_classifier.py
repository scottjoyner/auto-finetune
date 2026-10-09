"""Fail-closed tests for shadow-only predictive direction classification."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from experiments.dust import predictive_direction_classifier as research


class SyntheticClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = research.split_fixture()

    def test_splits_are_episode_disjoint(self):
        partitions = [set(r.episode for r in rows) for rows in
                      self.splits.values()]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])
        self.assertEqual([len(rows) for rows in self.splits.values()],
                         [1024, 256, 256])

    def test_outcomes_are_consistent_with_gain_and_candidates(self):
        for rows in self.splits.values():
            for row in rows:
                self.assertEqual(row.useful, int(row.actual_gain > 0))
                self.assertEqual(len(row.features),
                                 len(research.FEATURE_NAMES))
                self.assertTrue(all(math.isfinite(f) for f in row.features))
        a = research.make_episode(42)
        self.assertEqual(a, research.make_episode(42))
        self.assertNotEqual(a, research.make_episode(43))

    def test_training_only_episode_policy(self):
        norm = research.Standardizer.from_training(self.splits["train"])
        with self.assertRaisesRegex(ValueError, "non-train"):
            research.train_head(self.splits["validation"], norm, 7)
        with self.assertRaisesRegex(ValueError, "validation only"):
            research.choose_threshold([], norm, self.splits["train"])
        with self.assertRaisesRegex(ValueError, "non-test"):
            research.evaluate(self.splits["validation"], [], norm, .5)

    def test_feature_schema_and_nan_rejected(self):
        norm = research.Standardizer.from_training(self.splits["train"])
        with self.assertRaisesRegex(ValueError, "schema"):
            norm.transform((1., 2.))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            norm.transform((float("nan"),) * len(research.FEATURE_NAMES))

    def test_same_seed_weights_reproduce_and_other_seed_differs(self):
        norm = research.Standardizer.from_training(self.splits["train"])
        a = research.train_head(self.splits["train"], norm, 7, epochs=12)
        b = research.train_head(self.splits["train"], norm, 7, epochs=12)
        c = research.train_head(self.splits["train"], norm, 42, epochs=12)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), len(research.FEATURE_NAMES) + 1)

    def test_baseline_ranking_does_not_depend_on_classifier(self):
        held = self.splits["test"]
        first = research.selected_metrics(held,
                                          lambda row: row.momentum_score)
        second = research.selected_metrics(held,
                                           lambda row: row.momentum_score)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["precision_at_4"], 0)
        self.assertLessEqual(first["precision_at_4"], 1)

    def test_end_to_end_shadow_evidence_no_production_authority(self):
        report = research.run_synthetic()
        self.assertEqual(report["schema"],
                         "auto-finetune.dust-predictive-direction.synthetic.v1")
        self.assertTrue(report["research_only"])
        self.assertFalse(report["trainer_admission_authorized"])
        self.assertFalse(report["model_checkpoint_promotion"])
        self.assertFalse(report["pretrained_model_loaded"])
        self.assertEqual(report["optimizer_updates"], 0)
        self.assertEqual(report["hosted_provider_calls"], 0)
        self.assertEqual(report["backward_calls"], 0)
        self.assertEqual(report["gpu_seconds"], 0)
        self.assertEqual(report, research.run_synthetic())
        self.assertEqual(report["confirmation_episodes"], 48)
        self.assertTrue(report["test_is_development_exposed"])
        self.assertTrue(report["confirmation_is_first_view"])
        self.assertFalse(set(research.CONFIRMATION_EPISODES) &
                         set().union(*map(set, research.SPLIT_EPISODES.values())))
        self.assertIn(report["confirmation_metrics"]["confirmation_verdict"],
                      ("CONFIRMATORY_TOY_ADVANTAGE",
                       "NO_EVIDENCE_OF_SUPERIORITY"))
        self.assertEqual(
            report["confirmation_metrics"]["paired_episode_bootstrap"]
                  ["classifier_minus_curvature_precision_at_4"]
                  ["per_episode_count"], 48)
        m = report["metrics"]
        self.assertIn(m["shadow_recommendation"],
                      ("CONTINUE_SHADOW", "HOLD_NO_BASELINE_ADVANTAGE"))
        self.assertEqual(len(report["episode_split_sha256"]), 3)
        self.assertEqual(m["production_admission"],
                         "HOLD_NO_REAL_LABELS_OR_HELDOUT_CE")
        self.assertEqual(m["paired_episode_bootstrap"]["resample_unit"],
                         "independent_synthetic_episode")
        intervals = m["paired_episode_bootstrap"]
        for name in ("classifier_minus_curvature_precision_at_4",
                     "classifier_minus_curvature_true_gain_at_4"):
            lo, hi = intervals[name]["ci95"]
            self.assertLessEqual(lo, intervals[name]["mean_delta"])
            self.assertGreaterEqual(hi, intervals[name]["mean_delta"])
            self.assertLessEqual(lo, 0)
            self.assertGreaterEqual(hi, 0)
        for method in ("classifier", "momentum", "curvature_aware_momentum",
                       "random"):
            self.assertTrue(math.isfinite(
                m[method]["mean_true_gain_at_4"]))

    def test_cli_requires_explicit_synthetic_opt_in_and_never_overwrites(self):
        with self.assertRaises(SystemExit) as e:
            research.main([])
        self.assertEqual(e.exception.code, 2)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "synthetic-results.json"
            research.main(["--synthetic-only", "--output", str(path)])
            doc = json.loads(path.read_text())
            self.assertTrue(doc["research_only"])
            with self.assertRaises(FileExistsError):
                research.main(["--synthetic-only", "--output", str(path)])


if __name__ == "__main__":
    unittest.main()
