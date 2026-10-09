"""Frozen synthetic semantic challenge and cross-family review tests.

No torch, downloads, actual user prompts or real model needed in CI.
"""
from __future__ import annotations

import unittest

from experiments.dust.k2_semantic_challenge_contract import (
    PAIRS, THRESHOLDS, evaluate_challenge, summarize_collisions,
)


class ChallengeTests(unittest.TestCase):
    def test_prespecified_positive_hard_negative_count_and_scores(self):
        self.assertEqual(len(PAIRS), 20)
        self.assertEqual(sum(x[2] for x in PAIRS), 10)
        self.assertEqual(THRESHOLDS, (0.75, 0.85, 0.92))
        scores = [.91 if x[2] else .3 for x in PAIRS]
        report = evaluate_challenge(scores)
        self.assertTrue(report[
            "prespecified_threshold_results"]["0.85"][
            "challenge_pass_for_screening_research_only"])
        self.assertEqual(report["prespecified_threshold_results"]["0.85"][
            "true_positive"], 10)
        self.assertEqual(report["prespecified_threshold_results"]["0.85"][
            "false_positive"], 0)
        self.assertEqual(report["paired_positive_outscores_hard_negative"],10)
        self.assertEqual(report["positive_vs_negative_pairwise_auc"],1.0)
        self.assertTrue(report["ranking_metrics_are_exploratory_not_authority"])
        self.assertFalse(report["classifier_training_authorized"])

    def test_poor_model_fails_but_cannot_certify_independence(self):
        scores = [.40] * len(PAIRS)
        report = evaluate_challenge(scores)
        self.assertFalse(report["prespecified_threshold_results"]["0.75"][
            "challenge_pass_for_screening_research_only"])
        self.assertFalse(report["semantic_independence_certified"])

    def test_nan_wrong_row_count_and_out_of_bounds_denied(self):
        for scores in ([], [1.] * 19, [float("nan")] * 20,
                       [1.1] * 20):
            with self.subTest(count=len(scores)):
                with self.assertRaises(ValueError):
                    evaluate_challenge(scores)

    def test_cross_split_semantic_edges_are_counted_but_not_approved(self):
        summary = summarize_collisions(
            source_groups=["s0", "s1"],
            auxiliary_groups=["f1", "f2", "f2", "f3"],
            auxiliary_splits=["train", "validation", "validation", "test"],
            cross_hits=[(0, 0, .88), (1, 2, .90)],
            within_hits=[(0, 1, .90), (1, 2, .92), (1, 3, .86)],
            threshold=.85)
        self.assertEqual(summary["cross_corpus_embedding_edges"], 2)
        self.assertEqual(summary["within_auxiliary_new_cross_family_edges"], 2)
        self.assertEqual(summary["within_auxiliary_new_cross_split_edges"], 2)
        self.assertEqual(summary["within_auxiliary_existing_family_edges"], 1)
        self.assertEqual(summary["lexical_candidate_families_implicated"], 3)
        self.assertFalse(summary["heldout_splits_authorized"])
        self.assertFalse(summary["semantic_equivalence_certified"])

    def test_invalid_threshold_and_fabricated_similarity_denied(self):
        kwargs = dict(
            source_groups=["s0"], auxiliary_groups=["f1", "f2"],
            auxiliary_splits=["train", "test"], cross_hits=[],
            within_hits=[(0, 1, .9)])
        with self.assertRaisesRegex(ValueError, "unregistered"):
            summarize_collisions(**kwargs, threshold=.83)
        with self.assertRaisesRegex(ValueError, "invalid within-source"):
            summarize_collisions(**{**kwargs,
                "within_hits": [(1, 0, .9)]}, threshold=.85)
        with self.assertRaisesRegex(ValueError, "invalid cross-corpus"):
            summarize_collisions(**{**kwargs,
                "cross_hits": [(10, 1, .9)]}, threshold=.85)


if __name__ == "__main__":
    unittest.main()
