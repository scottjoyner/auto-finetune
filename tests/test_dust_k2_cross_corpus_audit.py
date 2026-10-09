"""Synthetic-only lexical cross-corpus screening checks."""
import unittest
from experiments.dust.k2_cross_corpus_audit import lexical_matches, summarize_screen
from experiments.dust.k2_cohort_preflight import SCHEMA

class TestLexicalAudit(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"sample_index": 0, "normalized_prompt": "the quick brown fox jumps over the fence", "episode_hmac_sha256": "a"*64},
            {"sample_index": 1, "normalized_prompt": "the quick brown fox jumps over the fence?", "episode_hmac_sha256": "b"*64},
            {"sample_index": 2, "normalized_prompt": "the blue whale swims in deep seawater", "episode_hmac_sha256": "c"*64},
        ]
        self.manifest = {
            "schema": SCHEMA, "source_dataset_sha256": "e"*64,
            "candidates": [
                {"sample_index": i, "episode_hmac_sha256": row["episode_hmac_sha256"],
                 "near_duplicate_cluster_sha256": ("d"*64 if i<2 else "f"*64),
                 "group_split": ("train", "test", "validation")[i]}
                for i, row in enumerate(self.rows)
            ],
        }

    def test_entire_cluster_quarantined_by_one_cross_corpus_match(self):
        report = summarize_screen(self.rows, self.manifest,
            [("aux.jsonl", ["the quick brown fox jumps over the fence"],
              "1"*64, 1)], expected_source_sha256="e"*64)
        self.assertEqual(report["quarantine_sample_indices"], [0, 1])
        self.assertEqual(report["quarantine_near_duplicate_clusters"], 1)
        self.assertFalse(report["classifier_training_authorized"])
        self.assertNotIn("the quick brown fox", str(report))

    def test_clean_lexical_audit_never_certifies_semantic_independence(self):
        report = summarize_screen(self.rows, self.manifest,
            [("aux.jsonl", ["volcanic stone and hot magma"], "1"*64, 1)],
            expected_source_sha256="e"*64)
        self.assertEqual(report["quarantine_sample_indices"], [])
        self.assertFalse(report["source_independence_certified"])
        self.assertFalse(report["semantic_paraphrase_screened"])

    def test_source_drift_and_aux_cap_fail_closed(self):
        with self.assertRaises(ValueError):
            summarize_screen(self.rows, self.manifest, [],
                             expected_source_sha256="0"*64)
        with self.assertRaises(ValueError):
            summarize_screen(self.rows, self.manifest,
                [("aux.jsonl", ["hello"]*513, "1"*64, 513)],
                expected_source_sha256="e"*64)

if __name__ == "__main__":
    unittest.main()
