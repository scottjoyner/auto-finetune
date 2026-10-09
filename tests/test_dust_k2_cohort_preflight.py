"""Source-cluster privacy, split isolation, and manifest admission tests."""
from __future__ import annotations

import unittest

from experiments.dust.k2_cohort_preflight import (
    authorize_index, build_manifest, cluster_records, SCHEMA,
)


def example(index, prompt, group, split):
    return {
        "sample_index": index,
        "normalized_prompt": prompt,
        "episode_hmac_sha256": group * 64,
        "partition": split,
        "prompt_token_count": 12,
        "assistant_token_count": 17,
    }


class CohortPreflightTests(unittest.TestCase):
    def data(self):
        return [
            example(0, "how do i train my model using cached gradients",
                    "a", "train"),
            example(1, "how do i train my model using cached gradients?",
                    "b", "test"),
            example(2, "what does rocM mean for an amd gpu deployment",
                    "c", "train"),
            example(3, "how to use a python dict in function definitions",
                    "d", "validation"),
            example(4, "how to use a python dict in function definitions",
                    "d", "validation"),
        ]

    def test_near_duplicate_across_splits_all_quarantined(self):
        raw = self.data()
        clusters = cluster_records(raw)
        self.assertEqual(len(clusters), 3)
        manifest = build_manifest(raw, source_sha256="f" * 64,
                                  model_config_sha256="e" * 64)
        candidates = manifest["candidates"]
        self.assertEqual(candidates[0]["disposition"],
                         "QUARANTINE_NEAR_DUPLICATE_CROSS_SPLIT")
        self.assertEqual(candidates[1]["disposition"],
                         "QUARANTINE_NEAR_DUPLICATE_CROSS_SPLIT")
        self.assertEqual(manifest["independent_cluster_counts"]
                         ["quarantined_clusters"], 1)
        self.assertEqual(manifest["independent_cluster_counts"]
                         ["validation_clusters"], 1)
        self.assertEqual(manifest["independent_cluster_counts"]
                         ["train_clusters"], 1)

    def test_no_raw_prompt_text_or_tokens_in_manifest(self):
        manifest = build_manifest(self.data(), source_sha256="f" * 64,
                                  model_config_sha256="e" * 64)
        text = str(manifest)
        self.assertNotIn("cached gradients", text)
        self.assertNotIn("rocM", text)
        self.assertNotIn("normalized_prompt", text)
        self.assertFalse(manifest["classifier_training_authorized"])
        self.assertFalse(manifest["independent_split_acceptance"])

    def test_admission_rejects_quarantine_mismatched_hmac_and_source(self):
        manifest = build_manifest(self.data(), source_sha256="f" * 64,
                                  model_config_sha256="e" * 64)
        valid = authorize_index(
            manifest, source_sha256="f" * 64,
            model_config_sha256="e" * 64,
            sample_index=2, max_tokens=128,
            episode_hmac_sha256="c" * 64)
        self.assertEqual(valid["group_split"], "train")
        with self.assertRaisesRegex(ValueError, "not individually approved"):
            authorize_index(manifest, source_sha256="f" * 64,
                            model_config_sha256="e" * 64,
                            sample_index=0, max_tokens=128,
                            episode_hmac_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "HMAC changed"):
            authorize_index(manifest, source_sha256="f" * 64,
                            model_config_sha256="e" * 64,
                            sample_index=2, max_tokens=128,
                            episode_hmac_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "drift"):
            authorize_index(manifest, source_sha256="0" * 64,
                            model_config_sha256="e" * 64,
                            sample_index=2, max_tokens=128,
                            episode_hmac_sha256="c" * 64)

    def test_indices_beyond_bounded_model_runner_not_admitted(self):
        data = [example(3, "much longer unique statement about biology",
                        "a", "train"),
                example(65, "different unique statement on hardware io",
                        "b", "test")]
        result = build_manifest(data, source_sha256="f" * 64,
                                model_config_sha256="e" * 64)
        self.assertEqual(result["candidates"][1]["disposition"],
                         "OUTSIDE_BOUNDED_RUNNER_INDEX")
        with self.assertRaises(ValueError):
            authorize_index(result, source_sha256="f" * 64,
                            model_config_sha256="e" * 64,
                            sample_index=65, max_tokens=128,
                            episode_hmac_sha256="b" * 64)

    def test_rejects_duplicate_indices_and_incorrect_token_cap(self):
        with self.assertRaises(ValueError):
            build_manifest([example(0, "a long special unique sentence a",
                                    "a", "train"),
                            example(0, "a long special unique sentence b",
                                    "b", "test")],
                           source_sha256="f" * 64,
                           model_config_sha256="e" * 64)
        with self.assertRaisesRegex(ValueError, "128"):
            build_manifest([example(0, "unique sentence", "a", "train")],
                           source_sha256="f" * 64,
                           model_config_sha256="e" * 64,
                           max_tokens=96)

    def test_train_validation_test_source_group_schema_unchanged(self):
        result = build_manifest(self.data(), source_sha256="f" * 64,
                                model_config_sha256="e" * 64)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertTrue(all(r["source_group_schema"] ==
                            "masked-prompt-prefix-v2"
                            for r in result["candidates"]))


if __name__ == "__main__":
    unittest.main()
