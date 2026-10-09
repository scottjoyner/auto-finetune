"""Fail-closed lexical group and private research candidate tests."""
from __future__ import annotations

from difflib import SequenceMatcher
import itertools
import unittest

from experiments.dust.k2_auxiliary_prompt_clusters import (
    build_private_manifest, lexical_components, THRESHOLD,
)


class AuxiliaryPromptClusteringTests(unittest.TestCase):
    def test_exhaustive_components_match_all_pair_brute_force(self):
        prompts = [
            "how to improve local fine tuning quality in a tiny gpu",
            "how to improve local fine tuning quality in a tiny gpu!",
            "why do bananas grow in warm climates",
            "why do bananas grow in warm climate?",
            "a vastly different request about test sample isolation",
            "explain whether hardware inference can run on amd gfx1201",
        ]
        components, evidence = lexical_components(prompts)
        root = list(range(len(prompts)))
        def find(i):
            while root[i] != i:
                root[i] = root[root[i]]
                i = root[i]
            return i
        for i, j in itertools.combinations(range(len(prompts)), 2):
            if SequenceMatcher(None, prompts[i], prompts[j],
                               autojunk=False).ratio() >= THRESHOLD:
                root[find(i)] = find(j)
        expected = {
            frozenset(i for i in range(len(prompts)) if find(i) == find(j))
            for j in range(len(prompts))
        }
        actual = {frozenset(members) for members in components}
        self.assertEqual(actual, expected)
        self.assertTrue(evidence["candidate_pair_accounting_complete"])
        self.assertEqual(evidence["all_unordered_distinct_prompt_pairs"], 15)
        self.assertEqual(evidence["unexamined_pairs_due_to_approximate_index"], 0)

    def test_duplicate_response_variants_are_one_candidate_group(self):
        key = b"z" * 32
        row = {
            "sample_index": 0,
            "normalized_prompt": "could we build a source duplicate checker",
            "prompt_token_count": 20,
            "assistant_token_count": 12,
        }
        result = build_private_manifest(
            [row, {**row, "sample_index": 1},
             {"sample_index": 2,
              "normalized_prompt": "why does the weather change at dawn",
              "prompt_token_count": 11,
              "assistant_token_count": 8}],
            key, source_sha="a"*64, config_sha="b"*64)
        self.assertEqual(result["source_tokenizable_pair_rows"], 3)
        self.assertEqual(result["exact_unique_normalized_prompts"], 2)
        self.assertEqual(result["response_variant_excess_rows"], 1)
        self.assertEqual(result["candidate_entries"][0]["paired_response_variants"], 2)
        self.assertFalse(result["classifier_training_authorized"])
        self.assertFalse(result["source_partition_authorized"])
        self.assertNotIn("could we build", str(result))
        self.assertTrue(all(len(x["prompt_hmac_sha256"]) == 64
                            for x in result["candidate_entries"]))

    def test_bad_deadline_or_duplicate_input_fails_closed(self):
        with self.assertRaises(ValueError):
            lexical_components(["same", "same"])
        with self.assertRaisesRegex(ValueError, "bounded deadline"):
            lexical_components(["one"], cutoff_seconds=0)
        with self.assertRaisesRegex(ValueError, "frozen lexical"):
            lexical_components(["one"], threshold=.80)
        with self.assertRaisesRegex(ValueError, "candidate count"):
            lexical_components([])
        with self.assertRaisesRegex(ValueError, "source and tokenizer"):
            build_private_manifest(
                [{"sample_index": 0, "normalized_prompt": "any text",
                  "prompt_token_count": 4, "assistant_token_count": 3}],
                b"0"*32, source_sha="unhashed", config_sha="b"*64)

    def test_transitive_near_duplicate_family_cannot_split(self):
        prompts = [
            "how do I make an android application run quickly",
            "how do I make an android application run quickly?",
            "how do I make an android application run quickly??",
        ]
        groups, _ = lexical_components(prompts)
        self.assertEqual(len(groups), 1)
        rows = [
            {"sample_index": i, "normalized_prompt": prompt,
             "prompt_token_count": 30, "assistant_token_count": 4}
            for i, prompt in enumerate(prompts)
        ]
        result = build_private_manifest(
            rows, b"x"*32, source_sha="a"*64, config_sha="b"*64)
        self.assertEqual(result["lexical_source_components"], 1)
        groups = {x["near_duplicate_cluster_hmac_sha256"]
                  for x in result["candidate_entries"]}
        splits = {x["partition_candidate_only"]
                  for x in result["candidate_entries"]}
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(splits), 1)
        self.assertTrue(all(not x["heldout_eligible"]
                            for x in result["candidate_entries"]))


if __name__ == "__main__":
    unittest.main()
