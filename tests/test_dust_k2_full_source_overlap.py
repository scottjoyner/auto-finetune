"""Full source vs auxiliary screening uses all prompts, not selected 128-token subset."""
import unittest

from experiments.dust.k2_full_source_overlap import screen_full_source_prompts


class FullSourceOverlapTests(unittest.TestCase):
    def test_overlaps_outside_tokenizable_subset_are_still_detected(self):
        source=[
            "what is an approximate nearest neighbor benchmark",
            "explain a rare syntax edge case in this research",
            "why is a tropical bird colorful in bright light",
        ]
        auxiliary=[
            "something wholly unrelated to model inference",
            source[2],
            source[1]+"!",
        ]
        result=screen_full_source_prompts(
            source,auxiliary,source_sha="a"*64,auxiliary_sha="b"*64)
        self.assertEqual(result["lexically_overlapping_original_prompt_count"],2)
        self.assertEqual(result["exactly_matching_original_prompt_count"],1)
        self.assertTrue(result["all_source_pairs_examined"])
        self.assertFalse(result["classifier_training_authorized"])
        self.assertFalse(result["dataset_rights_approved"])
        self.assertNotIn(source[1],str(result))

    def test_exact_same_prompt_different_responses_count_once(self):
        source=["duplicate source prompt"]*3+["other totally different topic"]
        auxiliary=["duplicate source prompt"]
        result=screen_full_source_prompts(
            source,auxiliary,source_sha="a"*64,auxiliary_sha="b"*64)
        self.assertEqual(result["source_unique_paired_rows"],4)
        self.assertEqual(result["source_exact_distinct_prompts"],2)
        self.assertEqual(result["lexically_overlapping_original_prompt_count"],1)

    def test_bound_and_bad_digest_denied(self):
        with self.assertRaises(ValueError):
            screen_full_source_prompts(
                ["valid source"],["auxiliary"],source_sha="untrusted",
                auxiliary_sha="b"*64)
        with self.assertRaises(ValueError):
            screen_full_source_prompts(
                ["valid source"],[],source_sha="a"*64,
                auxiliary_sha="b"*64)

if __name__=="__main__":
    unittest.main()
