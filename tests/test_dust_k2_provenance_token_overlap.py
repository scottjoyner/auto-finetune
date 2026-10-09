"""Offline rights-gate and token-order overlap regression tests."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dust.k2_norobots_provenance import summarize_provenance
from experiments.dust.k2_token_set_overlap import compare_normalized_prompts


class ProvenanceSafetyTests(unittest.TestCase):
    def test_provenance_labels_never_promote_data_rights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"no-robots.jsonl"
            path.write_text(
                '{"source":"norobots/Generation","messages":[]}\n'
                '{"source":"norobots/Rewrite","messages":[]}\n')
            import hashlib
            report=summarize_provenance(
                path,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(report["recognized_norobots_labeled_rows"],2)
            self.assertEqual(report["candidate_upstream_license"],"UNVERIFIED")
            self.assertFalse(report["commercial_training_rights_approved"])
            self.assertFalse(report["classifier_training_authorized"])
            self.assertFalse(report["local_corpus_transformation_provenance_verified"])
            self.assertNotIn('"messages"',str(report))

    def test_full_9500_row_metadata_supports_candidate_upstream_only(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"unmodified-frozen-source-format.jsonl"
            path.write_text('{"source":"norobots/Generation","messages":[]}\n'*9500)
            report=summarize_provenance(
                path,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(report["local_conversation_rows"],9500)
            self.assertEqual(report["recognized_norobots_labeled_rows"],9500)
            self.assertEqual(report["candidate_upstream_license"],"CC-BY-NC-4.0")
            self.assertEqual(report["candidate_upstream_dataset"],
                             "HuggingFaceH4/no_robots")
            self.assertFalse(report["local_corpus_license_grant_verified"])
            self.assertFalse(report["commercial_training_rights_approved"])
            self.assertFalse(report["classifier_training_authorized"])

    def test_metadata_without_provenance_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            import hashlib
            path=Path(tmp)/"corpus.jsonl"
            path.write_text('{"source":"unknown","messages":[]}\n')
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            report=summarize_provenance(path,source_sha256=sha)
            self.assertEqual(report["recognized_norobots_labeled_rows"],0)
            self.assertEqual(report["unexpected_source_category_rows"],1)
            with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                summarize_provenance(path,source_sha256="a"*64)


class TokenOrderTriageTests(unittest.TestCase):
    def test_reordered_paraphrase_candidate_detected(self):
        source=["how to audit a dataset using offline models"]
        other=["using offline models how to audit a dataset"]
        from difflib import SequenceMatcher
        # The triage must not rely on character-level order.
        result=compare_normalized_prompts(
            source,other,source_sha256="a"*64,auxiliary_sha256="b"*64)
        self.assertEqual(result["high_token_overlap_pairs"],1)
        self.assertEqual(result["unique_original_prompts_with_high_overlap"],1)
        self.assertEqual(result["all_source_auxiliary_prompt_pairs"],1)
        self.assertTrue(result["pair_accounting_complete"])
        self.assertFalse(result["semantic_paraphrase_detection_established"])
        self.assertFalse(result["classifier_training_authorized"])

    def test_no_overlap_cannot_claim_semantic_independence(self):
        result=compare_normalized_prompts(
            ["what is the price of tomorrow's electricity"],
            ["how do tropical butterflies evolve vibrant wing colors"],
            source_sha256="a"*64,auxiliary_sha256="b"*64)
        self.assertEqual(result["high_token_overlap_pairs"],0)
        self.assertFalse(result["semantic_paraphrase_detection_established"])
        self.assertFalse(result["new_source_partitions_authorized"])

    def test_short_and_malformed_sources_denied(self):
        result=compare_normalized_prompts(
            ["hi world"],["hello world"],source_sha256="a"*64,
            auxiliary_sha256="b"*64)
        self.assertEqual(result["source_prompts_too_short_for_token_triage"],1)
        with self.assertRaisesRegex(ValueError,"corpus size"):
            compare_normalized_prompts([],["hello"],source_sha256="a"*64,
                                       auxiliary_sha256="b"*64)
        with self.assertRaisesRegex(ValueError,"SHA256"):
            compare_normalized_prompts(["x"],["y"],source_sha256="bad",
                                       auxiliary_sha256="b"*64)

if __name__=="__main__":
    unittest.main()
