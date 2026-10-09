"""Lineage classification prevents derivative corpora becoming fake holdouts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dust.k2_cohort_preflight import SCHEMA
from experiments.dust.k2_cross_corpus_audit import summarize_screen
from experiments.dust.k2_source_lineage_gate import (
    classify_source, source_lineage_audit,
)


class SourceLineageTests(unittest.TestCase):
    def fixture(self, root: Path):
        prompts = [
            "What is a reproducible K2 stochastic observation?",
            "Explain why different prompt groups must stay disjoint.",
        ]
        rows = [
            {"sample_index": i, "normalized_prompt": prompt,
             "episode_hmac_sha256": str(i + 1) * 64}
            for i, prompt in enumerate(prompts)
        ]
        manifest = {
            "schema": SCHEMA,
            "source_dataset_sha256": "a" * 64,
            "model_config_sha256": "b" * 64,
            "candidates": [
                {
                    "sample_index": i,
                    "episode_hmac_sha256": row["episode_hmac_sha256"],
                    "near_duplicate_cluster_sha256":
                        ("c" if i == 0 else "d") * 64,
                    "group_split": "train" if i == 0 else "test",
                    "disposition": "ELIGIBLE",
                } for i, row in enumerate(rows)
            ],
        }
        source = root / "cohort.json"
        source.write_text(json.dumps(manifest, sort_keys=True))
        source.chmod(0o600)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        return rows, manifest, source, digest

    def audit(self, root, rows, manifest, manifest_sha,
              filename: str, aux: list[str], sha: str):
        report = summarize_screen(
            rows, manifest,
            [(filename, aux, sha, len(aux))],
            expected_source_sha256="a" * 64,
            auxiliary_pair_cap=16384)
        report["source_preflight_sha256"] = manifest_sha
        report["source_model_config_sha256"] = "b" * 64
        path = root / (filename + ".audit.json")
        path.write_text(json.dumps(report, sort_keys=True))
        path.chmod(0o600)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_derivative_exact_superposition_not_a_new_heldout_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, manifest, source, digest = self.fixture(root)
            mixed = self.audit(
                root, rows, manifest, digest, "train.mixed.jsonl",
                [r["normalized_prompt"] for r in rows], "e" * 64)
            clean = self.audit(
                root, rows, manifest, digest, "general-norobots.jsonl",
                ["Unrelated daytime weather patterns for desert habitats"],
                "f" * 64)
            outcome = source_lineage_audit(source, digest, [mixed, clean])
            self.assertEqual(outcome["audited_corpora"], 2)
            self.assertEqual(outcome["full_lexical_corpus_scans"], 2)
            self.assertEqual(outcome["lexically_overlapping_existing_clusters"], 2)
            by_name = {x["filename"]: x for x in outcome["sources"]}
            self.assertEqual(
                by_name["train.mixed.jsonl"]["lineage_observation"],
                "CONTAINS_ALL_CANDIDATE_PROMPTS_EXACTLY")
            self.assertEqual(
                by_name["general-norobots.jsonl"]["lineage_observation"],
                "NO_DETECTED_LEXICAL_OVERLAP__NOT_INDEPENDENCE_PROOF")
            self.assertTrue(all(
                not x["eligible_as_independent_heldout"]
                for x in outcome["sources"]))
            self.assertFalse(outcome["classifier_training_authorized"])
            self.assertFalse(outcome["source_independence_certified"])
            self.assertNotIn(rows[0]["normalized_prompt"], str(outcome))
            self.assertNotIn(rows[0]["episode_hmac_sha256"], str(outcome))

    def test_reject_duplicate_and_tampered_audit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows, manifest, source, digest = self.fixture(root)
            audit = self.audit(
                root, rows, manifest, digest, "train.mixed.jsonl",
                [x["normalized_prompt"] for x in rows], "e" * 64)
            with self.assertRaisesRegex(ValueError, "counted more than once"):
                source_lineage_audit(source, digest, [audit, audit])
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                source_lineage_audit(source, digest, [(audit[0], "0" * 64)])
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                source_lineage_audit(source, "f" * 64, [audit])

    def test_overlap_beyond_partial_coverage_never_claims_new_source(self):
        self.assertEqual(
            classify_source(candidate_rows=75, exact_matches=75,
                            lexical_matches=75, unique_pairs=6400,
                            screened=6400, truncated=False),
            "CONTAINS_ALL_CANDIDATE_PROMPTS_EXACTLY")
        self.assertEqual(
            classify_source(candidate_rows=75, exact_matches=0,
                            lexical_matches=0, unique_pairs=11037,
                            screened=512, truncated=True),
            "UNFINISHED_LEXICAL_SCAN")
        self.assertEqual(
            classify_source(candidate_rows=75, exact_matches=54,
                            lexical_matches=55, unique_pairs=283,
                            screened=283, truncated=False),
            "SHARES_CANDIDATE_PROMPTS")
        with self.assertRaises(ValueError):
            classify_source(candidate_rows=75, exact_matches=2,
                            lexical_matches=1, unique_pairs=512,
                            screened=512, truncated=False)


if __name__ == "__main__":
    unittest.main()
