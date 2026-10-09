"""Synthetic-only, confidential-data-free cross-corpus veto regressions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_dust_k2_feature16_readiness import sample_inputs, check
from experiments.dust.k2_cross_corpus_audit import summarize_screen
from experiments.dust.k2_crosscorpus_veto import bind_quarantine_overlay
from experiments.dust.k2_feature16_readiness import evaluate_readiness


def fixture(folder: Path, *, collision: bool = False,
            truncated: bool = False):
    private, manifest_path, manifest_sha = sample_inputs(folder)
    manifest = json.loads(manifest_path.read_text())
    approved = manifest["candidates"][0]
    prompt = "synthetic short training prompt that never reaches production"
    source = [{
        "sample_index": approved["sample_index"],
        "normalized_prompt": prompt,
        "episode_hmac_sha256": approved["episode_hmac_sha256"],
    }]
    # The audit checks sorted tokenizable-row indices, not arbitrary 1-based
    # sample IDs; prior single-source fixture uses index 1, normalize both.
    approved["sample_index"] = 0
    source[0]["sample_index"] = 0
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    auxiliary = [prompt] if collision else ["wholly unrelated ocean fact"]
    aux_count = 513 if truncated else len(auxiliary)
     # The source hash is synthetic and pinned inside the manifest.
    manifest["source_dataset_sha256"] = "f" * 64
    manifest["model_config_sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    report = summarize_screen(
        source, manifest,
        [("auxiliary.jsonl", auxiliary, "d" * 64, aux_count)],
        expected_source_sha256="f" * 64)
    report["source_preflight_sha256"] = manifest_sha
    report["source_model_config_sha256"] = "b" * 64
    path = folder / "audit.json"
    path.write_text(json.dumps(report, sort_keys=True))
    path.chmod(0o600)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return private, manifest_path, manifest_sha, path, digest


def readiness(private, manifest_path, manifest_sha, audit_path, audit_sha):
    return evaluate_readiness(
        [private], manifest_path,
        expected_manifest_sha=manifest_sha,
        expected_model_sha="b" * 64,
        cross_corpus_audit_path=audit_path,
        expected_cross_corpus_sha=audit_sha,
    )


class CrossCorpusVetoTests(unittest.TestCase):
    def test_cross_corpus_match_rejects_already_observed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), collision=True)
            with self.assertRaisesRegex(ValueError, "quarantined by cross-corpus"):
                readiness(*args)
            overlay = bind_quarantine_overlay(
                json.loads(args[1].read_text()), args[2], args[3], args[4])
            self.assertEqual(len(overlay["quarantined_clusters"]), 1)

    def test_no_lexical_match_not_semantic_or_classifier_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            report = readiness(*args)
            self.assertEqual(report["complete_direction_rows"], 8)
            self.assertEqual(report["cross_corpus_quarantined_source_clusters"], 0)
            self.assertEqual(
                report["cross_corpus_lexical_audit_coverage"],
                "COMPLETE_FOR_SUPPLIED_CORPORA_ONLY")
            self.assertFalse(report["independent_audit_of_cross_corpus_matching"])
            self.assertFalse(report["real_label_classifier_training_authorized"])
            self.assertFalse(report["cross_corpus_semantic_contamination_verified"])

    def test_truncated_auxiliary_report_never_certifies_full_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), truncated=True)
            result = readiness(*args)
            self.assertEqual(
                result["cross_corpus_lexical_audit_coverage"],
                "INCOMPLETE_TRUNCATED")
            self.assertFalse(result["real_label_classifier_training_authorized"])

    def test_changed_audit_or_preflight_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                readiness(args[0], args[1], args[2], args[3], "0" * 64)
            with self.assertRaisesRegex(ValueError, "cohort SHA256 changed"):
                readiness(args[0], args[1], "1" * 64,
                          args[3], args[4])

    def test_quarantine_must_cover_every_near_duplicate_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp), collision=True)
            manifest = json.loads(args[1].read_text())
            second = dict(manifest["candidates"][0])
            second["sample_index"] = 1
            second["episode_hmac_sha256"] = "c" * 64
            manifest["candidates"].append(second)
            args[1].write_text(json.dumps(manifest, sort_keys=True))
            updated_manifest_sha = hashlib.sha256(args[1].read_bytes()).hexdigest()
            audit = json.loads(args[3].read_text())
            audit["source_candidate_rows"] = 2
            audit["source_preflight_sha256"] = updated_manifest_sha
            # Deliberately leave index 1 out of a rehashed audit's veto.
            args[3].write_text(json.dumps(audit, sort_keys=True))
            updated_audit_sha = hashlib.sha256(args[3].read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "whole related prompt cluster"):
                bind_quarantine_overlay(
                    manifest, updated_manifest_sha, args[3], updated_audit_sha)

    def test_audit_without_any_scanned_corpus_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            manifest = json.loads(args[1].read_text())
            audit = json.loads(args[3].read_text())
            audit["aux_corpora"] = []
            args[3].write_text(json.dumps(audit))
            new_sha = hashlib.sha256(args[3].read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "not enough approved auxiliary"):
                bind_quarantine_overlay(manifest, args[2], args[3], new_sha)
            with self.assertRaisesRegex(ValueError, "no auxiliary corpora"):
                summarize_screen(
                    [{"sample_index": 0, "normalized_prompt": "prompt",
                      "episode_hmac_sha256": "a" * 64}],
                    manifest, [], expected_source_sha256="f" * 64)


if __name__ == "__main__":
    unittest.main()
