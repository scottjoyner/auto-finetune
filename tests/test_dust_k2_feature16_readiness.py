"""Real K2 16D dataset readiness: synthetic private evidence and denial."""
import hashlib
import json
import stat
from pathlib import Path
import tempfile
import unittest

from test_dust_k2_feature16_contract import feature_fixture
from experiments.dust.k2_feature16_contract import export_private
from experiments.dust.k2_feature16_readiness import (
    evaluate_readiness, validate_episode,
)
from experiments.dust.k2_cohort_preflight import SCHEMA as COHORT_SCHEMA
from experiments.dust.predictive_probe_contract import partition


def sample_inputs(root: Path, *, available=True):
    sources, *_ = feature_fixture(root)
    private = root / "features16.jsonl"
    export_private(*sources, private)
    row = json.loads(private.read_text().splitlines()[0])
    hmac = row["source_group_hmac_sha256"]
    manifest = root / "preflight.json"
    cohort = {
        "schema": COHORT_SCHEMA,
        "max_tokens": 128,
        "candidates": [{
            "sample_index": 1,
            "disposition": "ELIGIBLE" if available else "QUARANTINE",
            "episode_hmac_sha256": hmac,
            "near_duplicate_cluster_sha256": "e" * 64,
            "group_split": partition(hmac),
        }],
    }
    manifest.write_text(json.dumps(cohort))
    manifest.chmod(0o600)
    return private, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def check(features, manifest, manifest_hash):
    return evaluate_readiness(
        [features], manifest,
        expected_manifest_sha=manifest_hash,
        expected_model_sha="b" * 64,
    )


class Feature16ReadinessTests(unittest.TestCase):
    def test_complete_episode_is_one_group_not_eight_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            private, cohort, sha = sample_inputs(Path(temp))
            report = check(private, cohort, sha)
            self.assertEqual(report["independent_source_clusters_observed"], 1)
            self.assertEqual(report["complete_direction_rows"], 8)
            self.assertEqual(report["feature_columns_total"], 16)
            self.assertFalse(report["all_source_count_minima_met"])
            self.assertFalse(report["real_label_classifier_training_authorized"])
            self.assertFalse(report["receiver_key_isolation_verified"])
            self.assertEqual(sum(report["positive_direction_labels"].values()), 4)
            self.assertNotIn("a" * 64, str(report))
            self.assertNotIn("features16_pre_probe", str(report))

    def test_preflight_mismatch_and_quarantine_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            private, manifest, sha = sample_inputs(base)
            with self.assertRaisesRegex(ValueError, "SHA256 changed"):
                check(private, manifest, "f" * 64)
            blob = json.loads(manifest.read_text())
            blob["candidates"][0]["disposition"] = "QUARANTINE"
            manifest.write_text(json.dumps(blob))
            new_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "absent from eligible"):
                check(private, manifest, new_sha)

    def test_duplicate_source_group_cannot_be_promoted_as_more_episodes(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            private, manifest, sha = sample_inputs(base)
            other = base / "duplicate.jsonl"
            other.write_bytes(private.read_bytes())
            other.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "reused"):
                evaluate_readiness([private, other], manifest,
                                   expected_manifest_sha=sha,
                                   expected_model_sha="b" * 64)

    def test_privacy_and_science_contract_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            private, _, _ = sample_inputs(base)
            baseline = [json.loads(line) for line in private.read_text().splitlines()]
            mutations = [
                (0, "classifier_training_authorized", True),
                (0, "receiver_key_isolated", True),
                (0, "source_clusters_independently_verified", True),
                (0, "direction_index", 5),
                (0, "beneficial_plus_direction", 0),
                (0, "features16_pre_probe", [0.0] * 15),
                (0, "features16_pre_probe", [0.1] * 16),
                (0, "prompt", "private raw text"),
            ]
            for index, name, value in mutations:
                with self.subTest(field=name, value=str(value)[:15]):
                    rows = [dict(row) for row in baseline]
                    rows[index][name] = value
                    if name == "beneficial_plus_direction":
                        rows[index]["beneficial_plus_direction"] = 0
                    altered = base / "altered.jsonl"
                    altered.write_text("".join(json.dumps(x) + "\n" for x in rows))
                    altered.chmod(0o600)
                    with self.assertRaises(ValueError):
                        validate_episode(altered, "b" * 64)
            private.chmod(0o644)
            with self.assertRaises(PermissionError):
                validate_episode(private, "b" * 64)

    def test_extra_or_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            private, _, _ = sample_inputs(base)
            rows = private.read_text().splitlines()
            duplicate = rows[0].replace('"schema":', '"schema": "duplicated", "schema":', 1)
            malformed = base / "duplicate-key.jsonl"
            malformed.write_text("\n".join([duplicate] + rows[1:]) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                validate_episode(malformed, "b" * 64)
            nan = base / "nan.jsonl"
            nan.write_text("\n".join([rows[0].replace("0.1", "NaN", 1)] + rows[1:]))
            with self.assertRaises(ValueError):
                validate_episode(nan, "b" * 64)


if __name__ == "__main__":
    unittest.main()
