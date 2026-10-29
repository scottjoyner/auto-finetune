"""CPU-only synthetic contract tests for the independent-review classifier."""
import json
import tempfile
import unittest
from pathlib import Path
from src.lfm3_paired_holdout import InvalidEvidence, assess, load


def fixture(gain=0.04):
    return {
        "schema": "sensorgraft.lfm3.paired-holdout.v1",
        "base_sha256": "a"*64, "dataset_sha256": "b"*64,
        "eval_sha256": "c"*64, "benchmark_sha256": "d"*64,
        "baseline_optimizer": "adamw", "candidate_optimizer": "hybrid",
        "baseline": [{"seed": n, "checkpoint_sha256": f"{n+1:064x}",
                      "quality": 0.6, "p95_latency_ms": 100, "peak_memory_bytes": 1_000_000}
                     for n in range(3)],
        "candidate": [{"seed": n, "checkpoint_sha256": f"{n+10:064x}",
                       "quality": 0.6 + gain, "p95_latency_ms": 110, "peak_memory_bytes": 1_050_000}
                      for n in range(3)],
    }


class PairedHoldoutTests(unittest.TestCase):
    def assert_rejected(self, f):
        with self.assertRaises(InvalidEvidence):
            assess(f)

    def test_good_evidence_does_not_promote(self):
        result = assess(fixture())
        self.assertEqual(result["status"], "candidate_for_independent_review")
        self.assertFalse(result["weight_promotion_permitted"])
        self.assertFalse(result["evaluation_independence_verified"])

    def test_not_enough_quality_gain(self):
        self.assertEqual(assess(fixture(0.01))["status"], "hold")

    def test_one_seed_regresses(self):
        f = fixture()
        f["candidate"][-1]["quality"] = 0.58
        self.assertEqual(assess(f)["status"], "hold")

    def test_latency_regression(self):
        f = fixture()
        f["candidate"][0]["p95_latency_ms"] = 150
        self.assertEqual(assess(f)["status"], "hold")

    def test_memory_regression(self):
        f = fixture()
        f["candidate"][0]["peak_memory_bytes"] = 1_500_000
        self.assertEqual(assess(f)["status"], "hold")

    def test_unpaired_seeds(self):
        f = fixture()
        f["candidate"][0]["seed"] = 5
        self.assert_rejected(f)

    def test_duplicate_seed(self):
        f = fixture()
        f["candidate"][0]["seed"] = 1
        self.assert_rejected(f)

    def test_too_few_runs(self):
        f = fixture()
        f["candidate"] = f["candidate"][:2]
        self.assert_rejected(f)

    def test_same_checkpoints(self):
        f = fixture()
        f["candidate"][0]["checkpoint_sha256"] = f["baseline"][0]["checkpoint_sha256"]
        self.assert_rejected(f)

    def test_private_text_is_rejected(self):
        f = fixture()
        f["candidate"][0]["prompt"] = "secret"
        self.assert_rejected(f)

    def test_bools_nan_and_large_number(self):
        for k, v in (("quality", float("nan")), ("quality", True), ("peak_memory_bytes", True),
                     ("p95_latency_ms", float("inf")), ("quality", 1e30)):
            f = fixture()
            f["candidate"][0][k] = v
            self.assert_rejected(f)

    def test_provenance_must_be_distinct(self):
        f = fixture()
        f["eval_sha256"] = f["dataset_sha256"]
        self.assert_rejected(f)

    def test_strict_json_no_duplicates(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "m.json"
            p.write_text('{"x":1,"x":2}')
            with self.assertRaises(InvalidEvidence):
                load(p)

    def test_file_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            target = Path(t) / "m.json"
            target.write_text(json.dumps(fixture()))
            linked = Path(t) / "linked.json"
            linked.symlink_to(target)
            with self.assertRaises(OSError):
                load(linked)

    def test_file_size_cap(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "big.json"
            p.write_text(" " * 65537)
            with self.assertRaises(InvalidEvidence):
                load(p)


if __name__ == "__main__":
    unittest.main()
