"""Offline contract tests: synthetic cases only, no checkpoint or network."""
import hashlib
import unittest
from pathlib import Path

from src.lfm3_ab_protocol import (EvaluationRefused, evaluate, validate_manifest,
                              REVISION, CONFIG_SHA256,
                              validate_results, _load, sha_text)


def sha(x):
    return hashlib.sha256(x.encode()).hexdigest()


def fixture():
    manifest = {
        "schema": "sg-lfm3-ab.v1",
        "model_id": "LiquidAI/LFM3-1.7B",
        "revision": REVISION,
        "checkpoint_sha256": "b" * 64,
        "config_sha256": CONFIG_SHA256,
        "tokenizer_sha256": "d" * 64,
        "sampling": {"seed": 17, "temperature": 0, "max_new_tokens": 64},
        "conditions": ["baseline", "retrieval", "gates", "retrieval_gates"],
        "train_session_hashes": [sha("training-session")],
        "cases": [
            {"id": "example-one", "session_hash": sha("heldout-1"),
             "split": "heldout", "kind": "tool_choice",
             "expected_sha256": sha_text("tool-1")},
            {"id": "example-two", "session_hash": sha("heldout-2"),
             "split": "heldout", "kind": "abstain",
             "expected_sha256": sha_text("ABSTAIN")},
        ],
    }
    rows = []
    proofs = {
        "baseline": (None, None, None),
        "retrieval": ("e" * 64, None, None),
        "gates": (None, "f" * 64, None),
        "retrieval_gates": ("e" * 64, "f" * 64, None),
    }
    answers = {
        "baseline": ("tool-1", "wrong"),
        "retrieval": ("tool-1", "ABSTAIN"),
        "gates": ("tool-1", "wrong"),
        "retrieval_gates": ("tool-1", "ABSTAIN"),
    }
    for mode in manifest["conditions"]:
        for idx, case in enumerate(manifest["cases"]):
            memory, gate, adapter = proofs[mode]
            rows.append({
                "case_id": case["id"], "condition": mode, "answer": answers[mode][idx],
                "seed": 17, "checkpoint_sha256": "b" * 64,
                "wall_ms": 10 + idx, "prefill_tokens": 20, "decode_tokens": 4,
                "node_id": "xwing", "memory_sha256": memory,
                "gate_sha256": gate, "adapter_sha256": adapter,
            })
    return manifest, rows


class ABProtocolTest(unittest.TestCase):
    def test_paired_gain_is_scored_without_raw_answers(self):
        manifest, rows = fixture()
        receipt = evaluate(manifest, rows)
        grades = {r["condition"]: r for r in receipt["metrics"]}
        self.assertEqual(grades["baseline"]["exact_correct"], 1)
        self.assertEqual(grades["retrieval"]["exact_correct"], 2)
        self.assertEqual(grades["retrieval"]["paired_wins_vs_baseline"], 1)
        self.assertFalse(receipt["checkpoint_cryptographically_attested"])
        self.assertNotIn("tool-1", str(receipt))

    def test_leakage_and_cross_split_session_refused(self):
        manifest, rows = fixture()
        manifest["train_session_hashes"].append(manifest["cases"][0]["session_hash"])
        with self.assertRaises(EvaluationRefused):
            validate_manifest(manifest)
        manifest, rows = fixture()
        manifest["cases"][1]["session_hash"] = manifest["cases"][0]["session_hash"]
        manifest["cases"][1]["split"] = "dev"
        with self.assertRaises(EvaluationRefused):
            validate_manifest(manifest)

    def test_incomplete_duplicate_or_cross_node_matrix_refused(self):
        manifest, rows = fixture()
        with self.assertRaises(EvaluationRefused):
            validate_results(manifest, rows[:-1])
        manifest, rows = fixture()
        rows[1] = dict(rows[0])
        with self.assertRaises(EvaluationRefused):
            validate_results(manifest, rows)
        manifest, rows = fixture()
        rows[3]["node_id"] = "different-node"
        with self.assertRaises(EvaluationRefused):
            validate_results(manifest, rows)

    def test_condition_proof_pin_and_nonfinite_refused(self):
        manifest, rows = fixture()
        rows[0]["gate_sha256"] = "f" * 64
        with self.assertRaises(EvaluationRefused):
            evaluate(manifest, rows)
        manifest, rows = fixture()
        rows[2]["checkpoint_sha256"] = "a" * 64
        with self.assertRaises(EvaluationRefused):
            evaluate(manifest, rows)
        manifest, rows = fixture()
        rows[0]["wall_ms"] = float("nan")
        with self.assertRaises(EvaluationRefused):
            evaluate(manifest, rows)

    def test_model_revision_and_config_require_exact_pins(self):
        manifest, _ = fixture()
        manifest["revision"] = "a" * 40
        with self.assertRaises(EvaluationRefused):
            validate_manifest(manifest)
        manifest, _ = fixture()
        manifest["config_sha256"] = "a" * 64
        with self.assertRaises(EvaluationRefused):
            validate_manifest(manifest)

    def test_unknown_fields_and_duplicate_json_refused(self):
        manifest, rows = fixture()
        manifest["raw_trace"] = "do not accept"
        with self.assertRaises(EvaluationRefused):
            validate_manifest(manifest)
        with self.assertRaises(EvaluationRefused):
            _load(b'{"id":"one","id":"two"}', 4096)
        with self.assertRaises(EvaluationRefused):
            _load(b'{"score":NaN}', 4096)


if __name__ == "__main__":
    unittest.main()
