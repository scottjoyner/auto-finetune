"""Offline synthetic receipts; no GPU, network or training operation."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from src.lfm3_optimizer_receipt_audit import InvalidReceipt, audit, read


def specimen(opt):
    return {
        "adapter_sha256": ("d" if opt == "adamw" else "e") * 64,
        "adapter_update_norm_sum": 0.08,
        "automatically_promoted": False,
        "base_sha256": "b" * 64,
        "elapsed_seconds": 4.5,
        "holdout_disjoint": True,
        "holdout_loss_after": 0.4,
        "holdout_loss_before": 0.5,
        "model_backbone_weights_modified": False,
        "optimizer": opt, "production_ready": False,
        "real_private_traces_used": False,
        "status": "offline_synthetic_candidate_only",
        "steps": 2,
        "train_loss_after": 0.3,
        "train_loss_before": 0.4,
        "train_steps_loss": [0.38, 0.32],
        "trainable_params": 292864,
    }


class ReceiptAuditTests(unittest.TestCase):
    def test_smoke_pair_categorically_blocked(self):
        result = audit(specimen("adamw"), specimen("muon"))
        self.assertEqual(result["status"], "blocked")
        self.assertFalse(result["weight_promotion_permitted"])
        self.assertFalse(result["independent_holdout_quality_verified"])
        self.assertEqual(result["step_count_by_optimizer"], {"adamw": 2, "muon": 2})

    def test_negative_promotion_claim(self):
        a = specimen("adamw")
        a["automatically_promoted"] = True
        with self.assertRaises(InvalidReceipt):
            audit(a, specimen("muon"))

    def test_unverified_weight_modification(self):
        a = specimen("adamw")
        a["model_backbone_weights_modified"] = True
        with self.assertRaises(InvalidReceipt):
            audit(a, specimen("muon"))

    def test_mismatch_source(self):
        b = specimen("muon")
        b["base_sha256"] = "a" * 64
        with self.assertRaises(InvalidReceipt):
            audit(specimen("adamw"), b)

    def test_same_adapter(self):
        b = specimen("muon")
        b["adapter_sha256"] = "d" * 64
        with self.assertRaises(InvalidReceipt):
            audit(specimen("adamw"), b)

    def test_bad_optimizer_type(self):
        a = specimen("adamw")
        a["optimizer"] = ["adamw"]
        with self.assertRaises(InvalidReceipt):
            audit(a, specimen("muon"))

    def test_bool_step_rejected(self):
        a = specimen("adamw")
        a["steps"] = True
        with self.assertRaises(InvalidReceipt):
            audit(a, specimen("muon"))

    def test_unknown_raw_prompt_field(self):
        a = specimen("adamw")
        a["prompt"] = "SECRET_RAW_PRIVATE"
        with self.assertRaises(InvalidReceipt):
            audit(a, specimen("muon"))

    def test_nonfinite_metric(self):
        a = specimen("adamw")
        a["holdout_loss_after"] = float("nan")
        with self.assertRaises(InvalidReceipt):
            audit(a, specimen("muon"))

    def test_step_series_mismatch(self):
        a = specimen("adamw")
        a["train_steps_loss"].append(0.25)
        with self.assertRaises(InvalidReceipt):
            audit(a, specimen("muon"))

    def test_duplicate_json_keys_fail(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "bad.json"
            p.write_text('{"status":"ok","status":"bad"}')
            with self.assertRaises(InvalidReceipt):
                read(p)

    def test_symlink_file_denied(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.json"
            p.write_text(json.dumps(specimen("adamw")))
            q = Path(t) / "linked"
            q.symlink_to(p)
            with self.assertRaises(OSError):
                read(q)

    def test_hardlink_file_denied(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x.json"
            p.write_text(json.dumps(specimen("adamw")))
            os.link(p, Path(t) / "linked")
            with self.assertRaises(InvalidReceipt):
                read(p)

    def test_cli_fails_closed_without_echoing_paths(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "SECRET-adamw.json"
            q = Path(t) / "SECRET-muon.json"
            p.write_text(json.dumps(specimen("adamw")))
            q.write_text(json.dumps(specimen("muon")))
            result = subprocess.run([sys.executable, "-m",
                "src.lfm3_optimizer_receipt_audit",
                "--adamw", str(p), "--muon", str(q)],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["status"], "blocked")
            self.assertNotIn("SECRET", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
