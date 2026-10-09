"""LittleBit Qwen3 byte-accurate admission tests (meta integration opt-in).

Full meta test uses pinned public config, CPU-only Torch and cloned upstream
source, never downloads pretrained weights or runs training.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.littlebit import qwen_meta_admission as admission
from experiments.littlebit.qwen_metadata_budget import (
    projected_upstream_tensor_bytes,
)


class RankAdmissionTests(unittest.TestCase):
    def test_actual_upstream_nominal_055_rank_denied(self):
        x = admission.rank_admission(2048, 1024, 352, 0.55)
        self.assertEqual(x["status"], "DENY")
        self.assertEqual(x["maximum_eligible_rank"], 320)
        self.assertGreater(x["physical_bpw"], 0.55)

    def test_export_bf16_changes_which_modules_are_admissible(self):
        before = admission.rank_admission(2048, 1024, 352, 0.55)
        after = admission.rank_admission(2048, 1024, 352, 0.55,
                                         export_bf16=True)
        self.assertEqual(before["status"], "DENY")
        self.assertEqual(after["status"], "ADMIT")
        self.assertLess(after["projected_bytes"], before["projected_bytes"])
        self.assertEqual(after["storage_case"], "main_export_bf16")
        # Qwen MLP gate path still exceeds 0.55 after BF16 cast.
        self.assertEqual(admission.rank_admission(3072, 1024, 400, 0.55,
                                                 export_bf16=True)["status"], "DENY")

    def test_actual_upstream_nominal_030_rank_denied(self):
        x = admission.rank_admission(2048, 1024, 184, 0.30)
        self.assertEqual(x["status"], "DENY")
        self.assertEqual(x["maximum_eligible_rank"], 160)

    def test_best_safe_rank_is_admitted_without_mutation(self):
        x = admission.rank_admission(2048, 1024, 320, 0.55)
        self.assertEqual(x["status"], "ADMIT")
        self.assertEqual(x["selected_rank"], 320)
        self.assertEqual(x["maximum_eligible_rank"], 320)

    def test_residual_variant_has_different_budget(self):
        a = admission.rank_admission(2048, 1024, 168, 0.55, branches=2)
        b = admission.rank_admission(2048, 1024, 168, 0.55, branches=1)
        self.assertEqual(a["status"], "DENY")
        self.assertEqual(b["status"], "ADMIT")

    def test_padding_changes_transpose_cost(self):
        x = projected_upstream_tensor_bytes(128, 256, 16)
        y = projected_upstream_tensor_bytes(256, 128, 16)
        self.assertNotEqual(x, y)

    def test_reject_nonconforming_rank_or_dtype(self):
        for rank in (0, 7, 9, 2056, True):
            with self.assertRaises(ValueError):
                admission.rank_admission(2048, 1024, rank, 0.55)
        with self.assertRaises(ValueError):
            admission.rank_admission(2048, 1024, 320, 0.55, scale_bytes=8)

    def test_cli_default_refuses_execution(self):
        with self.assertRaises(SystemExit) as caught:
            admission.main(["--config", "/nothing", "--upstream-root", "/nothing"])
        self.assertEqual(caught.exception.code, 2)

    def test_source_pin_refuses_invalid_git_head(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(admission.subprocess, "check_output",
                              return_value="unknown_revision\n"):
                with self.assertRaisesRegex(ValueError, "commit mismatch"):
                    admission.verify_source(Path(root))

    def test_source_pin_refuses_modified_blob(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            for relative in admission.PINNED_BLOBS:
                file = folder / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("untrusted source")
            with patch.object(admission.subprocess, "check_output",
                              return_value=admission.UPSTREAM_COMMIT + "\n"):
                with self.assertRaisesRegex(ValueError, "source blob mismatch"):
                    admission.verify_source(folder)


@unittest.skipUnless(
    os.environ.get("LITTLEBIT_QWEN_CONFIG")
    and os.environ.get("LITTLEBIT_UPSTREAM_ROOT")
    and os.environ.get("LITTLEBIT_RUN_META_TEST") == "1",
    "real CPU Torch+Transformers meta integration is opt-in",
)
class RealQwenMetaTests(unittest.TestCase):
    def test_055_actual_conversion_is_deny_only(self):
        result = admission.inspect(
            Path(os.environ["LITTLEBIT_QWEN_CONFIG"]),
            Path(os.environ["LITTLEBIT_UPSTREAM_ROOT"]), 0.55)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["candidate_modules"], 197)
        self.assertEqual(result["converted_modules"], 196)
        self.assertEqual(result["violations"], 196)
        self.assertEqual(result["export_violations"], 84)
        self.assertAlmostEqual(result["estimated_export_linear_bpw"],
                               0.55194702, places=7)
        self.assertTrue(result["all_model_parameters_meta"])
        self.assertEqual(result["excluded_modules"], ["lm_head"])
        self.assertAlmostEqual(result["estimated_converted_linear_bpw"],
                               0.5797932942708334, places=9)
        self.assertTrue(all(row["status"] == "DENY"
                            for row in result["modules"]))
        self.assertEqual(result["gpu_seconds"], 0)
        self.assertFalse(result["training"])

    def test_030_actual_conversion_is_deny_only(self):
        result = admission.inspect(
            Path(os.environ["LITTLEBIT_QWEN_CONFIG"]),
            Path(os.environ["LITTLEBIT_UPSTREAM_ROOT"]), 0.30)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["violations"], 196)
        self.assertEqual(result["export_violations"], 140)
        self.assertAlmostEqual(result["estimated_converted_linear_bpw"],
                               0.33291829427083336, places=9)

    def test_055_residual_actual_conversion_is_deny_only(self):
        result = admission.inspect(
            Path(os.environ["LITTLEBIT_QWEN_CONFIG"]),
            Path(os.environ["LITTLEBIT_UPSTREAM_ROOT"]), 0.55,
            residual=True)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(result["violations"], 196)
        self.assertEqual(result["export_violations"], 112)
        self.assertAlmostEqual(result["estimated_converted_linear_bpw"],
                               0.6065348307291667, places=9)


if __name__ == "__main__":
    unittest.main()
