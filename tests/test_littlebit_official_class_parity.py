"""Minimal fail-closed tests; official runtime checks are opt-in with a local source clone.

Run on a CPU-only torch venv:
  LITTLEBIT_UPSTREAM_ROOT=/path/to/pinned/LittleBit \
    python -m unittest discover -s tests -p test_littlebit_official_class_parity.py -v

No real model weights, dataset downloads or network calls.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.littlebit import official_class_parity as parity


class SourceBoundaryTests(unittest.TestCase):
    def test_blob_hash_known_value(self):
        # git hash-object of an empty blob
        self.assertEqual(
            parity.git_blob_sha(b""),
            "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
        )

    def test_cli_requires_explicit_opt_in(self):
        with self.assertRaises(SystemExit) as error:
            parity.main(["--upstream-root", "/nonexistent/path"])
        self.assertEqual(error.exception.code, 2)

    def test_missing_and_tampered_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in parity.BLOBS:
                f = root / path
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("untrusted source", encoding="utf-8")
            with patch.object(parity.subprocess, "check_output",
                              return_value=parity.UPSTREAM_COMMIT + "\n"):
                with self.assertRaisesRegex(ValueError, "source blob mismatch"):
                    parity.verify_source(root)

    def test_head_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(parity.subprocess, "check_output",
                              return_value="not-the-pinned-revision\n"):
                with self.assertRaisesRegex(ValueError, "exact pinned revision"):
                    parity.verify_source(Path(tmp))

    def test_seeds_and_shapes_are_bounded_before_untrusted_load(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Torch not available for optional CPU guard")
        import torch
        with self.assertRaisesRegex(ValueError, "seeds exceed"):
            parity.run(Path("/missing"), seeds=(999,))
        with self.assertRaisesRegex(ValueError, "shapes exceed"):
            parity.run(Path("/missing"), shapes=((512, 512, 512),))
        with self.assertRaisesRegex(ValueError, "ITQ iterations"):
            parity.run(Path("/missing"), itq_iters=51)
        with patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "CPU-only build"):
                parity.run(Path("/missing"))


@unittest.skipUnless(
    os.environ.get("LITTLEBIT_UPSTREAM_ROOT") and
    importlib.util.find_spec("torch") is not None,
    "opt-in pinned checkout and CPU-only Torch needed",
)
class RealPinnedUpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ["LITTLEBIT_UPSTREAM_ROOT"])

    def test_real_source_hashes(self):
        result = parity.verify_source(self.root)
        self.assertEqual(set(result), set(parity.BLOBS))

    def test_primary_official_class_paired_and_fp32(self):
        result = parity.run(
            self.root, seeds=(7,), shapes=((64, 64, 8),), residual=False,
            itq_iters=5)
        self.assertEqual(result["paired_count"], 1)
        self.assertEqual(len(result["cases"]), 2)
        self.assertEqual(result["cases"][0]["original_weight_sha256"],
                         result["cases"][1]["original_weight_sha256"])
        for case in result["cases"]:
            self.assertLess(case["forward_dense_max_abs_error"], 1e-3)
            self.assertGreater(case["measured_tensor_bpw"],
                               case["reported_upstream_bpw"])
            self.assertGreater(case["measured_single_layer_zip_bpw"],
                               case["measured_tensor_bpw"])
            self.assertEqual(set(case["scale_dtypes"].values()),
                             {"torch.float32"})
        self.assertFalse(result["training"])
        self.assertFalse(result["production_dispatch"])

    def test_residual_official_class_packed_shapes_and_bytes(self):
        result = parity.run(
            self.root, seeds=(7,), shapes=((64, 64, 8),), residual=True,
            itq_iters=5)
        for case in result["cases"]:
            self.assertEqual(len(case["packed_tensor_shapes"]), 4)
            self.assertEqual(len(case["scale_dtypes"]), 8)
            self.assertTrue(all(dtype == "torch.float32"
                                for dtype in case["scale_dtypes"].values()))
            self.assertGreater(case["tensor_bytes"], case["scale_bytes"])
            self.assertLess(case["forward_dense_max_abs_error"], 1e-3)

    def test_repeat_is_deterministic_at_fixed_seed(self):
        a = parity.run(
            self.root, seeds=(42,), shapes=((64, 64, 8),), itq_iters=5)
        b = parity.run(
            self.root, seeds=(42,), shapes=((64, 64, 8),), itq_iters=5)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
