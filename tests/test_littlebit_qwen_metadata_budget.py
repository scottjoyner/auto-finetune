"""CPU/stdlib metadata-only feasibility tests; no model weights or Torch."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from experiments.littlebit import qwen_metadata_budget as budget


def fixture():
    return {
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "hidden_size": 1024, "intermediate_size": 3072,
        "num_attention_heads": 16, "num_key_value_heads": 8,
        "head_dim": 128, "num_hidden_layers": 28,
        "vocab_size": 151936, "tie_word_embeddings": True,
    }


class BudgetArithmeticTests(unittest.TestCase):
    def test_row_padding_and_four_fp32_scales(self):
        self.assertEqual(
            budget.projected_upstream_tensor_bytes(64, 64, 8), 944)
        self.assertEqual(
            budget.projected_upstream_tensor_bytes(128, 256, 16), 2736)
        self.assertEqual(
            budget.projected_upstream_tensor_bytes(256, 128, 16), 2992)

    def test_residual_doubles_branches_not_buffers(self):
        one = budget.projected_upstream_tensor_bytes(64, 64, 8)
        two = budget.projected_upstream_tensor_bytes(
            64, 64, 8, branches=2)
        self.assertEqual(two, 2 * one - 16)

    def test_infeasible_small_layer(self):
        self.assertEqual(
            budget.pick_rank(64, 64, target_bpw=.55)["status"], "INFEASIBLE")

    def test_invalid_dimensions_and_rank(self):
        for args in ((0, 64, 8), (64, 0, 8), (64, 64, 0), (64, 64, 80)):
            with self.assertRaises(ValueError):
                budget.projected_upstream_tensor_bytes(*args)
        with self.assertRaises(ValueError):
            budget.pick_rank(64, 64, target_bpw=1.5)

    def test_qwen3_expected_mapped_shapes(self):
        shape = budget.qwen3_linear_shapes(fixture())
        self.assertEqual(len(shape), 7)
        self.assertEqual(shape["self_attn.q_proj"], (2048, 1024))
        self.assertEqual(shape["self_attn.k_proj"], (1024, 1024))
        self.assertEqual(shape["self_attn.o_proj"], (1024, 2048))
        self.assertEqual(shape["mlp.down_proj"], (1024, 3072))

    def test_whole_model_tied_embedding_is_explicit(self):
        summary = budget.inspect_pinned_metadata(fixture())
        self.assertEqual(summary["original_linear_parameters"], 440401920)
        self.assertEqual(summary["tied_embedding_parameters"], 155582464)
        self.assertEqual(summary["selected_parameter_denominator"], 595984384)
        self.assertTrue(summary["no_pretrained_weights_loaded"])
        self.assertEqual(summary["gpu_seconds"], 0)
        self.assertEqual(summary["provider_calls"], 0)
        a, b = summary["scenarios"]
        self.assertEqual(a["projected_total_tensor_bytes_excluding_other_layers"],
                         340980672)
        self.assertEqual(b["projected_total_tensor_bytes_excluding_other_layers"],
                         327305920)
        self.assertLess(a["projected_block_bpw"], .55)
        self.assertLess(b["projected_block_bpw"], .30)
        self.assertGreater(a["projected_model_bpw_excluding_other_layers"], 4)

    def test_pinned_config_rejects_unrelated_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            p.write_text(json.dumps(fixture()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config digest changed"):
                budget.read_pinned_config(p)

    def test_cli_disarmed_without_opt_in(self):
        with self.assertRaises(SystemExit) as err:
            budget.main(["--config", "/does/not/matter.json"])
        self.assertEqual(err.exception.code, 2)


@unittest.skipUnless(os.environ.get("LITTLEBIT_QWEN_CONFIG"),
                     "opt-in pinned 726-byte Qwen config not supplied")
class PinnedConfigTests(unittest.TestCase):
    def test_actual_pinned_config(self):
        path = Path(os.environ["LITTLEBIT_QWEN_CONFIG"])
        cfg = budget.read_pinned_config(path)
        self.assertEqual(cfg["num_hidden_layers"], 28)
        self.assertEqual(cfg["vocab_size"], 151936)
        self.assertTrue(cfg["tie_word_embeddings"])


if __name__ == "__main__":
    unittest.main()
