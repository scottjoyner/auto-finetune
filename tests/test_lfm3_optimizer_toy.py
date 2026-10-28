"""Real PyTorch optimizer grouping and toy-run checks, no LFM3 checkpoint."""
import importlib.util
import unittest

from src.lfm3_optimizer_guard import OptimizerResearchRefused

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch optional in lightweight CI")
class OptimizerToyTests(unittest.TestCase):
    def test_muon_only_reviewed_hidden_2d_and_adamw_rest(self):
        import torch
        from torch import nn
        from src.lfm3_optimizer_toy import classify_for_muon
        model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        named = list(model.named_parameters())
        chosen, rest = classify_for_muon(named, {"0.weight"})
        self.assertEqual(len(chosen), 1)
        self.assertEqual(len(rest), 3)
        self.assertEqual(chosen[0].ndim, 2)

    def test_disallow_unsafe_embeddings_norms_and_low_rank_adapters(self):
        import torch
        from torch import nn
        from src.lfm3_optimizer_toy import classify_for_muon
        for forbidden in ("model.embed_tokens.weight", "model.lm_head.weight",
                          "layers.0.conv.weight", "layers.0.lora_A.weight",
                          "layers.0.gate_proj.weight", "layer_norm.weight"):
            with self.subTest(forbidden=forbidden), self.assertRaises(OptimizerResearchRefused):
                classify_for_muon([(forbidden, nn.Parameter(torch.ones(4, 4)))],
                                  {forbidden})
        with self.assertRaises(OptimizerResearchRefused):
            classify_for_muon([("hidden.weight", nn.Parameter(torch.ones(1, 9)))],
                              {"hidden.weight"})

    def test_duplicate_alias_and_frozen_weights_denied(self):
        import torch
        from torch import nn
        from src.lfm3_optimizer_toy import classify_for_muon
        p = nn.Parameter(torch.ones(5, 6))
        with self.assertRaises(OptimizerResearchRefused):
            classify_for_muon([("hidden.weight", p), ("other.weight", p)],
                              {"hidden.weight"})
        p.requires_grad_(False)
        with self.assertRaises(OptimizerResearchRefused):
            classify_for_muon([("hidden.weight", p)], {"hidden.weight"})

    def test_both_native_optimizers_really_step_and_do_not_save_weights(self):
        from src.lfm3_optimizer_toy import run_toy_arm
        adamw = run_toy_arm("adamw", steps=3)
        muon = run_toy_arm("muon_hybrid", steps=3)
        self.assertEqual(adamw["nonfinite_updates"], 0)
        self.assertEqual(muon["nonfinite_updates"], 0)
        self.assertLess(adamw["train_loss_end"], adamw["train_loss_start"])
        self.assertLess(muon["train_loss_end"], muon["train_loss_start"])
        self.assertFalse(muon["actual_lfm3_weights_trained"])
        self.assertFalse(muon["model_weights_written"])
        self.assertFalse(adamw["model_weights_written"])


if __name__ == "__main__":
    unittest.main()
