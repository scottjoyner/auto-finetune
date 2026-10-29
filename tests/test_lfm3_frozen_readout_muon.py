"""Explicit non-production Muon research exception for frozen LFM3 feature readout."""
import importlib.util
import unittest
from src.lfm3_real_checkpoint_smoke import ResearchRefused
from src.lfm3_frozen_readout_lab import fit_readout_adapter

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch optional in hosted minimal runtime")
class ReadoutMuonTests(unittest.TestCase):
    def fixture(self):
        import torch
        from torch import nn
        generator = torch.Generator(device="cpu").manual_seed(9)
        x0 = torch.randn((2, 2048), generator=generator)
        x1 = torch.randn((2, 2048), generator=generator)
        y0 = torch.tensor((1, 2), dtype=torch.long)
        y1 = torch.tensor((2, 1), dtype=torch.long)
        head = nn.Linear(2048, 16, bias=False).to(dtype=torch.bfloat16)
        head.requires_grad_(False)
        return (x0, x1), (y0, y1), head

    def test_muon_not_eligible_without_manual_lab_exception(self):
        feats, labels, head = self.fixture()
        with self.assertRaises(ResearchRefused):
            fit_readout_adapter(feats, labels, head, lr=0.001, steps=1,
                                optimizer_kind="muon_readout_research")
        with self.assertRaises(ResearchRefused):
            fit_readout_adapter(feats, labels, head, lr=0.001, steps=1,
                                optimizer_kind="unknown", explicit_toy_muon_review=True)

    def test_native_muon_or_explicit_unsupported_runtime_refusal(self):
        import torch
        feats, labels, head = self.fixture()
        version_before = head.weight._version
        if not hasattr(torch.optim, "Muon"):
            with self.assertRaises(ResearchRefused):
                fit_readout_adapter(feats, labels, head, lr=0.001, steps=1,
                                    optimizer_kind="muon_readout_research",
                                    explicit_toy_muon_review=True)
            return
        result = fit_readout_adapter(
            feats, labels, head, lr=0.001, steps=1,
            optimizer_kind="muon_readout_research", explicit_toy_muon_review=True
        )
        self.assertEqual(result["optimizer"], "muon_torch_on_toy_readout_only")
        self.assertEqual(result["trainable_residual_parameters"], 16384)
        self.assertEqual(head.weight._version, version_before)
        self.assertFalse(result["muon_on_actual_lfm3_weights"])
        self.assertFalse(result["weights_saved"])


if __name__ == "__main__":
    unittest.main()
