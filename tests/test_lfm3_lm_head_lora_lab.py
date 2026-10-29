"""True frozen-LM-head LoRA algebra and optimizer policy; no model load."""
import importlib.util
import unittest
from src.lfm3_real_checkpoint_smoke import ResearchRefused
HAS_TORCH = importlib.util.find_spec("torch") is not None

@unittest.skipUnless(HAS_TORCH, "Torch optional in hosted minimal tests")
class FrozenLoRATests(unittest.TestCase):
    def fixture(self):
        import torch
        from torch import nn
        gen = torch.Generator().manual_seed(91)
        features = (torch.randn(2, 2048, generator=gen),
                    torch.randn(2, 2048, generator=gen))
        labels = (torch.tensor([1, 2]), torch.tensor([2, 1]))
        head = nn.Linear(2048, 64, bias=False).to(dtype=torch.bfloat16)
        head.requires_grad_(False)
        return features, labels, head

    def test_adamw_changes_only_lora(self):
        from src.lfm3_lm_head_lora_lab import fit_frozen_lm_head_lora
        f, y, h = self.fixture()
        version = h.weight._version
        receipt = fit_frozen_lm_head_lora(f,y,h,learning_rate=0.001,steps=3)
        self.assertEqual(receipt["target"], "model.lm_head.weight")
        self.assertEqual(receipt["trainable_parameters"], 4*(2048+64))
        self.assertEqual(h.weight._version, version)
        self.assertIsNone(h.weight.grad)
        self.assertFalse(receipt["trained_adapter_written"])
        self.assertTrue(receipt["base_head_version_unchanged"])

    def test_muon_requires_explicit_exception_and_native_optimizer(self):
        from src.lfm3_lm_head_lora_lab import fit_frozen_lm_head_lora
        from torch import optim
        f, y, h = self.fixture()
        with self.assertRaises(ResearchRefused):
            fit_frozen_lm_head_lora(f,y,h,learning_rate=0.001,
                                    optimizer="muon_research",steps=1)
        if not hasattr(optim, "Muon"):
            with self.assertRaises(ResearchRefused):
                fit_frozen_lm_head_lora(f,y,h,learning_rate=0.001,
                    optimizer="muon_research",muon_research_exception=True,steps=1)
        else:
            result = fit_frozen_lm_head_lora(f,y,h,learning_rate=0.001,
                optimizer="muon_research",muon_research_exception=True,steps=1)
            self.assertEqual(result["optimizer"], "muon_experimental")
            self.assertIsNone(h.weight.grad)

    def test_muon_rms_scaling_requires_explicit_review(self):
        from src.lfm3_lm_head_lora_lab import fit_frozen_lm_head_lora
        from torch import optim
        features, labels, head = self.fixture()
        with self.assertRaises(ResearchRefused):
            fit_frozen_lm_head_lora(
                features, labels, head, learning_rate=0.001, steps=1,
                optimizer="adamw", muon_lr_scaling="match_rms_adamw",
            )
        with self.assertRaises(ResearchRefused):
            fit_frozen_lm_head_lora(
                features, labels, head, learning_rate=0.001, steps=1,
                optimizer="muon_research", muon_lr_scaling="match_rms_adamw",
            )
        with self.assertRaises(ResearchRefused):
            fit_frozen_lm_head_lora(
                features, labels, head, learning_rate=0.001, steps=1,
                optimizer="muon_research", muon_research_exception=True,
                muon_lr_scaling="unknown",
            )
        if hasattr(optim, "Muon"):
            result = fit_frozen_lm_head_lora(
                features, labels, head, learning_rate=0.001, steps=1,
                optimizer="muon_research", muon_research_exception=True,
                muon_lr_scaling="match_rms_adamw",
            )
            self.assertEqual(result["optimizer"], "muon_experimental_match_rms_adamw")
            self.assertFalse(result["trained_adapter_written"])

    def test_unreviewed_learning_rates_refused(self):
        from src.lfm3_lm_head_lora_lab import fit_frozen_lm_head_lora
        f,y,h=self.fixture()
        for v in (0.003, 0.0, -1.0, "0.001", float("nan")):
            with self.subTest(value=v), self.assertRaises(ResearchRefused):
                fit_frozen_lm_head_lora(f,y,h,learning_rate=v,steps=1)


if __name__=="__main__":
    unittest.main()
