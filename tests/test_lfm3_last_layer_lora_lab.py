"""Actual layer-LoRA autograd works while base torch.Linear stays unchanged."""
import importlib.util
import unittest
from src.lfm3_real_checkpoint_smoke import ResearchRefused
HAS_TORCH = importlib.util.find_spec("torch") is not None

@unittest.skipUnless(HAS_TORCH, "Torch optional in hosted minimal environment")
class LayerLoRAGraphTests(unittest.TestCase):
    def test_frozen_layer_receives_real_gradient_only_through_adapter(self):
        import torch
        from torch import nn
        from src.lfm3_last_layer_lora_lab import install_adapter
        layer = nn.Linear(32, 32, bias=False)
        layer.requires_grad_(False)
        w_before = layer.weight.detach().clone()
        version = layer.weight._version
        A, B, handle = install_adapter(layer)
        try:
            x = torch.randn(2, 32)
            y = layer(x)
            self.assertTrue(y.requires_grad)
            y.square().mean().backward()
            self.assertIsNone(layer.weight.grad)
            self.assertIsNotNone(B.weight.grad)
            self.assertGreater(B.weight.grad.abs().sum().item(), 0)
            self.assertTrue(torch.equal(layer.weight.detach(), w_before))
            self.assertEqual(layer.weight._version, version)
        finally:
            handle.remove()

    def test_full_layer_training_budget_is_fixed(self):
        from src.lfm3_last_layer_lora_lab import run
        for steps in (0, 2, 4, 10, 1.0, "3"):
            with self.subTest(steps=steps), self.assertRaises(ResearchRefused):
                run(None, steps=steps)

    def test_unreviewed_or_trainable_layer_refused(self):
        from torch import nn
        from src.lfm3_last_layer_lora_lab import install_adapter
        layer = nn.Linear(32, 32)
        with self.assertRaises(ResearchRefused):install_adapter(layer)
        layer.requires_grad_(False)
        with self.assertRaises(ResearchRefused):install_adapter(layer, rank=8)
        with self.assertRaises(ResearchRefused):install_adapter(nn.ReLU())

if __name__ == "__main__":
    unittest.main()
