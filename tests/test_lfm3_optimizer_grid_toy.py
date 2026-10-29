"""Fixed synthetic Muon/AdamW grid contracts, no training on import."""
import unittest
from src.lfm3_optimizer_grid_toy import GRID
from src.lfm3_optimizer_guard import OptimizerResearchRefused
from src.lfm3_optimizer_toy import optimizers_for_toy


class OptimizerGridTests(unittest.TestCase):
    def test_grid_bounded_with_baseline_and_mixed_optimizers(self):
        self.assertEqual(len(GRID), 6)
        self.assertEqual(len(set(GRID)), 6)
        self.assertEqual(GRID[1], ("adamw", 0.001))
        self.assertEqual({o for o, lr in GRID}, {"adamw", "muon_hybrid"})
        self.assertLessEqual(max(lr for _, lr in GRID), 0.002)

    def test_invalid_learning_rate_denied_before_torch_optimizer_instantiation(self):
        class FakeModel:
            def named_parameters(self):
                raise AssertionError("should deny before touching model")
        for value in (0.0, -0.001, 0.1, float("nan"), float("inf"), "0.001"):
            with self.subTest(value=value), self.assertRaises(OptimizerResearchRefused):
                optimizers_for_toy(FakeModel(), "adamw", learning_rate=value)


if __name__ == "__main__":
    unittest.main()
