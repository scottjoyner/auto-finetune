"""Fixed readout Muon/AdamW observer, no pretrained model import."""
import copy
import json
import unittest
from pathlib import Path
from src.lfm3_readout_grid_observer import (
    readout_grid_view, messages, OptimizerResearchRefused
)


def fixture():
    p = (Path(__file__).resolve().parents[1] / "research" / "results"
         / "lfm3_real_frozen_readout_muon_adamw_xwing_20261028.json")
    return json.loads(p.read_text())


class FrozenMuonObserverTests(unittest.TestCase):
    def test_strict_grid_and_observed_best_is_adamw_002(self):
        view, best = readout_grid_view(fixture())
        self.assertEqual(best, "B")
        self.assertEqual(len(view["options"]), 4)
        self.assertEqual(view["options"][2]["optimizer"], "muon_torch_on_toy_readout_only")
        prompt = messages(view)[-1]["content"]
        self.assertNotIn("synthetic_train_sha256", prompt)
        self.assertNotIn("synthetic_dev_sha256", prompt)
        self.assertNotIn("heldout", prompt)

    def test_forged_muon_promotion_and_mismatched_receipts_rejected(self):
        for k, v in (
            ("production_ready", True),
            ("no_muon_adapter_optimization_executed", True),
            ("actual_full_lfm3_finetuning", True),
        ):
            x = copy.deepcopy(fixture())
            x[k] = v
            with self.subTest(k=k), self.assertRaises(OptimizerResearchRefused):
                readout_grid_view(x)
        x = copy.deepcopy(fixture())
        x["arms"][2]["muon_on_actual_lfm3_weights"] = True
        with self.assertRaises(OptimizerResearchRefused):
            readout_grid_view(x)
        x = copy.deepcopy(fixture())
        x["arms"][3]["dev_loss_before"] = 100.0
        with self.assertRaises(OptimizerResearchRefused):
            readout_grid_view(x)
        x = copy.deepcopy(fixture())
        x["train_count"] = 100
        with self.assertRaises(OptimizerResearchRefused):
            readout_grid_view(x)


if __name__ == "__main__":
    unittest.main()
