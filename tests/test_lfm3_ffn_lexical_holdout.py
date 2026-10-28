"""No model download: verify precommitted dev-only selection and frozen split."""
import copy
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_lexical_holdout import HOLDOUT, validate_protocol, fixed_selection
from src.lfm3_real_checkpoint_smoke import ResearchRefused


def data():
    p = Path(__file__).resolve().parents[1] / "research" / "results" / "lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json"
    return json.loads(p.read_text())


class FFNHoldoutProtocolTests(unittest.TestCase):
    def test_precommitted_dev_selector_has_no_heldout_input(self):
        result = fixed_selection(data())
        self.assertEqual(result["optimizer"], "adamw")
        self.assertEqual(result["three_training_steps"], 3)
        self.assertFalse(result["heldout_influenced_selection"])
        self.assertFalse(result["may_execute_outside_scratch"])

    def test_holdout_source_is_frozen_and_distinct(self):
        self.assertTrue(validate_protocol())
        with self.assertRaises(ResearchRefused):
            validate_protocol(HOLDOUT[:3] + (HOLDOUT[1],))

    def test_invalid_dev_scores_or_optimizer_identity_refused(self):
        x = data()
        x["arms"][2]["dev_loss_after"] = float("nan")
        with self.assertRaises(ResearchRefused):
            fixed_selection(x)
        x = data()
        x["arms"][0]["optimizer"] = "muon_original"
        with self.assertRaises(ResearchRefused):
            fixed_selection(x)
        x = data()
        x["source_generation_verified"] = True
        with self.assertRaises(ResearchRefused):
            fixed_selection(x)


if __name__ == "__main__":
    unittest.main()
