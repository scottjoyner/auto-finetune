"""Numeric guard tests for the six actual-LFM3 native Muon scaling trials."""
import copy
import json
import unittest
from pathlib import Path

from src.lfm3_muon_scaling_evidence_gate import assess_muon_scaling
from src.lfm3_real_checkpoint_smoke import ResearchRefused


def receipt():
    path = (
        Path(__file__).resolve().parents[1]
        / "research" / "results"
        / "lfm3_real_ffn_muon_original_vs_rms_three_seeds_xwing_20261028.json"
    )
    return json.loads(path.read_text())


class MuonScalingEvidenceTests(unittest.TestCase):
    def test_actual_native_muon_scaling_is_a_near_tie_not_optimizer_approval(self):
        r = assess_muon_scaling(receipt())
        self.assertEqual(r["total_training_trials"], 6)
        self.assertTrue(r["tiny_change_conclusion"])
        self.assertFalse(r["evidence_warrants_new_research_review"])
        self.assertFalse(r["may_change_optimizer"])
        self.assertLess(max(r["absolute_dev_gap_per_seed"]), 0.00001)

    def test_nonfinite_metrics_or_source_authority_cannot_be_forged(self):
        for key in ("production_ready", "source_independently_attested",
                    "training_authority_granted", "heldout_evaluated"):
            x = receipt()
            x[key] = True
            with self.subTest(key=key), self.assertRaises(ResearchRefused):
                assess_muon_scaling(x)
        x = receipt()
        x["arms"][2]["dev_loss_after"] = float("nan")
        with self.assertRaises(ResearchRefused):
            assess_muon_scaling(x)

    def test_seed_alias_optimizer_swap_and_forged_winner_are_refused(self):
        x = receipt()
        x["arms"][4]["seed"] = 20261028
        with self.assertRaises(ResearchRefused):
            assess_muon_scaling(x)
        x = receipt()
        x["arms"][1]["optimizer"] = "adamw"
        with self.assertRaises(ResearchRefused):
            assess_muon_scaling(x)
        x = receipt()
        x["per_seed_original_muon_wins"] = [True, True, True]
        with self.assertRaises(ResearchRefused):
            assess_muon_scaling(x)


if __name__ == "__main__":
    unittest.main()
