"""Strict numeric recheck of real LFM3 FFN five-arm LR research."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_optimizer_lr_gate import gate_calibration
from src.lfm3_real_checkpoint_smoke import ResearchRefused
ROOT=Path(__file__).resolve().parents[1]/"research"/"results"
def fixture():
    measured=json.loads((ROOT/"lfm3_real_ffn_optimizer_lr_calibration_15_trials_xwing_20261028.json").read_text())
    original=json.loads((ROOT/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json").read_text())
    return measured,original

class LRCalibrationGateTests(unittest.TestCase):
    def test_real_calibration_nominates_nonexecuting_research_only(self):
        verdict=gate_calibration(*fixture())
        self.assertEqual(verdict["best_option_index"],2)
        self.assertEqual(verdict["best_option_each_seed"],[2,2,2])
        self.assertTrue(verdict["development_only_research_review_eligible"])
        self.assertFalse(verdict["may_adjust_live_optimizer"])
        self.assertFalse(verdict["heldout_evaluated"])
    def test_bad_gradient_lr_seed_or_reported_metric_fails_closed(self):
        for key,val in (("learning_rate",0.02),("seed",20261029),
                        ("dev_loss_after",float("nan")),
                        ("full_model_gradient_verified",False)):
            report,original=fixture()
            report["trials"][0][key]=val
            with self.subTest(key=key),self.assertRaises(ResearchRefused):
                gate_calibration(report,original)
    def test_untrusted_model_or_source_cannot_promote(self):
        for k in ("model_advice_used_for_training","source_authenticated",
                  "holdout_evaluated","learning_rate_promoted","may_auto_execute"):
            report,original=fixture()
            report[k]=True
            with self.subTest(key=k),self.assertRaises(ResearchRefused):
                gate_calibration(report,original)
    def test_original_development_receipt_locked(self):
        report,original=fixture()
        original["arms"][0]["dev_loss_after"]=10.0
        with self.assertRaises(ResearchRefused):
            gate_calibration(report,original)

if __name__=="__main__":unittest.main()
