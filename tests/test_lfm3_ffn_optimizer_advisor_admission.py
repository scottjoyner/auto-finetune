"""An apparent 4/4 model advisor collapses under actual label reversal."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_optimizer_advisor_admission import evaluate_optimizer_advisor
from src.lfm3_optimizer_guard import OptimizerResearchRefused
R=Path(__file__).resolve().parents[1]/"research"/"results"
def evidence():
    def read(x):return json.loads((R/x).read_text())
    return (
        read("lfm3_real_ffn_optimizer_lr_calibration_15_trials_xwing_20261028.json"),
        read("lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json"),
        read("lfm3_real_ffn_native_fixed_vs_step1_lr_switch_12trials_xwing_20261028.json"),
        read("lfm3_real_ffn_stepwise_lr_observer_four_correct_xwing_20261028.json"),
        read("lfm3_real_ffn_label_reversal_four_failures_xwing_20261028.json"),
    )
class OptimizerAdvisorAdmissionTests(unittest.TestCase):
    def test_real_model_correct_only_when_best_was_A(self):
        v=evaluate_optimizer_advisor(*evidence())
        self.assertEqual(v["paired_observation_accuracy"],.5)
        self.assertTrue(v["label_position_bias_detected"])
        self.assertIsNone(v["reviewable_hypothesis"])
        self.assertFalse(v["model_may_change_optimizer"])
        self.assertFalse(v["model_may_execute"])
    def test_fake_production_status_and_relabeling_refused(self):
        for k in ("production_ready","optimizer_modified",
                  "model_can_train","independently_authenticated"):
            d=list(evidence())
            d[4][k]=True
            with self.subTest(field=k),self.assertRaises(OptimizerResearchRefused):
                evaluate_optimizer_advisor(*d)
        d=list(evidence())
        d[4]["reversed_model_votes"][0]["matches_reversed_truth"]=True
        with self.assertRaises(OptimizerResearchRefused):
            evaluate_optimizer_advisor(*d)
    def test_forged_numeric_source_denied(self):
        d=list(evidence())
        d[2]["trials"][0]["observations"][0]["dev_loss_after"]=float("nan")
        with self.assertRaises(Exception):
            evaluate_optimizer_advisor(*d)
if __name__=="__main__":unittest.main()
