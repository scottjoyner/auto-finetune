"""Frozen LFM3 A/B counterbalance, independent numeric provenance and safety."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_schedule_counterbalance import (
    reverse_label_prompt, counterbalance_judge
)
from src.lfm3_ffn_schedule_self_observer import prepare_views
from src.lfm3_optimizer_guard import OptimizerResearchRefused
R=Path(__file__).resolve().parents[1]/"research"/"results"
def data():
    sch=json.loads((R/"lfm3_real_ffn_native_fixed_vs_step1_lr_switch_12trials_xwing_20261028.json").read_text())
    grid=json.loads((R/"lfm3_real_ffn_optimizer_lr_calibration_15_trials_xwing_20261028.json").read_text())
    old=json.loads((R/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json").read_text())
    original=json.loads((R/"lfm3_real_ffn_stepwise_lr_observer_four_correct_xwing_20261028.json").read_text())
    views,gate=prepare_views(sch,grid,old)
    return views,gate,original
def fake_votes(choices):
    return [{"optimizer":("adamw","muon_match_rms_adamw")[i//2],
             "layout":i%2,"choice":ch,"logits":{"A":6. if ch=="A" else 2.,
                                             "B":6. if ch=="B" else 2.},
             "reversed_numeric_truth":"B",
             "matches_reversed_truth":ch=="B"}
            for i,ch in enumerate(choices)]
class CounterbalancedTests(unittest.TestCase):
    def test_label_flip_changes_truth_but_never_metric(self):
        views,_,_=data()
        for view in views:
            for layout in (0,1):
                prompt=reverse_label_prompt(view,layout)
                self.assertEqual(len(prompt),2)
                self.assertIn("Which schedule",prompt[1]["content"])
                self.assertNotIn("execute",prompt[1]["content"])
    def test_wrong_fixed_A_bias_denied(self):
        _,gate,prior=data()
        verdict=counterbalance_judge(gate,prior,fake_votes("AAAA"))
        self.assertFalse(verdict["reversed_format_all_correct"])
        self.assertEqual(verdict["correct_votes_out_of_eight"],4)
        self.assertFalse(verdict["model_can_select_new_lr"])
    def test_four_correct_reversed_votes_still_never_train(self):
        _,gate,prior=data()
        verdict=counterbalance_judge(gate,prior,fake_votes("BBBB"))
        self.assertTrue(verdict["reversed_format_all_correct"])
        self.assertEqual(verdict["correct_votes_out_of_eight"],8)
        self.assertFalse(verdict["model_can_train"])
        self.assertFalse(verdict["production_ready"])
    def test_forged_loss_witness_and_logits_refused(self):
        _,gate,prior=data()
        votes=fake_votes("BBBB")
        votes[0]["matches_reversed_truth"]=False
        with self.assertRaises(OptimizerResearchRefused):
            counterbalance_judge(gate,prior,votes)
        votes=fake_votes("BBBB")
        prior["optimizer_settings_modified"]=True
        with self.assertRaises(OptimizerResearchRefused):
            counterbalance_judge(gate,prior,votes)
if __name__=="__main__":unittest.main()
