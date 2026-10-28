"""Actual frozen LFM3 fixed-vs-switch trials fail closed on forged scores."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_fixed_lr_schedule_gate import review_fixed_lr_schedules
from src.lfm3_real_checkpoint_smoke import ResearchRefused
R=Path(__file__).resolve().parents[1]/"research"/"results"
def load():
    data=json.loads((R/"lfm3_real_ffn_native_fixed_vs_step1_lr_switch_12trials_xwing_20261028.json").read_text())
    grid=json.loads((R/"lfm3_real_ffn_optimizer_lr_calibration_15_trials_xwing_20261028.json").read_text())
    baseline=json.loads((R/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json").read_text())
    return data,grid,baseline

class NativeScheduleReceiptTests(unittest.TestCase):
    def test_actual_fixed_lr_beats_step_one_switch_both_optimizers(self):
        review=review_fixed_lr_schedules(*load())
        self.assertTrue(review["all_fixed_lrs_better_than_switches"])
        self.assertTrue(review["learning_rate_changed_during_bounded_training"])
        self.assertFalse(review["may_execute_optimizer_proposal"])
        self.assertFalse(review["may_change_fleet_optimizer"])
        self.assertFalse(review["production_ready"])
    def test_forged_gradient_or_lr_schedule_rejected(self):
        for key,val in (("learning_rate",0.2),("optimizer_changed_by_model",True),
                        ("dev_loss_after",float("nan")),("step",3)):
            data,grid,baseline=load()
            data["trials"][1]["observations"][0][key]=val
            with self.subTest(key=key),self.assertRaises(ResearchRefused):
                review_fixed_lr_schedules(data,grid,baseline)
    def test_fake_source_permission_and_baseline_refused(self):
        for key in ("automatic_optimizer_switching_authorized",
                    "model_recommendation_executed","authentic_producer_witness",
                    "new_training_authorized","heldout_evaluated"):
            data,grid,baseline=load()
            data[key]=True
            with self.subTest(key=key),self.assertRaises(ResearchRefused):
                review_fixed_lr_schedules(data,grid,baseline)
        data,grid,baseline=load()
        data["trials"][0]["final_dev_loss"]=0.1
        with self.assertRaises(ResearchRefused):
            review_fixed_lr_schedules(data,grid,baseline)
        data,grid,baseline=load()
        data["trials"][4]["initial_dev_loss"]=44.
        with self.assertRaises(ResearchRefused):
            review_fixed_lr_schedules(data,grid,baseline)

if __name__=="__main__":unittest.main()
