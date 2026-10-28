"""Original LFM3 stepwise LR observer can advise but never train."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_schedule_self_observer import (
    prepare_views,prompts,independent_judge,OPTIMIZERS
)
from src.lfm3_optimizer_guard import OptimizerResearchRefused
R=Path(__file__).resolve().parents[1]/"research"/"results"
def fixture():
    schedule=json.loads((R/"lfm3_real_ffn_native_fixed_vs_step1_lr_switch_12trials_xwing_20261028.json").read_text())
    cal=json.loads((R/"lfm3_real_ffn_optimizer_lr_calibration_15_trials_xwing_20261028.json").read_text())
    prior=json.loads((R/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json").read_text())
    return prepare_views(schedule,cal,prior)

class LFM3ScheduleObserverTests(unittest.TestCase):
    def test_read_only_model_telemetry_views_are_bounded(self):
        views,gate=fixture()
        self.assertEqual(len(views),2)
        self.assertEqual([x["optimizer"] for x in views],list(OPTIMIZERS))
        for v in views:
            for layout in (0,1):
                m=prompts(v,layout)
                self.assertEqual(len(m),2)
                self.assertNotIn("source_commit",m[1]["content"])
        self.assertFalse(gate["may_change_fleet_optimizer"])
    def test_consistent_votes_only_propose_nonexecuting_review(self):
        _,gate=fixture()
        votes=[{"optimizer":OPTIMIZERS[i//2],"layout":i%2,"choice":"A",
                "logits":{"A":9.,"B":2.},"matches_external_gate":True}
               for i in range(4)]
        result=independent_judge(gate,votes)
        self.assertFalse(result["model_can_modify_optimizer"])
        self.assertFalse(result["future_synthetic_protocol_review_only"]["execute"])
    def test_forged_independent_numeric_gate_cannot_authorize_model(self):
        _,gate=fixture()
        votes=[{"optimizer":OPTIMIZERS[i//2],"layout":i%2,"choice":"A",
                "logits":{"A":9.,"B":2.},"matches_external_gate":True}
               for i in range(4)]
        for key,val in (("may_promote",True),("all_fixed_lrs_better_than_switches",False),
                        ("authentic_producer_witness",True),("original_backbone_frozen",False)):
            copy=__import__("copy").deepcopy(gate)
            copy[key]=val
            with self.subTest(key=key),self.assertRaises(OptimizerResearchRefused):
                independent_judge(copy,votes)

    def test_wrong_model_choice_vetoed_and_fake_logits_denied(self):
        _,gate=fixture()
        votes=[{"optimizer":OPTIMIZERS[i//2],"layout":i%2,"choice":"B",
                "logits":{"A":1.,"B":2.},"matches_external_gate":False}
               for i in range(4)]
        self.assertIsNone(independent_judge(gate,votes)["future_synthetic_protocol_review_only"])
        votes[0]["matches_external_gate"]=True
        with self.assertRaises(OptimizerResearchRefused):
            independent_judge(gate,votes)
if __name__=="__main__":unittest.main()
