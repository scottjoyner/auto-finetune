"""Fix two self-observer layouts and forbid logit/permission forgery."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_holdout_observer import view_and_gate, messages, judge
from src.lfm3_optimizer_guard import OptimizerResearchRefused

R=Path(__file__).resolve().parents[1]/"research"/"results"
def data():
    h=json.loads((R/"lfm3_real_ffn_dev_selected_lexical_holdout_3seed_xwing_20261028.json").read_text())
    d=json.loads((R/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json").read_text())
    return view_and_gate(h,d)

class HoldoutObserverTests(unittest.TestCase):
    def test_two_formats_are_readonly_and_numeric(self):
        view,gate=data()
        self.assertTrue(gate["eligible_for_new_synthetic_review"])
        for layout in (0,1):
            payload=messages(view,layout)
            self.assertIn("holdout",payload[-1]["content"])
            self.assertNotIn("checkpoint_sha256",payload[-1]["content"])
        self.assertFalse(view["may_execute"])
    def test_correct_scores_only_nominate_nonexecuting_review(self):
        _,gate=data()
        votes=[{"layout":i,"choice":"A","scores":{"A":9.,"B":2.},
                "matches_numeric_gate":True} for i in (0,1)]
        result=judge(gate,votes)
        self.assertTrue(result["all_votes_correct_and_agree"])
        self.assertFalse(result["new_synthetic_trial_review_only"]["execute"])
        self.assertFalse(result["can_modify_optimizer"])
    def test_wrong_vote_veto_and_forged_logits_fail(self):
        _,gate=data()
        votes=[{"layout":i,"choice":"A","scores":{"A":9.,"B":2.},
                "matches_numeric_gate":True} for i in (0,1)]
        votes[1]={"layout":1,"choice":"B","scores":{"A":2.,"B":9.},
                  "matches_numeric_gate":False}
        self.assertIsNone(judge(gate,votes)["new_synthetic_trial_review_only"])
        votes[1]["matches_numeric_gate"]=True
        with self.assertRaises(OptimizerResearchRefused):judge(gate,votes)

if __name__=="__main__":unittest.main()
