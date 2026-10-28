"""Two read-only LFM3 observer layouts cannot grant trial authority."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_optimizer_lr_observer import read_only_view,prompt,judge,LABELS
from src.lfm3_optimizer_guard import OptimizerResearchRefused
ROOT=Path(__file__).resolve().parents[1]/"research"/"results"
def fixture():
    m=json.loads((ROOT/"lfm3_real_ffn_optimizer_lr_calibration_15_trials_xwing_20261028.json").read_text())
    p=json.loads((ROOT/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json").read_text())
    return read_only_view(m,p)

class ObserverTests(unittest.TestCase):
    def test_five_options_in_two_static_layouts(self):
        view,gate=fixture()
        self.assertEqual(gate["best_option_index"],2)
        for i in (0,1):
            messages=prompt(view,i)
            self.assertIn("three",messages[0]["content"])
            self.assertFalse("checkpoint_sha256" in messages[1]["content"])
    def test_correct_voting_allows_only_human_review_not_execution(self):
        view,gate=fixture()
        votes=[{"layout":i,"choice":"C","logits":{l:(5. if l=="C" else 1.) for l in LABELS},
                "matches_external_gate":True} for i in (0,1)]
        r=judge(gate,votes)
        self.assertTrue(r["two_layouts_consistent_with_evaluator"])
        self.assertEqual(r["dev_only_review_candidate"]["learning_rate"],0.004)
        self.assertFalse(r["dev_only_review_candidate"]["execute"])
        self.assertFalse(r["may_change_optimizer"])
    def test_incorrect_advice_vetoed_and_forged_logits_denied(self):
        _,gate=fixture()
        votes=[{"layout":i,"choice":"A","logits":{l:(5. if l=="A" else 1.) for l in LABELS},
                "matches_external_gate":False} for i in (0,1)]
        self.assertIsNone(judge(gate,votes)["dev_only_review_candidate"])
        votes[0]["matches_external_gate"]=True
        with self.assertRaises(OptimizerResearchRefused):judge(gate,votes)

if __name__=="__main__":unittest.main()
