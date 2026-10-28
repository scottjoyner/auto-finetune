"""Trust boundary of real-three-seed LoRA advisor: model suggestion != authority."""
import json
from pathlib import Path
import unittest
from src.lfm3_ffn_multiseed_observer import advisor_view, prompts, judge_votes
from src.lfm3_optimizer_guard import OptimizerResearchRefused

def sample():
    p=Path(__file__).resolve().parents[1]/"research"/"results"/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json"
    return json.loads(p.read_text())

class MultiSeedObserverTests(unittest.TestCase):
    def setUp(self):
        self.view,self.gate=advisor_view(sample())
    def votes(self):
        return [
            {"prompt_variant":i,"model_choice":"A","external_best":"A",
             "matches_external":True,"logit_scores":{"A":10.0,"B":7.0}}
            for i in (0,1)]
    def test_dev_only_view_uses_no_raw_session_identity(self):
        self.assertEqual(self.gate["fixed_objective_best"],"A")
        for v in (0,1):
            messages=prompts(self.view,v)
            self.assertNotIn("synthetic_train_sha256",messages[-1]["content"])
            self.assertNotIn("synthetic_dev_sha256",messages[-1]["content"])
        result=judge_votes(self.gate,self.votes())
        self.assertEqual(result["synthetic_review_candidate_only"]["execute"],False)
        self.assertTrue(result["two_prompt_consistent_and_correct"])
        self.assertFalse(result["automatic_optimizer_change"])
    def test_one_wrong_model_vote_vetoes(self):
        votes=self.votes()
        votes[1]["model_choice"]="B"
        votes[1]["matches_external"]=False
        votes[1]["logit_scores"]={"A":8.0,"B":9.0}
        result=judge_votes(self.gate,votes)
        self.assertIsNone(result["synthetic_review_candidate_only"])
    def test_forged_logit_choice_or_source_refused(self):
        x=self.votes()
        x[0]["model_choice"]="B"
        with self.assertRaises(OptimizerResearchRefused):judge_votes(self.gate,x)
        x=self.votes()
        x[0]["logit_scores"]["A"]=float("nan")
        with self.assertRaises(OptimizerResearchRefused):judge_votes(self.gate,x)
        with self.assertRaises(OptimizerResearchRefused):prompts(self.view,2)

if __name__=="__main__":
    unittest.main()
