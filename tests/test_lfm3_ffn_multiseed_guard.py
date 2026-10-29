"""No model import: independent admission and falsification of real three-seed FFN receipts."""
import copy
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_multiseed_guard import admit_three_seed_training
from src.lfm3_real_checkpoint_smoke import ResearchRefused

def source():
    p=Path(__file__).resolve().parents[1]/"research"/"results"/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json"
    return json.loads(p.read_text())

class FFNMultiseedGuardTests(unittest.TestCase):
    def test_fixed_source_gate_measures_real_three_seed_trials(self):
        v=admit_three_seed_training(source())
        self.assertEqual(v["fixed_objective_best"],"A")
        self.assertEqual(v["arm_count"],6)
        self.assertTrue(v["minimum_two_percent_gain_all_seeds"])
        self.assertTrue(v["may_propose_next_synthetic_review_only"])
        self.assertFalse(v["auto_execute"])
    def test_poor_seed_cannot_win(self):
        x=source()
        x["arms"][4]["dev_loss_after"]=12.0
        x["per_seed_adamw_wins"][2]=False
        x["eligible_for_manual_review_only"]=False
        v=admit_three_seed_training(x)
        self.assertFalse(v["may_propose_next_synthetic_review_only"])
        self.assertFalse(v["minimum_two_percent_gain_all_seeds"])
    def test_custody_authority_and_metric_forgery_refused(self):
        for key,value in [
            ("production_ready",True),("independently_attested",True),
            ("source_generation_verified",True),("heldout_evaluated",True),
            ("optimizer_model_chosen",True),("model_revision","some-other-model"),
            ("may_promote_fleet",True),
        ]:
            x=source(); x[key]=value
            with self.subTest(field=key),self.assertRaises(ResearchRefused):
                admit_three_seed_training(x)
        x=source();x["arms"][3]["dev_loss_after"]=float("nan")
        with self.assertRaises(ResearchRefused):admit_three_seed_training(x)
    def test_aliases_unknown_fields_and_incorrect_claim_fail(self):
        x=source();x["arms"][3]["seed"]=x["arms"][1]["seed"]
        with self.assertRaises(ResearchRefused):admit_three_seed_training(x)
        x=source();x["shell_command"]="unsafe"
        with self.assertRaises(ResearchRefused):admit_three_seed_training(x)
        x=source();x["per_seed_adamw_wins"]=[False,False,False]
        with self.assertRaises(ResearchRefused):admit_three_seed_training(x)

if __name__=="__main__":unittest.main()
