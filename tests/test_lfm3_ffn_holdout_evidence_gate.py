"""Covers exact synthetic holdout admission and all authority denials."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_holdout_evidence_gate import verify_holdout
from src.lfm3_real_checkpoint_smoke import ResearchRefused

R = Path(__file__).resolve().parents[1] / "research" / "results"

def inputs():
    a=json.loads((R/"lfm3_real_ffn_dev_selected_lexical_holdout_3seed_xwing_20261028.json").read_text())
    b=json.loads((R/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json").read_text())
    return a,b

class HoldoutEvidenceTests(unittest.TestCase):
    def test_measured_real_lfm3_holdout_all_three_seeds(self):
        r=verify_holdout(*inputs())
        self.assertEqual(r["trials"],6)
        self.assertEqual(r["selected_adamw_lower_holdout_loss_than_muon"],[True]*3)
        self.assertTrue(r["eligible_for_new_synthetic_review"])
        self.assertFalse(r["independent_task_generalization_verified"])
        self.assertFalse(r["execute_optimizer_change"])
    def test_false_authority_and_holdout_source_rejected(self):
        for key in ("independent_task_family","may_promote","model_training_authorized",
                    "source_authenticated","additional_tuning_after_holdout"):
            a,b=inputs();a[key]=True
            with self.subTest(key=key),self.assertRaises(ResearchRefused):
                verify_holdout(a,b)
        a,b=inputs();a["selection"]["heldout_influenced_selection"]=True
        with self.assertRaises(ResearchRefused):verify_holdout(a,b)
    def test_metric_forgery_and_dev_drift_refused(self):
        a,b=inputs();a["trials"][2]["dev_loss_reproduced"]=42.
        with self.assertRaises(ResearchRefused):verify_holdout(a,b)
        a,b=inputs();a["trials"][4]["one_shot_lexical_holdout_loss"]=float("nan")
        with self.assertRaises(ResearchRefused):verify_holdout(a,b)
        a,b=inputs();a["selected_adamw_beats_muon_on_holdout_per_seed"]=[False]*3
        with self.assertRaises(ResearchRefused):verify_holdout(a,b)

if __name__=="__main__":unittest.main()
