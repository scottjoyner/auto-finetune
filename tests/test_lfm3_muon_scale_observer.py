"""Strict six-option Muon scaling advisor and immutable objective boundary."""
import copy
import json
import unittest
from pathlib import Path
from src.lfm3_muon_scale_observer import validate_six_arm_grid, prompts
from src.lfm3_optimizer_guard import OptimizerResearchRefused

def load():
    p=Path(__file__).resolve().parents[1]/"research"/"results"/"lfm3_real_lora_muon_scaling_six_arm_xwing_20261028.json"
    return json.loads(p.read_text())

class MuonScalingObserverTests(unittest.TestCase):
    def test_six_fixed_synthetic_arms_and_best_f(self):
        rows,best=validate_six_arm_grid(load())
        self.assertEqual(best,"F")
        self.assertEqual(len(rows),6)
        self.assertEqual(rows[5]["optimizer"],"muon_experimental_match_rms_adamw")
        for variant in (0,1):
            messages=prompts(rows,variant)
            self.assertNotIn("training_split_sha256",messages[-1]["content"])
            self.assertNotIn("dev_split_sha256",messages[-1]["content"])
        self.assertIn("lowest",prompts(rows,0)[0]["content"].lower())

    def test_unknown_fields_and_forged_authority_refused(self):
        for key,value in (
            ("production_ready",True),
            ("source_custody_authenticated",True),
            ("heldout_final_evaluation",True),
            ("muon_rms_scaling_compare_enabled",False),
            ("model_or_adapter_weights_written",True),
        ):
            data=copy.deepcopy(load())
            data[key]=value
            with self.subTest(key=key),self.assertRaises(OptimizerResearchRefused):
                validate_six_arm_grid(data)
        data=load()
        data["arms"][5]["dev_loss_after"]=float("nan")
        with self.assertRaises(OptimizerResearchRefused):validate_six_arm_grid(data)
        data=load()
        data["shell"]="do something unsafe"
        with self.assertRaises(OptimizerResearchRefused):validate_six_arm_grid(data)

    def test_wrong_menu_and_mismatched_baselines_refused(self):
        rows,_=validate_six_arm_grid(load())
        with self.assertRaises(OptimizerResearchRefused):prompts(rows,2)
        with self.assertRaises(OptimizerResearchRefused):prompts(rows[:4],0)
        data=load()
        data["arms"][3]["train_loss_before"]=999.0
        with self.assertRaises(OptimizerResearchRefused):validate_six_arm_grid(data)

if __name__=="__main__":
    unittest.main()
