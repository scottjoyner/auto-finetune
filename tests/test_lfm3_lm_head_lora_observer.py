"""Fail closed on metric tampering and unsafe model advice in head LoRA."""
import copy
import json
import unittest
from pathlib import Path
from src.lfm3_lm_head_lora_observer import validate_lora_comparison, prompts
from src.lfm3_optimizer_guard import OptimizerResearchRefused

def fixture():
    p = (Path(__file__).resolve().parents[1] / "research" / "results"
         / "lfm3_real_frozen_lm_head_lora_adamw_muon_xwing_20261028.json")
    return json.loads(p.read_text())

class LoRAObserverTests(unittest.TestCase):
    def test_measured_best_and_two_fixed_formats(self):
        rows,best=validate_lora_comparison(fixture())
        self.assertEqual(best,"B")
        self.assertEqual(len(rows),4)
        self.assertEqual(rows[1]["lr"],0.002)
        for i in (0,1):
            p=prompts(rows,i)
            self.assertIn("synthetic",p[1]["content"].lower())
            self.assertNotIn("training_split_sha256",p[1]["content"])
            self.assertNotIn("dev_split_sha256",p[1]["content"])

    def test_refuse_forged_source_trust_or_metric(self):
        for key,value in [
            ("production_ready", True), ("source_custody_authenticated",True),
            ("end_to_end_backpropagated",True),
            ("model_or_adapter_weights_written",True),
        ]:
            j=copy.deepcopy(fixture())
            j[key]=value
            with self.subTest(key=key), self.assertRaises(OptimizerResearchRefused):
                validate_lora_comparison(j)
        j=copy.deepcopy(fixture())
        j["arms"][2]["dev_loss_after"]=float("inf")
        with self.assertRaises(OptimizerResearchRefused):validate_lora_comparison(j)
        j=copy.deepcopy(fixture())
        j["arms"][0]["train_loss_before"]=600
        with self.assertRaises(OptimizerResearchRefused):validate_lora_comparison(j)
        j=copy.deepcopy(fixture())
        j["untrusted_model_advice_executed"]=True
        with self.assertRaises(OptimizerResearchRefused):validate_lora_comparison(j)

    def test_reject_unauthorized_prompt(self):
        rows,_=validate_lora_comparison(fixture())
        with self.assertRaises(OptimizerResearchRefused):prompts(rows,2)
        with self.assertRaises(OptimizerResearchRefused):prompts(rows[:2],0)

if __name__=="__main__":unittest.main()
