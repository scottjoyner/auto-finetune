"""Strict pre-model validation of real FFN adapter experiment observation."""
import copy
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_lora_observer import validate_three_step, prompts
from src.lfm3_optimizer_guard import OptimizerResearchRefused


def fixture():
    p=(Path(__file__).resolve().parents[1] / "research" / "results" /
        "lfm3_real_last_ffn_lora_3steps_adamw_muon_xwing_20261028.json")
    return json.loads(p.read_text())


class RealLayerObserverTests(unittest.TestCase):
    def test_two_real_gradient_arms_with_independent_best(self):
        rows, best, gap=validate_three_step(fixture())
        self.assertEqual(best, "A")
        self.assertGreater(gap, 0.8)
        self.assertEqual(rows[1]["optimizer"], "muon_match_rms_adamw")
        for format in (0,1):
            p=prompts(rows,format)
            self.assertNotIn("synthetic_train_sha256",p[-1]["content"])
            self.assertNotIn("synthetic_dev_sha256",p[-1]["content"])

    def test_refuse_forged_source_and_optimizer_claims(self):
        for key,val in [
            ("base_model_weights_modified",True),
            ("source_independently_authenticated",True),
            ("heldout_evaluated",True),
            ("production_ready",True),
            ("full_frozen_model_forward_used",False),
        ]:
            j=copy.deepcopy(fixture())
            j[key]=val
            with self.subTest(key=key),self.assertRaises(OptimizerResearchRefused):
                validate_three_step(j)
        j=copy.deepcopy(fixture())
        j["arms"][1]["dev_loss_after"]=float("nan")
        with self.assertRaises(OptimizerResearchRefused):validate_three_step(j)
        j=copy.deepcopy(fixture())
        j["arms"][0]["input_matrix_updated"]=False
        with self.assertRaises(OptimizerResearchRefused):validate_three_step(j)

    def test_prompt_variants_refuse_injected_labels(self):
        rows,_,_=validate_three_step(fixture())
        with self.assertRaises(OptimizerResearchRefused):prompts(rows,2)
        rows[0]["label"]="TRAIN"
        with self.assertRaises(OptimizerResearchRefused):prompts(rows,0)


if __name__=="__main__":
    unittest.main()
