"""Admission for independently initialized six-arm real frozen-LFM3 LoRA sweeps."""
import copy
import unittest
from src.lfm3_lora_multiseed_lab import assess, SEEDS, OPTIONS
from src.lfm3_real_checkpoint_smoke import ResearchRefused


def sample():
    return [
        {"seed": seed, "arms": [
            {"optimizer": ("adamw" if mode == "adamw" else
                           "muon_experimental" if scaling == "original"
                           else "muon_experimental_match_rms_adamw"),
             "seed": seed, "learning_rate": rate, "steps": 8, "lora_rank": 4,
             "trainable_parameters": 520192, "target": "model.lm_head.weight",
             "trained_adapter_written": False, "base_head_version_unchanged": True,
             "production_ready": False,
             "train_loss_before": 9.8, "dev_loss_before": 11.9,
             "train_loss_after": 7.4, "dev_loss_after": 11.5-i/10,
             "peak_preclip_grad_norm": 12.0}
            for i, (mode, scaling, rate) in enumerate(OPTIONS)]}
        for seed in SEEDS]


class LoRAMultiseedTests(unittest.TestCase):
    def test_multi_seed_six_arm_pass_is_research_only(self):
        s=assess(sample())
        self.assertEqual(s["total_trials"], 18)
        self.assertEqual(s["median_best_arm_index"], 5)
        self.assertTrue(s["rms_muon_better_in_every_seed"])
        self.assertTrue(s["may_nominate_new_synthetic_research_trial_for_review"])
        self.assertFalse(s["production_ready"])

    def test_one_seed_muon_regression_veto(self):
        r=sample()
        r[1]["arms"][5]["dev_loss_after"]=13.1
        result=assess(r)
        self.assertFalse(result["rms_muon_better_in_every_seed"])
        self.assertFalse(result["may_nominate_new_synthetic_research_trial_for_review"])

    def test_forged_training_or_nonfinite_metric_denied(self):
        for key, value in [
            ("base_head_version_unchanged",False),
            ("trained_adapter_written",True),
            ("production_ready",True),
            ("dev_loss_after",float("nan")),
            ("seed",191919),
        ]:
            r=sample()
            r[2]["arms"][4][key]=value
            with self.subTest(key=key), self.assertRaises(ResearchRefused):
                assess(r)
        r=sample()
        r[2]["arms"][3]["dev_loss_before"]=11.5
        with self.assertRaises(ResearchRefused):assess(r)


if __name__=="__main__":
    unittest.main()
