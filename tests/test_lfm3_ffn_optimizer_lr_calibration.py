"""Frozen LFM3 optimizer/LR five-arm calibration cannot be widened."""
import unittest
from src.lfm3_ffn_optimizer_lr_calibration import (
    ARMS,RESEARCH_SEEDS,STEPS,check_calibration_contract,
)
from src.lfm3_real_checkpoint_smoke import ResearchRefused

class OptimizerCalibrationContractTests(unittest.TestCase):
    def test_only_fifteen_precommitted_synthetic_trials(self):
        self.assertTrue(check_calibration_contract())
        self.assertEqual(len(ARMS)*len(RESEARCH_SEEDS),15)
        self.assertEqual(STEPS,3)
    def test_model_cannot_add_lr_seed_step_or_optimizer(self):
        for overrides in (
            {"arms":ARMS+(("adamw",0.1),)},
            {"seeds":RESEARCH_SEEDS+(20261031,)},
            {"steps":8},
            {"arms":tuple(reversed(ARMS))},
        ):
            with self.subTest(overrides=overrides),self.assertRaises(ResearchRefused):
                check_calibration_contract(**overrides)

if __name__=="__main__":unittest.main()
