"""Freeze native LFM3 fixed-vs-step-one LR schedule experiment."""
import unittest
from src.lfm3_ffn_fixed_lr_schedule import (
    SCHEDULES,RESEARCH_SEEDS,fixed_schedule_contract
)
from src.lfm3_real_checkpoint_smoke import ResearchRefused
class FixedScheduleTests(unittest.TestCase):
    def test_real_ffn_schedule_matrix_is_precommitted(self):
        self.assertTrue(fixed_schedule_contract())
        self.assertEqual(len(SCHEDULES)*len(RESEARCH_SEEDS),12)
    def test_model_cannot_select_optimizer_or_unlisted_lr(self):
        for kw in (
            {"schedules":SCHEDULES+(("adamw",(0.004,0.008,0.008)),)},
            {"schedules":SCHEDULES[::-1]},
            {"seeds":RESEARCH_SEEDS+(20261031,)},
        ):
            with self.subTest(kw=kw),self.assertRaises(ResearchRefused):
                fixed_schedule_contract(**kw)
if __name__=="__main__": unittest.main()
