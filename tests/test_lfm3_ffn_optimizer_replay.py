"""No GPU required: fixed real-LFM3 replay admission cannot be widened."""
import unittest
from src.lfm3_ffn_optimizer_replay import strict_budget
from src.lfm3_real_checkpoint_smoke import ResearchRefused

class RealLfm3ReplayBudgetTests(unittest.TestCase):
    def test_exact_one_seed_three_step_restart_allowed(self):
        self.assertTrue(strict_budget())
    def test_unreviewed_seeds_steps_and_restart_points_refused(self):
        for args in [
            (20261029,3,1), (20261028,4,1),
            (20261028,3,2), (20261028,0,1),
            (True,3,1), (20261028,True,1),
        ]:
            with self.subTest(args=args),self.assertRaises(ResearchRefused):
                strict_budget(*args)

if __name__=="__main__":unittest.main()
