"""Frozen one-shot fictional parity source is disjoint from earlier examples."""
import unittest
from src.lfm3_ffn_adamw_parity_transfer import PARITY,LRS,STEPS,frozen_parity_contract
from src.lfm3_real_checkpoint_smoke import ResearchRefused
class ParityProtocolTests(unittest.TestCase):
    def test_precommitted_other_task_family(self):
        self.assertTrue(frozen_parity_contract())
        self.assertEqual(len(PARITY),4)
        self.assertEqual(LRS,(0.002,0.004))
    def test_cannot_reuse_codebook_or_widen_learning_rate(self):
        with self.assertRaises(ResearchRefused):
            frozen_parity_contract(rows=PARITY[:3]+(PARITY[0],))
        with self.assertRaises(ResearchRefused):
            frozen_parity_contract(lrs=(0.002,0.004,0.008))
        with self.assertRaises(ResearchRefused):
            frozen_parity_contract(steps=6)
if __name__=="__main__":unittest.main()
