"""No checkpoint load: fabricated frozen LFM3 adapter admission contract."""
import unittest
from src.lfm3_frozen_readout_lab import TRAIN, DEV, STEPS, LRS, assert_fixed_lab_contract
from src.lfm3_real_checkpoint_smoke import ResearchRefused


class FrozenAdapterAdmissionTests(unittest.TestCase):
    def test_fictional_samples_do_not_overlap(self):
        self.assertTrue(assert_fixed_lab_contract())
        self.assertEqual(len(TRAIN), 2)
        self.assertEqual(len(DEV), 2)
        self.assertEqual(len({x[0] for x in TRAIN + DEV}), 4)
        self.assertEqual(LRS, (0.001, 0.002))
        self.assertEqual(STEPS, 6)

    def test_foreign_source_and_reused_sample_refused(self):
        with self.assertRaises(ResearchRefused):
            assert_fixed_lab_contract(
                train=(("real production trace", "A"), TRAIN[1]), dev=DEV
            )
        with self.assertRaises(ResearchRefused):
            assert_fixed_lab_contract(train=TRAIN, dev=(TRAIN[0], DEV[1]))
        with self.assertRaises(ResearchRefused):
            assert_fixed_lab_contract(train=TRAIN, dev=(("Fictional puzzle: secret", "C"), DEV[1]))


if __name__ == "__main__":
    unittest.main()
