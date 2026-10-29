"""No GPU required: fixed real-LFM3 replay admission cannot be widened."""
import hashlib
import unittest
from src.lfm3_ffn_optimizer_replay import strict_budget, admit_checkpoint_bytes
from src.lfm3_real_checkpoint_smoke import ResearchRefused

class RealLfm3ReplayBudgetTests(unittest.TestCase):
    def test_exact_one_seed_three_step_restart_allowed(self):
        self.assertTrue(strict_budget())
    def test_checkpoint_digest_detects_tamper_and_aliases(self):
        payload = b"fabricated optimizer state only"
        digest = hashlib.sha256(payload).hexdigest()
        self.assertTrue(admit_checkpoint_bytes(payload, digest))
        for data, claimed in (
            (payload+b"x", digest),
            (payload, "0"*64),
            (bytearray(payload), digest),
            (payload, "not-a-canonical-digest"),
            (b"", hashlib.sha256(b"").hexdigest()),
        ):
            with self.subTest(value=claimed, length=len(data)):
                with self.assertRaises(ResearchRefused):
                    admit_checkpoint_bytes(data, claimed)

    def test_unreviewed_seeds_steps_and_restart_points_refused(self):
        for args in [
            (20261029,3,1), (20261028,4,1),
            (20261028,3,2), (20261028,0,1),
            (True,3,1), (20261028,True,1),
        ]:
            with self.subTest(args=args),self.assertRaises(ResearchRefused):
                strict_budget(*args)

if __name__=="__main__":unittest.main()
