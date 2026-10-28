"""Exact optimizer-state resume proof on tiny synthetic CPU-only network."""
import importlib.util
import unittest
from src.lfm3_optimizer_guard import OptimizerResearchRefused
HAS_TORCH = importlib.util.find_spec("torch") is not None

@unittest.skipUnless(HAS_TORCH, "Torch optional in minimal CI")
class OptimizerResumeTests(unittest.TestCase):
    def test_adamw_restart_matches_uninterrupted(self):
        from src.lfm3_optimizer_resume_lab import run
        r = run("adamw")
        self.assertTrue(r["bit_exact_parameter_resume"])
        self.assertTrue(r["bit_exact_optimizer_state_resume"])
        self.assertFalse(r["checkpoint_written_to_disk"])
        self.assertEqual(r["interrupted_after_step"], 5)

    def test_muon_restart_or_explicit_runtime_refusal(self):
        from src.lfm3_optimizer_resume_lab import run
        import torch
        if not hasattr(torch.optim, "Muon"):
            with self.assertRaises(OptimizerResearchRefused):
                run("muon_hybrid")
        else:
            r = run("muon_hybrid")
            self.assertTrue(r["bit_exact_parameter_resume"])
            self.assertTrue(r["bit_exact_optimizer_state_resume"])

    def test_corrupted_state_fails_closed(self):
        from src.lfm3_optimizer_resume_lab import (
            build, capture_in_memory, restore_in_memory,
        )
        model, opts = build("adamw")
        import hashlib
        raw = capture_in_memory(model, opts, optimizer="adamw", next_step=5)
        digest = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(OptimizerResearchRefused):
            restore_in_memory(raw, optimizer="muon_hybrid", expected_sha256=digest)
        with self.assertRaises(OptimizerResearchRefused):
            restore_in_memory(raw+b"X"*257000, optimizer="adamw", expected_sha256=digest)
        with self.assertRaises(OptimizerResearchRefused):
            restore_in_memory(raw[:-1]+bytes([raw[-1] ^ 1]),
                              optimizer="adamw", expected_sha256=digest)
        with self.assertRaises(OptimizerResearchRefused):
            restore_in_memory(raw, optimizer="adamw", expected_sha256="0"*64)
        with self.assertRaises(OptimizerResearchRefused):
            capture_in_memory(model,opts,optimizer="adamw",next_step=4)


if __name__ == "__main__":
    unittest.main()
