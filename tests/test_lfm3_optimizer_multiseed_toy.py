"""No model load: independent seeds, fixed source split and exact quality gate."""
import copy
import unittest
from src.lfm3_optimizer_multiseed_toy import SEEDS, GRID, assess
from src.lfm3_optimizer_guard import CHECKPOINT, OptimizerResearchRefused


def observation(seed, i):
    mode, lr = GRID[i]
    return {
        "schema": "sg-lfm3-optimizer-observation.v1",
        "source": "fabricated-synthetic-lab-only",
        "model_scope": "toy-proxy-not-lfm3-finetuning",
        "reference_lfm3_checkpoint_sha256": CHECKPOINT,
        "run_id": f"toy-{seed}",
        "train_session_sha256": f"{seed:064x}",
        "dev_session_sha256": f"{seed+1000:064x}",
        "seed": seed, "steps": 24,
        "optimizer": mode, "learning_rate": lr,
        "weight_decay": 0.01, "momentum": 0.95,
        "ns_steps": 0 if mode == "adamw" else 5,
        "train_loss_start": 1.0, "train_loss_end": 0.7,
        "dev_loss": 0.9 if i != 2 else 0.6,
        "max_grad_norm": 0.7, "step_mean_ms": 4.0,
        "peak_memory_mb": 128.0, "nonfinite_updates": 0,
        "actual_lfm3_weights_trained": False, "model_weights_written": False,
        "production_ready": False,
    }


def fabricated():
    return [{"seed": seed, "arms": [observation(seed, i) for i in range(6)]}
            for seed in SEEDS]


class MultiSeedTests(unittest.TestCase):
    def test_deterministic_admission_all_seeds_and_no_real_training(self):
        summary = assess(fabricated())
        self.assertEqual(summary["median_best_grid_index"], 2)
        self.assertEqual(summary["trial_count"], 18)
        self.assertTrue(summary["best_improves_reference_every_seed_by_two_percent"])
        self.assertFalse(summary["may_promote_fleet"])
        self.assertFalse(summary["independent_heldout_evaluated"])

    def test_one_seed_regression_vetoes_research_review(self):
        data = fabricated()
        data[-1]["arms"][2]["dev_loss"] = 1.1
        summary = assess(data)
        self.assertFalse(summary["best_improves_reference_every_seed_by_two_percent"])
        self.assertFalse(summary["may_submit_for_manual_research_review"])

    def test_duplicate_seed_alias_and_split_leaks_fail(self):
        data = fabricated()
        data[1]["seed"] = data[0]["seed"]
        with self.assertRaises(OptimizerResearchRefused):
            assess(data)
        data = fabricated()
        data[2]["arms"][0]["dev_session_sha256"] = "f" * 64
        with self.assertRaises(OptimizerResearchRefused):
            assess(data)
        data = fabricated()
        data[2]["arms"][1]["train_session_sha256"] = data[0]["arms"][1]["train_session_sha256"]
        with self.assertRaises(OptimizerResearchRefused):
            assess(data)

    def test_forged_metadata_and_nonfinite_loss_fail_closed(self):
        data = fabricated()
        data[1]["arms"][2]["dev_loss"] = float("nan")
        with self.assertRaises(OptimizerResearchRefused):
            assess(data)
        data = fabricated()
        data[0]["arms"][5]["actual_lfm3_weights_trained"] = True
        with self.assertRaises(OptimizerResearchRefused):
            assess(data)


if __name__ == "__main__":
    unittest.main()
