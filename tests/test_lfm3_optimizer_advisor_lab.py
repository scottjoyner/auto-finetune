"""No inference: validate LFM3 advisor JSON/firewall and prompt isolation."""
import copy
import json
import unittest
from src.lfm3_optimizer_advisor_lab import admission_from_model_advice, prompt_for_view
from src.lfm3_optimizer_guard import CHECKPOINT, OptimizerResearchRefused, observer_view


def baseline():
    return {
        "schema": "sg-lfm3-optimizer-observation.v1",
        "source": "fabricated-synthetic-lab-only",
        "model_scope": "toy-proxy-not-lfm3-finetuning",
        "reference_lfm3_checkpoint_sha256": CHECKPOINT,
        "run_id": "toy-20261027", "train_session_sha256": "a"*64,
        "dev_session_sha256": "b"*64, "seed": 71, "steps": 24,
        "optimizer": "adamw", "learning_rate": 0.001,
        "weight_decay": 0.01, "momentum": 0.95, "ns_steps": 0,
        "train_loss_start": 0.8, "train_loss_end": 0.6,
        "dev_loss": 0.67, "max_grad_norm": 1.0,
        "step_mean_ms": 1.0, "peak_memory_mb": 300.0,
        "nonfinite_updates": 0,
        "actual_lfm3_weights_trained": False,
        "model_weights_written": False, "production_ready": False,
    }


class LFM3OptimizerAdvisorTests(unittest.TestCase):
    def test_lfm3_advice_admission_is_data_only(self):
        obj = {"target_optimizer":"muon_hybrid",
               "lr_multiplier":0.5,"reason_code":"controlled_optimizer_comparison"}
        p = admission_from_model_advice(json.dumps(obj).encode(), baseline())
        self.assertEqual(p["learning_rate"], 0.0005)
        self.assertEqual(p["execution_scope"], "new-synthetic-experiment-only")
        self.assertEqual(p["cycle_index"], 1)

    def test_bad_model_text_never_overrides_training_controls(self):
        cases = [
            b'{"target_optimizer":"muon_hybrid","lr_multiplier":1,"reason_code":"no_change","shell":"danger"}',
            b'proposed change: {"target_optimizer":"adamw","lr_multiplier":1,"reason_code":"no_change"}',
            b'{"target_optimizer":"adamw","lr_multiplier":1e400,"reason_code":"no_change"}',
            b'{"target_optimizer":"adamw","lr_multiplier":1,"reason_code":"no_change","reason_code":"gradient_instability"}',
        ]
        for content in cases:
            with self.subTest(content=content[:40]), self.assertRaises(OptimizerResearchRefused):
                admission_from_model_advice(content, baseline())

    def test_observer_has_no_session_ids_or_heldout_metrics(self):
        view = observer_view(baseline())
        messages = prompt_for_view(view)
        self.assertNotIn("train_session_sha256", messages[-1]["content"])
        self.assertNotIn("dev_session_sha256", messages[-1]["content"])
        self.assertNotIn("heldout", messages[-1]["content"])
        self.assertIn("fabricated", messages[0]["content"])
        corrupted = copy.deepcopy(view)
        corrupted["permitted_role"] = "modify_running_optimizer"
        with self.assertRaises(OptimizerResearchRefused):
            prompt_for_view(corrupted)


if __name__ == "__main__":
    unittest.main()
