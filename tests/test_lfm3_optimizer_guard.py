"""Strict optimizer self-observation policy, no Torch needed."""
import copy
import unittest
from src.lfm3_optimizer_guard import (
    CHECKPOINT, OptimizerResearchRefused, compare_dev_only, fingerprint,
    observer_view, parse_strict, validate_observation, validate_proposal
)


def obs(optimizer="adamw", **overrides):
    o = {
        "schema": "sg-lfm3-optimizer-observation.v1",
        "source": "fabricated-synthetic-lab-only",
        "model_scope": "toy-proxy-not-lfm3-finetuning",
        "reference_lfm3_checkpoint_sha256": CHECKPOINT,
        "run_id": "toy-20261027", "train_session_sha256": "a"*64,
        "dev_session_sha256": "b"*64, "seed": 71, "steps": 12,
        "optimizer": optimizer, "learning_rate": 0.001,
        "weight_decay": 0.01, "momentum": 0.95,
        "ns_steps": 0 if optimizer == "adamw" else 5,
        "train_loss_start": 1.0, "train_loss_end": 0.8,
        "dev_loss": 0.9, "max_grad_norm": 1.0,
        "step_mean_ms": 10.0, "peak_memory_mb": 300.0,
        "nonfinite_updates": 0,
        "actual_lfm3_weights_trained": False,
        "model_weights_written": False,
        "production_ready": False,
    }
    o.update(overrides)
    return o


def proposal(parent, **overrides):
    p = {
        "schema": "sg-lfm3-optimizer-proposal.v1",
        "observer": "lfm3-untrusted-advisor",
        "parent_observation_sha256": fingerprint(parent),
        "target_optimizer": "muon_hybrid",
        "learning_rate": 0.001, "weight_decay": 0.01,
        "momentum": 0.95, "ns_steps": 5, "steps": parent["steps"],
        "cycle_index": 1, "reason_code": "controlled_optimizer_comparison",
        "execution_scope": "new-synthetic-experiment-only",
    }
    p.update(overrides)
    return p


class OptimizerGuardTests(unittest.TestCase):
    def test_observer_shares_only_dev_metrics_no_ids(self):
        o = obs()
        public = observer_view(o)
        self.assertNotIn("train_session_sha256", public)
        self.assertNotIn("dev_session_sha256", public)
        self.assertNotIn("run_id", public)
        self.assertEqual(public["observation_sha256"], fingerprint(o))
        self.assertTrue(validate_observation(o))

    def test_bound_candidate_is_not_execution_permission(self):
        o = obs()
        p = proposal(o)
        self.assertEqual(validate_proposal(p, o), p)
        self.assertEqual(p["execution_scope"], "new-synthetic-experiment-only")

    def test_reject_malicious_injections_and_over_budget_proposals(self):
        parent = obs()
        cases = (
            {"learning_rate": 1.0},
            {"learning_rate": 0.005},
            {"weight_decay": 0.3},
            {"ns_steps": 12},
            {"target_optimizer": "rm -rf /nas"},
            {"execution_scope": "production-now"},
            {"steps": 120},
            {"cycle_index": 10},
            {"parent_observation_sha256": "0"*64},
            {"reason_code": "evaluate-heldout-every-step"},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(OptimizerResearchRefused):
                validate_proposal(proposal(parent, **changes), parent)
        dirty = proposal(parent)
        dirty["train_on_private_traces"] = True
        with self.assertRaises(OptimizerResearchRefused):
            validate_proposal(dirty, parent)
        with self.assertRaises(OptimizerResearchRefused):
            validate_proposal(proposal(parent), obs(nonfinite_updates=1))

    def test_reject_nonfinite_unpaired_and_heldout_override(self):
        for value in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(OptimizerResearchRefused):
                validate_observation(obs(dev_loss=value))
        with self.assertRaises(OptimizerResearchRefused):
            validate_observation(obs(train_session_sha256="b"*64))
        with self.assertRaises(OptimizerResearchRefused):
            compare_dev_only(obs(), obs(dev_session_sha256="f"*64))
        with self.assertRaises(OptimizerResearchRefused):
            validate_observation(dict(obs(), heldout_loss=0.0))

    def test_loss_improvement_and_regression_threshold(self):
        baseline = obs()
        candidate = obs("muon_hybrid", dev_loss=0.85,
                        step_mean_ms=10.5, peak_memory_mb=310)
        decision = compare_dev_only(baseline, candidate)
        self.assertTrue(decision["bounded_candidate_for_manual_review"])
        self.assertFalse(decision["heldout_evaluated"])
        self.assertFalse(decision["may_promote_to_fleet"])
        bad = compare_dev_only(baseline, dict(candidate, step_mean_ms=16))
        self.assertFalse(bad["bounded_candidate_for_manual_review"])
        worse = compare_dev_only(baseline, dict(candidate, dev_loss=0.93))
        self.assertFalse(worse["bounded_candidate_for_manual_review"])

    def test_strict_json_duplicate_and_nonfinite_refusal(self):
        with self.assertRaises(OptimizerResearchRefused):
            parse_strict(b'{"steps":4,"steps":7}')
        with self.assertRaises(OptimizerResearchRefused):
            parse_strict(b'{"lr": NaN}')
        with self.assertRaises(OptimizerResearchRefused):
            parse_strict(b"x"*4097)


if __name__ == "__main__":
    unittest.main()
