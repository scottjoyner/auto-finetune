"""Contract tests for read-only text-task report verifier; no model import."""
import copy

import pytest

from src.lfm3_lora_text_task_probe import _digest, dataset
from src.lfm3_lora_text_task_review import review


def score(correct=4):
    return {
        "correct": correct, "total": 9,
        "mean_correct_minus_foil_logprob": 0.2,
        "families": {
            "parity": {"correct": min(correct, 3), "total": 3},
            "greater": {"correct": min(max(correct - 3, 0), 3), "total": 3},
            "sum": {"correct": min(max(correct - 6, 0), 3), "total": 3},
        },
    }


@pytest.fixture
def report():
    seed, steps = 1, 2
    arms = []
    for index, name in enumerate(("adamw", "muon"), 1):
        arms.append({
            "run_id": f"{index:032x}", "seed": seed,
            "optimizer": name, "learning_rate": 1e-4,
            "trainable_parameters": 146432,
            "initial_adapter_sha256": "a" * 64,
            "final_adapter_sha256": f"{index + 1:x}" * 64,
            "selection_before": score(5), "selection_after": score(5),
            "events": [{"step": i, "optimizer": name,
                        "learning_rate": 1e-4, "train_loss": 8.3,
                        "unclipped_grad_norm": 0.6,
                        "grad_clip_limit": 1.0, "update_norm": 0.01,
                        "elapsed_s": 0.4} for i in range(steps)],
            "weight_promotion_permitted": False,
        })
    return {
        "schema": "sensorgraft.lfm3.lora-text-task-probe.v1",
        "status": "research_text_task_diagnostic_only",
        "base_sha256": "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330",
        "model": "LiquidAI/LFM3-1.7B",
        "data_kind": "procedural_public_arithmetic_and_parity_text",
        "answer_metric": "two_choice_teacher_forced_logprob",
        "selection_used_for_reporting_not_optimizer_choice": True,
        "untouched_test_evaluated": False,
        "dataset_sha256_by_seed": {
            str(seed): {k: _digest(v) for k, v in dataset(seed).items()}},
        "seeds": [seed], "steps": steps, "arms": arms,
        "real_task_generalization_established": False,
        "weight_promotion_permitted": False, "trial_authorized": False,
    }


def test_contract_requires_manual_review_even_with_positive_selection(report):
    result = review(report)
    assert result["status"] == "review_required"
    assert result["true_optimizer_steps"] == 4
    assert result["optimizer_selection_permitted"] is False
    assert result["independent_real_task_quality_proven"] is False
    assert result["source_authenticated"] is False
    assert result["weight_promotion_permitted"] is False


@pytest.mark.parametrize("transform", [
    lambda d: d.update({"untouched_test_evaluated": True}),
    lambda d: d.update({"weight_promotion_permitted": True}),
    lambda d: d["dataset_sha256_by_seed"]["1"].update({"train": "b" * 64}),
    lambda d: d["arms"][1].update({"initial_adapter_sha256": "f" * 64}),
    lambda d: d["arms"][1].update({"run_id": d["arms"][0]["run_id"]}),
    lambda d: d["arms"][1]["selection_before"].update({"correct": 7}),
    lambda d: d["arms"][0]["events"][0].update({"unclipped_grad_norm": float("nan")}),
    lambda d: d["arms"][0]["events"][0].update({"step": 3}),
    lambda d: d["arms"][0]["events"][0].update({"learning_rate": 0.01}),
    lambda d: d["arms"][0]["events"][0].update({"command": "train"}),
    lambda d: d.update({"trial_authorized": True}),
])
def test_tampering_or_escalation_denied(report, transform):
    sample = copy.deepcopy(report)
    transform(sample)
    result = review(sample)
    assert result["status"] == "blocked"
    assert result["weight_promotion_permitted"] is False
