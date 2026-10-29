"""Offline contract tests; no model weights, GPUs, network or private traces."""

from src.lfm3_optimizer_telemetry import observe


def sample(loss_slope=0.0):
    return [
        {
            "schema": "sensorgraft.lfm3.optimizer-step.v1",
            "run_id": "a" * 32,
            "step": i,
            "seed": 37,
            "base_sha256": "b" * 64,
            "dataset_sha256": "c" * 64,
            "optimizer": "adamw",
            "learning_rate": 0.0001,
            "train_loss": 1.0 - i * loss_slope,
            "grad_norm": 1.0,
            "update_norm": 0.05,
            "elapsed_s": 0.1,
        }
        for i in range(20)
    ]


def test_plateau_only_recommends_an_experiment():
    result = observe(sample())
    assert result["status"] == "review_required"
    assert result["proposal"]["learning_rate"] == 0.00009
    assert result["proposal"]["requires_fresh_holdout_evaluation"] is True
    assert result["weight_promotion_permitted"] is False


def test_improving_loss_does_not_need_adjustment():
    result = observe(sample(loss_slope=0.01))
    assert result["status"] == "observed"
    assert result["proposal"] is None


def test_insufficient_steps_refused():
    assert observe(sample()[:5])["reason"] == "insufficient_contiguous_steps"


def test_no_data_or_large_input():
    assert observe([])["status"] == "blocked"
    assert observe(sample() * 501)["reason"] == "invalid_event_count"


def test_gap_duplicate_and_reorder_refused():
    r = sample()
    r[10]["step"] = 100
    assert observe(r)["reason"] == "duplicate_or_missing_step"


def test_mixed_dataset_is_refused():
    r = sample()
    r[-1]["dataset_sha256"] = "d" * 64
    assert observe(r)["reason"] == "mixed_experiment_provenance"


def test_nan_or_overflow_refused():
    r = sample()
    r[-1]["train_loss"] = float("nan")
    assert observe(r)["reason"] == "nonfinite_metric"


def test_raw_prompt_field_refused():
    r = sample()
    r[0]["prompt"] = "private content"
    assert observe(r)["reason"] == "unexpected_or_missing_fields"


def test_optimizer_lr_out_of_bounds():
    r = sample()
    r[3]["learning_rate"] = 4.0
    assert observe(r)["reason"] == "unsafe_learning_rate"


def test_gradient_explosion_requires_stop():
    r = sample()
    for e in r[-5:]:
        e["grad_norm"] = 20.0
    assert observe(r)["reason"] == "severe_training_instability"


def test_moderate_instability_suggests_lower_lr():
    r = sample()
    for e in r[-5:]:
        e["grad_norm"] = 4.0
    result = observe(r)
    assert result["status"] == "review_required"
    assert result["proposal"]["learning_rate"] == 0.00005


def test_lr_floor_refuses_ineffective_proposal():
    r = sample()
    for e in r:
        e["learning_rate"] = 1e-6
    result = observe(r)
    assert result["status"] == "observed"
    assert result["proposal"] is None


def test_reject_invalid_seed_and_digest():
    r = sample()
    r[0]["seed"] = True
    assert observe(r)["reason"] == "invalid_step_or_seed"
    r = sample()
    r[0]["base_sha256"] = "g" * 64
    assert observe(r)["reason"] == "invalid_provenance_digest"


def test_unhashable_optimizer_fails_closed():
    r = sample()
    r[0]["optimizer"] = ["adamw"]
    assert observe(r)["reason"] == "unknown_optimizer"


def test_mixed_seed_blocks():
    r = sample()
    r[8]["seed"] = 38
    assert observe(r)["reason"] == "mixed_experiment_provenance"