"""Focused offline coverage for deterministic pure helpers in the runtime modules.

Scope is deliberately limited to helpers that touch no network, GPU, DB, model
downloads, wall clock, or scheduler side effects:

  * ``src.self_evolution.next_checkpoint``            (pure arithmetic)
  * ``src.self_evolution.build_checkpoint_matrix``    (pure transform of CKPT_MAP)
  * ``src.scheduled_retrain.ScheduleConfig``          (plain value dataclass)
  * ``src.scheduled_retrain.ScheduleState``           (plain value dataclass)

Everything else in these modules evaluates checkpoints, lists directories,
reads the clock, probes GPUs/processes, or drives the training pipeline, so it
is intentionally out of scope for this offline test file.
"""
from __future__ import annotations

from dataclasses import asdict, fields

import pytest

from src import self_evolution as SE
from src import scheduled_retrain as SR


# ---------------------------------------------------------------------------
# self_evolution.next_checkpoint
# ---------------------------------------------------------------------------


def test_next_checkpoint_defaults():
    assert SE.next_checkpoint() == 290


def test_next_checkpoint_custom_step_and_increment():
    assert SE.next_checkpoint(100, 25) == 125
    assert SE.next_checkpoint(0, 50) == 50


def test_next_checkpoint_zero_and_negative_increment():
    assert SE.next_checkpoint(240, 0) == 240
    assert SE.next_checkpoint(240, -50) == 190


def test_next_checkpoint_rejects_non_numeric_input():
    # Pure helper: a bad type fails deterministically, with no I/O.
    with pytest.raises(TypeError):
        SE.next_checkpoint("240", 50)


# ---------------------------------------------------------------------------
# self_evolution.build_checkpoint_matrix
# ---------------------------------------------------------------------------


def test_build_checkpoint_matrix_shape_and_order():
    matrix = SE.build_checkpoint_matrix()
    steps = [row["step"] for row in matrix]

    assert steps == sorted(SE.CKPT_MAP)
    assert len(matrix) == len(SE.CKPT_MAP)
    assert all(row["model"] == "MiniCPM5-2B" for row in matrix)
    assert set(matrix[0]) == {
        "checkpoint", "model", "step", "harness_version", "date", "metrics", "status",
    }


def test_build_checkpoint_matrix_checkpoint_label_floor_at_zero():
    matrix = SE.build_checkpoint_matrix()
    by_step = {row["step"]: row for row in matrix}

    # max(step - 30, 0) keeps the earliest checkpoint at zero, never negative.
    assert by_step[30]["checkpoint"] == "checkpoint-0"
    assert by_step[60]["checkpoint"] == "checkpoint-30"
    assert by_step[240]["checkpoint"] == "checkpoint-210"
    assert all(row["checkpoint"].startswith("checkpoint-") for row in matrix)


def test_build_checkpoint_matrix_only_final_step_active():
    matrix = SE.build_checkpoint_matrix()
    active = [row["step"] for row in matrix if row["status"] == "active"]
    archived = [row["step"] for row in matrix if row["status"] == "archived"]

    assert active == [240]
    assert 240 not in archived


def test_build_checkpoint_matrix_reflects_source_data():
    matrix = SE.build_checkpoint_matrix()
    for row in matrix:
        source = SE.CKPT_MAP[row["step"]]
        assert row["harness_version"] == source["harness"]
        assert row["date"] == source["date"]
        assert row["metrics"] is source["metrics"]


# ---------------------------------------------------------------------------
# scheduled_retrain.ScheduleConfig
# ---------------------------------------------------------------------------


def test_schedule_config_defaults():
    cfg = SR.ScheduleConfig()
    assert cfg.interval_hours == 6.0
    assert cfg.min_new_data == 50
    assert cfg.max_staleness_hours == 24.0
    assert cfg.require_gpu_free is True
    assert cfg.require_data_fresh is True
    assert cfg.notify_on_train is True
    assert cfg.notify_on_skip is False
    assert cfg.max_consecutive_failures == 3


def test_schedule_config_overrides():
    cfg = SR.ScheduleConfig(interval_hours=1.5, min_new_data=0,
                            require_gpu_free=False, notify_on_skip=True)
    assert cfg.interval_hours == 1.5
    assert cfg.min_new_data == 0
    assert cfg.require_gpu_free is False
    assert cfg.notify_on_skip is True
    # Unspecified fields keep their defaults.
    assert cfg.max_consecutive_failures == 3
    assert cfg.max_staleness_hours == 24.0


def test_schedule_config_instances_are_independent():
    a = SR.ScheduleConfig()
    b = SR.ScheduleConfig()
    a.min_new_data = 999
    assert b.min_new_data == 50
    assert [f.name for f in fields(SR.ScheduleConfig)] == [
        "interval_hours", "min_new_data", "max_staleness_hours", "require_gpu_free",
        "require_data_fresh", "notify_on_train", "notify_on_skip",
        "max_consecutive_failures",
    ]


# ---------------------------------------------------------------------------
# scheduled_retrain.ScheduleState
# ---------------------------------------------------------------------------


def test_schedule_state_defaults():
    state = SR.ScheduleState()
    assert state.last_run == 0.0
    assert state.last_train == 0.0
    assert state.runs_completed == 0
    assert state.runs_skipped == 0
    assert state.runs_failed == 0
    assert state.consecutive_failures == 0
    assert state.current_status == "idle"
    assert state.last_error is None


def test_schedule_state_overrides_and_asdict():
    state = SR.ScheduleState(runs_completed=2, consecutive_failures=1,
                             current_status="error", last_error="boom")
    assert state.runs_completed == 2
    assert state.consecutive_failures == 1
    assert state.current_status == "error"
    assert state.last_error == "boom"

    data = asdict(state)
    assert data["current_status"] == "error"
    assert data["last_error"] == "boom"
    assert set(data) == {f.name for f in fields(SR.ScheduleState)}


def test_schedule_state_instances_are_independent():
    a = SR.ScheduleState()
    b = SR.ScheduleState()
    a.consecutive_failures = 5
    assert b.consecutive_failures == 0
