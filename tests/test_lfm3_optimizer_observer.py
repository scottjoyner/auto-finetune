"""CPU-free contract tests using the committed real-checkpoint synthetic evidence."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_optimizer_observer import BASE, _read_json, observe

EVIDENCE = Path(__file__).resolve().parents[1] / (
    "research/evidence/lfm3-real-lora-20261029/"
    "lfm3-real-lora-3seed-20step-equal-lr-20261029.json"
)


@pytest.fixture(scope="module")
def report():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_observer_can_only_generate_non_executable_research_grid(report):
    out = observe(report)
    assert out["status"] == "research_observation_only"
    assert len(out["observations"]) == 6
    assert len(out["proposed_next_experiment_grid"]) == 6
    assert out["trial_authorized"] is False
    assert out["weight_promotion_permitted"] is False
    assert out["optimizer_switch_permitted"] is False
    assert out["fresh_independent_holdout_required"] is True


def test_valid_model_feedback_remains_advisory_only(report):
    suggestion = {"optimizer": "adamw", "learning_rate": 9e-5,
                  "base_sha256": BASE, "reason": "Trial lower LR in isolated run"}
    out = observe(report, suggestion)
    assert out["model_suggestion_review"]["status"] == "bounded_research_suggestion_only"
    assert out["model_suggestion_review"]["trial_authorized"] is False


@pytest.mark.parametrize("change", [
    {"learning_rate": 1.0},
    {"learning_rate": float("nan")},
    {"learning_rate": True},
    {"base_sha256": "wrong"},
    {"optimizer": "not-an-optimizer"},
    {"reason": ""},
    {"action": "deploy"},
])
def test_untrusted_model_proposals_fail_closed(report, change):
    suggestion = {"optimizer": "adamw", "learning_rate": 1e-4,
                  "base_sha256": BASE, "reason": "Examine training trace"}
    suggestion.update(change)
    out = observe(report, suggestion)
    assert out["model_suggestion_review"]["status"] == "rejected"
    assert out["weight_promotion_permitted"] is False


def test_source_report_tampering_blocks(report):
    changed = copy.deepcopy(report)
    changed["arms"][0]["events"][0]["run_id"] = "f" * 32
    assert observe(changed)["status"] == "blocked"


def test_numerical_spike_blocks(report):
    changed = copy.deepcopy(report)
    changed["arms"][0]["events"][0]["grad_norm"] = 1e7
    assert observe(changed)["reason"] == "numerical_telemetry_anomaly"


def test_short_experiment_not_authorized(report):
    changed = copy.deepcopy(report)
    changed["steps"] = 10
    for arm in changed["arms"]:
        arm["steps"] = 10
        arm["events"] = arm["events"][:10]
    assert observe(changed)["reason"] == "insufficient_replicates_or_steps"


def test_json_duplicate_keys_rejected(tmp_path):
    p = tmp_path / "dup.json"
    p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="duplicate_json_key"):
        _read_json(p)


def test_symlink_not_accepted(tmp_path):
    original = tmp_path / "original.json"
    original.write_text('{"a":1}')
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(original)
    with pytest.raises(ValueError, match="unsafe_input"):
        _read_json(symlink)
