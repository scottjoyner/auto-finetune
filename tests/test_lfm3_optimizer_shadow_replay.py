"""Pure-Python shadow feedback tests from immutable real LFM3 probe telemetry."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_optimizer_observer import BASE, observe
from src.lfm3_optimizer_shadow_replay import MAX_CHANGES_PER_ARM, replay

EVIDENCE = Path(__file__).resolve().parents[1] / (
    "research/evidence/lfm3-real-lora-20261029/"
    "lfm3-real-lora-3seed-20step-equal-lr-20261029.json"
)


@pytest.fixture(scope="module")
def report():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def proposal(report, **override):
    candidate = {
        "optimizer": "adamw",
        "learning_rate": 0.00009,
        "base_sha256": BASE,
        "report_sha256": observe(report)["report_sha256"],
        "reason": "Shadow compare: no production changes",
    }
    candidate.update(override)
    return candidate


def test_deterministic_all_arm_replay(report):
    left, right = replay(report), replay(report)
    assert left == right
    assert left["status"] == "shadow_replay_only"
    assert len(left["per_seed_arms"]) == 6
    assert len(left["replay_sha256"]) == 64
    assert left["uses_heldout_for_lr_decisions"] is False
    assert left["training_configuration_applied"] is False
    assert left["trial_authorized"] is False
    assert left["weight_promotion_permitted"] is False
    for arm in left["per_seed_arms"]:
        assert arm["weight_updates_performed"] == 0
        assert len(arm["decisions"]) <= MAX_CHANGES_PER_ARM
        for item in arm["decisions"]:
            assert item["applied_to_training"] is False
            assert item["proposed_lr"] >= arm["initial_lr"] * 0.8 - 1e-15
            assert item["proposed_lr"] <= arm["initial_lr"] * 1.2 + 1e-15
            assert item["effective_from_step"] > item["observed_steps"][1]


def test_good_model_proposal_shadow_only_adamw(report):
    result = replay(report, proposal(report))
    assert result["status"] == "shadow_replay_only"
    assert result["selected_by_model"] is True
    assert result["model_response_executed"] is False
    assert len(result["per_seed_arms"]) == 3
    assert all(a["optimizer"] == "adamw" for a in result["per_seed_arms"])
    assert all(a["initial_lr"] == 0.00009 for a in result["per_seed_arms"])


@pytest.mark.parametrize("override", [
    {"learning_rate": 0.01},
    {"learning_rate": float("nan")},
    {"learning_rate": False},
    {"optimizer": "not_allowed"},
    {"base_sha256": "incorrect"},
    {"report_sha256": "stale"},
    {"reason": "bad\ncontrol"},
    {"action": "run"},
])
def test_malicious_or_stale_model_proposals_blocked(report, override):
    result = replay(report, proposal(report, **override))
    assert result["status"] == "blocked"
    assert result["reason"] == "untrusted_optimizer_suggestion"
    assert result["weight_promotion_permitted"] is False


def test_invalid_source_provenance_blocks(report):
    changed = copy.deepcopy(report)
    changed["arms"][0]["events"][0]["seed"] = 999
    assert replay(changed)["status"] == "blocked"


def test_telemetry_changes_replay_hash_without_model_calls(report):
    changed = copy.deepcopy(report)
    changed["arms"][0]["events"][0]["grad_norm"] *= 1.01
    left, right = replay(report), replay(changed)
    assert left["status"] == right["status"] == "shadow_replay_only"
    assert left["replay_sha256"] != right["replay_sha256"]
    assert not right["source_authenticated"]


def test_not_a_meaningful_model_selection_or_quality_claim(report):
    output = replay(report)
    assert output["synthetic_token_training_only"] is True
    assert output["improvement_established"] is False
    assert output["real_training_executed"] is False
