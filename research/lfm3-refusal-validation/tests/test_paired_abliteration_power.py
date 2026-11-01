"""Tests for model-free, privacy-preserving paired-power planning."""
import copy
import json
from pathlib import Path

import pytest
from src.paired_abliteration_power import (
    analyze, exact_mcnemar, minimum_planning_n, power_approx,
)


@pytest.fixture
def evidence():
    p = Path(__file__).parents[1] / "evidence/20261030_xstest_proxy_paired_exact.json"
    return json.loads(p.read_text())


def test_frozen_six_comparisons_complete(evidence):
    x = analyze(evidence)
    assert x["model_weights_changed"] is False
    assert x["new_generations"] == 0
    assert x["n_unique_existing_prompts"] == 450
    assert len(x["planning"]) == 6
    assert x["human_behavioral_grades_completed"] is False
    r = next(y for y in x["planning"]
             if y["stratum"] == "safe" and y["right"].startswith("B_"))
    assert (r["gain"], r["loss"], r["observed_net_difference_pp"]) == (1, 9, -3.2)
    assert r["observed_exact_p"] == pytest.approx(.021484375)


def test_discordance_probability_sanity():
    assert exact_mcnemar(0, 0) == 1
    assert exact_mcnemar(1, 0) == 1
    assert exact_mcnemar(0, 5) == 0.0625
    assert exact_mcnemar(1, 9) == pytest.approx(.021484375)


def test_power_planning_monotonic_and_conservative():
    assert power_approx(250, .004, .036) < power_approx(500, .004, .036)
    assert minimum_planning_n(.004, .036) is not None
    assert minimum_planning_n(.004, .004) is None
    assert (minimum_planning_n(.004, .036)
            >= minimum_planning_n(.004, .036, alpha=.05))


@pytest.mark.parametrize("field,value", [
    ("checkpoint_count", 4),
    ("human_refusal_scoring_complete", True),
    ("n_unique_prompts", 449),
    ("supports_modified_model_release", True),
])
def test_refuses_changed_identity_or_release(evidence, field, value):
    evidence[field] = value
    with pytest.raises(ValueError):
        analyze(evidence)


def test_refuses_aliased_checkpoints(evidence):
    values = list(evidence["checkpoint_sha256"].values())
    evidence["checkpoint_sha256"]["B_M1_abliterated_smoke"] = values[0]
    with pytest.raises(ValueError):
        analyze(evidence)


def test_refuses_changed_counts_or_duplicate(evidence):
    bad = copy.deepcopy(evidence)
    bad["comparisons"][0]["lexical_keyword_newly_present"] += 1
    with pytest.raises(ValueError):
        analyze(bad)
    bad = copy.deepcopy(evidence)
    bad["comparisons"][-1] = bad["comparisons"][0]
    with pytest.raises(ValueError):
        analyze(bad)


def test_refuses_changed_exact_or_holm_p(evidence):
    bad = copy.deepcopy(evidence)
    bad["comparisons"][0]["exploratory_exact_conditional_mcnemar_p"] = 0
    with pytest.raises(ValueError):
        analyze(bad)
    bad = copy.deepcopy(evidence)
    bad["comparisons"][0]["holm_familywise_adjusted_p_six_exploratory_tests"] = 0.01
    with pytest.raises(ValueError):
        analyze(bad)


@pytest.mark.parametrize("args", [
    (0, .004, .036), (250, -.001, .036),
    (250, .6, .5), (250, float("nan"), .1)
])
def test_refuses_invalid_power_inputs(args):
    with pytest.raises(ValueError):
        power_approx(*args)