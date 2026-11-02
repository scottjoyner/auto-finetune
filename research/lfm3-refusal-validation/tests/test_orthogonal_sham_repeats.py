"""Synthetic-only random-control inference and statistical integrity tests."""
import copy
import json
import pytest

from src.eval_contracts import refusal_proxy
from src.orthogonal_sham_repeats import (
    SHAM_SEEDS, exact_binomial_p, holm_six, read_new_journal, summarize,
)


def synthetic_panel():
    ids = ["p03", "p04"]
    source = {}
    fresh = {}
    for p in ids:
        for cls in ("benign", "adversarial"):
            baseline = "I can explain this."
            fitted = "I can clarify this." if p == "p03" else baseline
            for cond in ("baseline", "alpha_0", "alpha_0_25", "alpha_0_50",
                         "alpha_1_00", "random_1_00"):
                response = fitted if cond == "alpha_1_00" else baseline
                source[(p, cls, cond)] = {
                    "response": response, "keyword_positive": refusal_proxy(response)}
            for seed in SHAM_SEEDS:
                response = "Alternate harmless wording." if seed == 23 and p == "p04" else baseline
                fresh[(p, cls, seed)] = {"pair": p, "category": cls,
                    "seed": seed, "response": response,
                    "keyword_positive": refusal_proxy(response)}
    return source, fresh, ids
def test_exact_conditional_and_holm():
    assert exact_binomial_p(0, 0) == 1
    assert exact_binomial_p(3, 0) == .25
    assert exact_binomial_p(1, 1) == 1
    assert holm_six([.02, .20, 1., .50, .03, .40]) == [.12, .8, 1., 1., .15, 1.]
    with pytest.raises(ValueError):
        holm_six([.1])


def test_complete_report_is_aggregate_only():
    source, fresh, ids = synthetic_panel()
    result = summarize(source, fresh, ids, {"layer": 10, "max_new_tokens": 24},
                       "source_sha", "fresh_sha")
    assert result["n_new_generations"] == 12
    assert result["n_original_generations"] == 24
    assert result["strata"]["benign"]["23"]["random_changed_vs_baseline"] == 1
    assert result["strata"]["benign"]["23"]["fitted_alpha_1_changed_vs_baseline"] == 1
    assert result["strata"]["benign"]["23"]["discordant_fit_same_sham_changed"] == 1
    assert result["strata"]["benign"]["23"]["discordant_fit_changed_sham_same"] == 1
    assert result["human_refusals_verified"] is False
    assert "p03" not in json.dumps(result)
    assert "Alternate harmless wording" not in json.dumps(result)


def test_incomplete_panel_rejected():
    source, fresh, ids = synthetic_panel()
    fresh.pop(next(iter(fresh)))
    with pytest.raises(ValueError, match="incomplete"):
        summarize(source, fresh, ids, {"layer":10,"max_new_tokens":24}, "a", "b")
def test_resume_journal_rejects_duplicates_corruption_and_unknown_keys(tmp_path):
    _, fresh, ids = synthetic_panel()
    expected = set(fresh)
    path = tmp_path / "private_generations.jsonl"
    valid = list(fresh.values())
    path.write_text("".join(json.dumps(x) + "\n" for x in valid))
    assert len(read_new_journal(path, expected)) == len(fresh)
    path.write_text(json.dumps(valid[0]) + "\n" + json.dumps(valid[0]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_new_journal(path, expected)
    path.write_text(json.dumps(valid[0]))
    with pytest.raises(ValueError, match="unfinished"):
        read_new_journal(path, expected)
    changed = dict(valid[0], seed=999)
    path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="out-of-scope"):
        read_new_journal(path, expected)
    changed = dict(valid[0], keyword_positive=True)
    path.write_text(json.dumps(changed) + "\n")
    with pytest.raises(ValueError, match="flags"):
        read_new_journal(path, expected)
