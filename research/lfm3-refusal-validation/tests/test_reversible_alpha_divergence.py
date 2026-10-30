"""Synthetic audit tests; no actual model generations are read."""
import json
import pytest

from src.eval_contracts import refusal_proxy
from src.reversible_alpha_controls import CONDITIONS, EXPECTED_MODEL, EXPECTED_PAIRS, EXPECTED_VECTOR
from src.reversible_alpha_divergence import audit_folder, changes_against_baseline


def fixtures(tmp_path):
    rows = []
    for category in ("benign", "adversarial"):
        for condition in CONDITIONS:
            response = "I cannot assist." if category == "adversarial" else "Here is help."
            rows.append({"pair": "p03", "category": category,
                         "condition": condition, "response": response,
                         "keyword_positive": refusal_proxy(response)})
    manifest = {
        "study": "lfm3-reversible-alpha-control-smoke-v1",
        "checkpoint_sha256": EXPECTED_MODEL, "direction_sha256": EXPECTED_VECTOR,
        "pair_dataset_sha256": EXPECTED_PAIRS, "conditions": list(CONDITIONS),
        "holdout_ids": ["p03"], "layer": 10, "seed": 17, "device": "cpu",
        "tool_execution": False, "max_new_tokens": 24,
    }
    (tmp_path / "private_manifest.json").write_text(json.dumps(manifest))
    return rows, manifest
def test_divergence_detects_text_changes_without_keyword_flip(tmp_path):
    rows, _ = fixtures(tmp_path)
    for record in rows:
        if record["condition"] == "alpha_0_50" and record["category"] == "benign":
            record["response"] = "Alternative harmless phrasing."
            record["keyword_positive"] = False
    assert changes_against_baseline(rows, ["p03"])["benign"]["alpha_0_50"] == 1
    (tmp_path / "private_generations.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in rows))
    report = audit_folder(tmp_path)
    assert report["n_generations"] == 12
    assert report["exact_response_changed_vs_baseline"]["benign"]["alpha_0_50"] == 1
    assert report["keyword_positive_counts"]["benign"]["alpha_0_50"] == 0
    assert report["human_safety_reviewed"] is False
    assert "Alternative harmless phrasing" not in json.dumps(report)


def test_reject_invalid_manifest_or_incomplete_journal(tmp_path):
    rows, manifest = fixtures(tmp_path)
    (tmp_path / "private_generations.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in rows[:-1]))
    with pytest.raises(ValueError):
        audit_folder(tmp_path)
    manifest["seed"] = 999
    (tmp_path / "private_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol"):
        audit_folder(tmp_path)
