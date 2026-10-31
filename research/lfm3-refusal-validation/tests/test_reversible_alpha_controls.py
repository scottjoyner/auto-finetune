"""Synthetic-only tests: never inspect or transmit real private completions."""
import json
import pytest
import torch

from src.eval_contracts import refusal_proxy
from src.reversible_alpha_controls import (
    CONDITIONS, holdouts, random_orthogonal_unit, read_journal, summary,
    write_exclusive,
)


def synthetic_records():
    records = []
    for category in ("benign", "adversarial"):
        for condition in CONDITIONS:
            response = ("I cannot assist." if category == "adversarial"
                        else "I can help.")
            records.append({"pair": "p03", "category": category,
                            "condition": condition, "response": response,
                            "keyword_positive": refusal_proxy(response)})
    return records


def test_holdouts_require_original_frozen_eight():
    rows = [{"id": f"p{i:02d}"} for i in range(1, 9)]
    assert [r["id"] for r in holdouts(rows, 2)] == ["p03", "p04"]
    for invalid in (0, 7, True):
        with pytest.raises(ValueError):
            holdouts(rows, invalid)
    with pytest.raises(ValueError):
        holdouts(rows[:7], 1)
def test_random_control_deterministic_and_orthogonal():
    unit = torch.zeros(2048, dtype=torch.float32)
    unit[0] = 1
    a = random_orthogonal_unit(unit)
    b = random_orthogonal_unit(unit)
    assert torch.equal(a, b)
    assert torch.dot(a, unit).abs().item() < 1e-5
    assert abs(a.norm().item() - 1) < 1e-6
    with pytest.raises(ValueError):
        random_orthogonal_unit(torch.zeros(2048))
    with pytest.raises(ValueError):
        random_orthogonal_unit(torch.ones(2047))


def test_complete_synthetic_summary_contains_no_raw_text():
    manifest = {"checkpoint_sha256": "a", "direction_sha256": "b",
                "pair_dataset_sha256": "c", "layer": 10,
                "seed": 17, "max_new_tokens": 24}
    d = summary(synthetic_records(), ["p03"], manifest, "journal_hash")
    assert d["n_generations"] == 12
    assert d["baseline_vs_alpha_zero_exact_match"] is True
    assert d["behavioral_safety_reviewed"] is False
    assert d["keyword_positive_counts"]["adversarial"]["alpha_1_00"] == 1
    assert "I cannot assist" not in json.dumps(d)
    assert "p03" not in json.dumps(d)
def test_alpha_zero_parity_failure_blocks_claim():
    manifest = {"checkpoint_sha256": "a", "direction_sha256": "b",
                "pair_dataset_sha256": "c", "layer": 10,
                "seed": 17, "max_new_tokens": 24}
    r = synthetic_records()
    r[1]["response"] = "different generation"
    with pytest.raises(ValueError, match="parity"):
        summary(r, ["p03"], manifest, "hash")


def test_private_journal_fails_closed(tmp_path):
    journal = tmp_path / "private_generations.jsonl"
    records = synthetic_records()
    journal.write_text("".join(json.dumps(x)+"\n" for x in records))
    assert len(read_journal(journal, {"p03"})) == 12
    journal.write_text(json.dumps(records[0])+"\n"+json.dumps(records[0])+"\n")
    with pytest.raises(ValueError):
        read_journal(journal, {"p03"})
    journal.write_text(json.dumps(records[0]))
    with pytest.raises(ValueError, match="unterminated"):
        read_journal(journal, {"p03"})


def test_never_replace_aggregate(tmp_path):
    target = tmp_path / "report.json"
    write_exclusive(target, {"n": 1})
    with pytest.raises(FileExistsError):
        write_exclusive(target, {"n": 2})
    assert json.loads(target.read_text()) == {"n": 1}
