"""Synthetic privacy/diversity tests for K2 evaluation integrity audit."""
import json

from experiments.dust.k2_data import normalized
from experiments.dust.k2_eval_integrity import (
    audit_probe, audit_response_eval, audit_tasks, prompt_hash,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        user, assistant = messages
        ids = ([201] + [ord(c) for c in user["content"]] + [202, 203]
               + [ord(c) for c in assistant["content"]] + [204])
        return {"input_ids": ids}


def _row(q, a):
    return json.dumps({"messages": [{"role": "user", "content": q},
                                    {"role": "assistant", "content": a}]}) + "\n"


def test_response_audit_reports_prompt_diversity_without_text(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        _row("Repeated evaluation prompt", "Answer one")
        + _row(" repeated  evaluation prompt ", "Answer two")
        + _row("Fresh independent prompt", "Answer three"))
    train_prompts = {prompt_hash("Training prompt only")}
    report = audit_response_eval(path, train_prompts, FakeTokenizer(), 256)
    assert report["tokenizable_pairs"] == 3
    assert report["prompt_disjoint_pairs"] == 3
    assert report["prompt_disjoint_summary"]["unique_prompt_hashes"] == 2
    assert report["prompt_disjoint_summary"]["duplicate_prompt_rows"] == 1
    assert "Repeated evaluation prompt" not in json.dumps(report)


def test_probe_and_task_audit_detect_exact_train_overlap_without_raw_text(tmp_path):
    train_hash = prompt_hash("Same as training")
    probe = tmp_path / "probe.jsonl"
    probe.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Same as training"}]}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "Probe unique prompt"}]}) + "\n")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"id": "task-one", "prompt": "Task unique prompt"}) + "\n"
        + json.dumps({"id": "task-two", "prompt": " same as training "}) + "\n")
    p = audit_probe(probe, {train_hash})
    t = audit_tasks(tasks, {train_hash})
    assert p["rows"] == 2 and p["unique_prompt_hashes"] == 2
    assert p["prompt_overlap_with_train"] == 1
    assert t["rows"] == 2 and t["prompt_overlap_with_train"] == 1
    rendered = json.dumps({"probe": p, "tasks": t})
    assert "Same as training" not in rendered
    assert "Task unique prompt" not in rendered


def test_prompt_hash_normalizes_case_and_whitespace():
    assert prompt_hash("  Hello   WORLD ") == prompt_hash("hello world")
    assert normalized("  Hello   WORLD ") == "hello world"
