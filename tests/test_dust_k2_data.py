"""Synthetic-only corpus-disjointness and causal response-masking tests."""
import json

import pytest

from experiments.dust.k2_data import extract_pairs, select_disjoint, tokenize_pair


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        user, assistant = messages
        assert user["role"] == "user"
        assert assistant["role"] == "assistant"
        # Synthetic token encoding, with explicit assistant header and end token.
        ids = ([201] + [ord(c) for c in user["content"]] + [202, 203]
               + [ord(c) for c in assistant["content"]] + [204])
        return {"input_ids": ids}


def test_finds_every_adjacent_plain_pair_without_tool_call():
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer", "tool_calls": [{"x": 1}]},
        {"role": "user", "content": "Third question"},
        {"role": "assistant", "content": "Third answer"},
    ]
    pairs = list(extract_pairs({"messages": messages}))
    assert len(pairs) == 2
    assert pairs[0] == ("First question", "First answer")


def test_assistant_only_loss_mask_preserves_shift():
    sample = tokenize_pair(FakeTokenizer(), ("SENSITIVE PROMPT", "private response"), 100)
    assert sample is not None
    assert sample["labels"][:len("SENSITIVE PROMPT")+3] == (
        [-100] * (len("SENSITIVE PROMPT")+3))
    assert sample["assistant_tokens"] == len("private response") + 1
    assert sample["labels"][-1] == 204


def test_exact_prompt_leakage_block_and_no_text_in_manifest(tmp_path):
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    def row(q, a):
        return json.dumps({"messages": [{"role": "user", "content": q},
                                        {"role": "assistant", "content": a}]}) + "\n"
    train.write_text(
        row("SENSITIVE SAME PROMPT", "Answer first")
        + row("OTHER UNIQUE PROMPT 1", "Answer second")
        + row("OTHER UNIQUE PROMPT 2", "Answer third"))
    heldout.write_text(
        row("  sensitive same prompt  ", "Leaks same prompt")
        + row("NEW HELDOUT PROMPT 1", "A totally separate answer")
        + row("NEW HELDOUT PROMPT 2", "Another separate answer"))
    tr, ev, manifest = select_disjoint(train, heldout, FakeTokenizer(),
                                      train_count=2, eval_count=2,
                                      max_tokens=128, eval_max_tokens=128)
    assert len(tr) == len(ev) == 2
    assert manifest["excluded_eval_prompts_shared_with_train_source"] == 1
    assert set(manifest["train_selected_pair_sha256"]).isdisjoint(
        manifest["eval_selected_pair_sha256"])
    assert "SENSITIVE" not in json.dumps(manifest)
    with pytest.raises(ValueError):
        select_disjoint(train, train, FakeTokenizer())


def test_eval_requires_unique_normalized_prompts(tmp_path):
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    def row(q, a):
        return json.dumps({"messages": [{"role": "user", "content": q},
                                        {"role": "assistant", "content": a}]}) + "\n"
    train.write_text(
        row("TRAIN UNIQUE PROMPT 1", "Training answer one")
        + row("TRAIN UNIQUE PROMPT 2", "Training answer two"))
    heldout.write_text(
        row("REPEATED HELDOUT PROMPT", "First heldout answer")
        + row(" repeated   heldout prompt ", "Second heldout answer")
        + row("SECOND UNIQUE HELDOUT", "Third heldout answer"))
    tr, ev, manifest = select_disjoint(
        train, heldout, FakeTokenizer(),
        train_count=2, eval_count=2,
        max_tokens=128, eval_max_tokens=128)
    assert len(tr) == 2
    assert len(ev) == 2
    assert manifest["selection"] == "sha256-ranked-unique-normalized-prompt.v2"
    assert manifest["eval_unique_prompt_hashes"] == 2
    assert len(set(manifest["eval_selected_prompt_sha256"])) == 2
    assert manifest["excluded_eval_pairs_repeating_selected_prompt"] == 1
    assert "REPEATED HELDOUT PROMPT" not in json.dumps(manifest)


def test_eval_fails_closed_when_rows_do_not_supply_enough_unique_prompts(tmp_path):
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    def row(q, a):
        return json.dumps({"messages": [{"role": "user", "content": q},
                                        {"role": "assistant", "content": a}]}) + "\n"
    train.write_text(
        row("TRAIN UNIQUE PROMPT 1", "Training answer one")
        + row("TRAIN UNIQUE PROMPT 2", "Training answer two"))
    heldout.write_text(
        row("ONLY HELDOUT PROMPT", "First answer")
        + row(" only   heldout prompt ", "Second answer"))
    with pytest.raises(ValueError, match="Not enough disjoint"):
        select_disjoint(
            train, heldout, FakeTokenizer(),
            train_count=2, eval_count=2,
            max_tokens=128, eval_max_tokens=128)
