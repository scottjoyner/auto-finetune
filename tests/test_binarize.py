"""CPU tests for src/binarize.py (pre-tokenized Arrow mix).

Uses a deterministic mock tokenizer so no model download is needed; the
real run later uses the Qwen/RefinedToolCallV5 tokenizer via AutoTokenizer.
"""
from pathlib import Path

from src.binarize import binarize, load_tokenized, tokenize_example


class _MockTok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        if tokenize:  # pragma: no cover - not used by binarize
            raise NotImplementedError
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    def __call__(self, text, truncation=False, max_length=8192):
        ids = [(ord(c) % 97) + 1 for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _rows():
    return [{
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
    }]


def test_tokenize_example_full_supervision():
    ex = _rows()[0]
    tokd = tokenize_example(ex, _MockTok(), 8192)
    assert set(tokd) == {"input_ids", "attention_mask", "labels"}
    assert tokd["labels"] == tokd["input_ids"]          # full-sequence supervision
    assert len(tokd["attention_mask"]) == len(tokd["input_ids"])


def test_binarize_and_load_roundtrip(tmp_path: Path):
    ds = binarize(_rows(), _MockTok(), 8192)
    assert len(ds) == 1
    assert "input_ids" in ds.column_names

    out = tmp_path / "arrow"
    ds.save_to_disk(str(out))
    loaded = load_tokenized(str(out))
    assert len(loaded) == 1
    assert loaded[0]["input_ids"] == ds[0]["input_ids"]
