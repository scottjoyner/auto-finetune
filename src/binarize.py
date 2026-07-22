"""Binarize a training mix to a pre-tokenized Arrow dataset.

The focused ``train`` step (src/train.py) normally renders each example
through ``apply_chat_template`` and lets TRL's SFTTrainer tokenize the
result every epoch. For a 10k mix on a single iGPU that re-tokenization
is wasted CPU. ``binarize`` pre-computes ``input_ids`` /
``attention_mask`` / ``labels`` with the real base tokenizer once, so the
GPU step just loads the Arrow shards (``datasets.load_from_disk``) and
trains.

The tokenization matches what SFTTrainer would do for a ``text`` field:
render the chat template to text, then tokenize the whole sequence with
full-sequence supervision (labels == input_ids), consistent with the
existing pipeline.
"""
from __future__ import annotations

import os
from typing import Any

from src.config import Config
from src.train import _to_messages, load_dataset


def tokenize_example(ex: dict, tokenizer, max_seq: int) -> dict[str, list[int]]:
    """Render one example to token ids. Labels == input_ids (full-supervision)."""
    msgs = _to_messages(ex)
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(text, truncation=True, max_length=max_seq)
    ids: list[int] = enc["input_ids"]
    return {
        "input_ids": ids,
        "attention_mask": enc["attention_mask"],
        "labels": list(ids),
    }


def binarize(rows: list[dict], tokenizer, max_seq: int) -> "Any":
    """Tokenize every row and return a ``datasets.Dataset`` (in-memory)."""
    from datasets import Dataset

    feats: dict[str, list[Any]] = {}
    for ex in rows:
        tokd = tokenize_example(ex, tokenizer, max_seq)
        for k, v in tokd.items():
            feats.setdefault(k, []).append(v)
    if not feats:
        raise ValueError("no features produced (empty rows?)")
    return Dataset.from_dict(feats)


def binarize_mix(src_jsonl: str, out_dir: str, model_name: str,
                 max_seq: int) -> int:
    """Load JSONL rows, tokenize with the real model tokenizer, save to Arrow."""
    from transformers import AutoTokenizer

    if not os.path.exists(src_jsonl):
        raise FileNotFoundError(f"source mix not found: {src_jsonl}")
    rows = load_dataset(src_jsonl)
    if not rows:
        raise ValueError(f"source mix is empty: {src_jsonl}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = binarize(rows, tokenizer, max_seq)
    os.makedirs(out_dir, exist_ok=True)
    ds.save_to_disk(out_dir)
    return len(ds)


def load_tokenized(dir_path: str) -> "Any":
    """Load a pre-tokenized Arrow dataset produced by :func:`binarize_mix`."""
    from datasets import load_from_disk

    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"tokenized dataset not found: {dir_path}")
    return load_from_disk(dir_path)


def _default_mix_path(cfg: Config) -> str:
    return os.path.join(cfg.path("analysis_dir"), "train.balanced.jsonl")


def _default_arrow_path(cfg: Config) -> str:
    return os.path.join(cfg.path("analysis_dir"), "train.balanced.arrow")


def main(cfg: Config, argv: list[str]) -> int:
    from src.cli import _parse_str_flag

    src = _parse_str_flag(argv, "--src") or _default_mix_path(cfg)
    out = _parse_str_flag(argv, "--out") or _default_arrow_path(cfg)
    model = _parse_str_flag(argv, "--model") or cfg.get(
        "train", "model_name", default="Qwen/Qwen2.5-7B-Instruct"
    )
    max_seq = int(_parse_str_flag(argv, "--max-seq")
                  or cfg.get("train", "max_seq_length", default=8192))
    n = binarize_mix(src, out, model, max_seq)
    print(f"[binarize] wrote {n} pre-tokenized rows -> {out}")
    return 0
