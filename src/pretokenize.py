"""Pre-tokenize dataset to Arrow format for faster training.

CPU-only heavy lifting that tokenizes the entire dataset once, then
memory-maps it during training to avoid re-tokenization each epoch.

Usage:
    python -m src.cli pretokenize [--label=<name>] [--model=<path>] [--max-length=2048]
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from src.config import Config


def _load_tokenizer(model_name: str):
    """Load tokenizer from transformers."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _format_conversation(rec: dict) -> str:
    """Format a session record into a single text string for tokenization."""
    messages = []
    for msg in rec.get("messages", []):
        role = msg.get("role", "assistant")
        content_parts = []

        for p in msg.get("parts", []):
            ptype = p.get("type")
            if ptype == "text":
                text = p.get("text", "").strip()
                if text:
                    content_parts.append(text)
            elif ptype == "tool":
                tool_name = p.get("tool", "unknown")
                inp = p.get("input", {})
                out = p.get("output", "")

                # Format tool call
                if isinstance(inp, dict):
                    args_str = json.dumps(inp, indent=2)
                else:
                    args_str = str(inp)

                if out:
                    content_parts.append(
                        f"[Tool: {tool_name}]\nInput: {args_str}\nOutput: {out[:500]}"
                    )
                else:
                    content_parts.append(f"[Tool: {tool_name}]\nInput: {args_str}")
            elif ptype == "patch":
                patch = p.get("patch", "")
                if patch:
                    content_parts.append(f"[Patch]\n{patch[:1000]}")

        if content_parts:
            messages.append({
                "role": role,
                "content": "\n".join(content_parts)
            })

    # Simple chat format
    formatted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted.append(f"{'<|user|>' if role == 'user' else '<|assistant|>'}\n{content}")
    formatted.append("<|assistant|>\n")

    return "\n".join(formatted)


def tokenize_dataset(
    sessions: list[dict],
    tokenizer,
    max_length: int = 2048,
    output_dir: str = ".",
) -> dict:
    """Tokenize sessions and save to Arrow format.

    Args:
        sessions: List of session records
        tokenizer: HuggingFace tokenizer
        max_length: Maximum token length per example
        output_dir: Directory to save Arrow files

    Returns:
        Statistics about the tokenization
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(output_dir, exist_ok=True)

    # Tokenize all sessions
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    metadata = []

    skipped = 0
    truncated = 0
    total_tokens = 0

    start_time = time.time()

    for idx, rec in enumerate(sessions):
        text = _format_conversation(rec)

        # Tokenize
        encoded = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if len(input_ids) < 10:
            skipped += 1
            continue

        if len(input_ids) == max_length:
            truncated += 1

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(input_ids.copy())
        total_tokens += len(input_ids)

        metadata.append({
            "session_id": rec.get("session_id", ""),
            "source": rec.get("source", ""),
            "n_tokens": len(input_ids),
        })

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"  tokenized {idx + 1}/{len(sessions)} ({rate:.1f} sessions/sec)")

    # Pad to same length for Arrow storage
    max_len = max(len(ids) for ids in input_ids_list) if input_ids_list else 0
    padded_inputs = []
    padded_masks = []
    padded_labels = []

    for ids, mask, labs in zip(input_ids_list, attention_mask_list, labels_list):
        pad_len = max_len - len(ids)
        padded_inputs.append(ids + [tokenizer.pad_token_id] * pad_len)
        padded_masks.append(mask + [0] * pad_len)
        padded_labels.append(labs + [-100] * pad_len)

    # Create Arrow table
    table = pa.table({
        "input_ids": padded_inputs,
        "attention_mask": padded_masks,
        "labels": padded_labels,
        "session_id": [m["session_id"] for m in metadata],
        "source": [m["source"] for m in metadata],
        "n_tokens": [m["n_tokens"] for m in metadata],
    })

    # Write Parquet (Arrow-based format)
    parquet_path = os.path.join(output_dir, "tokenized.parquet")
    pq.write_table(table, parquet_path)

    elapsed = time.time() - start_time

    return {
        "n_sessions": len(sessions),
        "n_tokenized": len(input_ids_list),
        "n_skipped": skipped,
        "n_truncated": truncated,
        "total_tokens": total_tokens,
        "avg_tokens": total_tokens // max(1, len(input_ids_list)),
        "max_tokens": max_len,
        "elapsed_seconds": round(elapsed, 1),
        "sessions_per_second": round(len(sessions) / elapsed, 1),
        "parquet_path": parquet_path,
    }


def main(
    cfg: Config,
    label: str | None = None,
    model: str | None = None,
    max_length: int = 2048,
) -> int:
    """Pre-tokenize dataset to Arrow format."""
    from src.clean import _dedup_by_session

    cleaned_dir = cfg.path("cleaned_dir")

    # Use config or default model
    if model is None:
        model = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")

    print(f"[pretokenize] loading tokenizer: {model}")
    tokenizer = _load_tokenizer(model)

    # Load sessions
    sessions = []
    if label:
        src = os.path.join(cleaned_dir, label)
        if os.path.isdir(src):
            for fn in sorted(os.listdir(src)):
                if fn.endswith(".json"):
                    try:
                        sessions.append(json.loads(Path(os.path.join(src, fn)).read_text()))
                    except Exception:
                        continue
    else:
        for path in sorted(Path(cleaned_dir).rglob("*.json")):
            try:
                sessions.append(json.loads(path.read_text()))
            except Exception:
                continue

    if not sessions:
        print("[pretokenize] no sessions found")
        return 0

    # Deduplicate
    sessions = list(_dedup_by_session(sessions).values())
    print(f"[pretokenize] {len(sessions)} sessions, max_length={max_length}")

    # Output directory
    dataset_dir = cfg.path("dataset_dir")
    if label:
        out_dir = os.path.join(dataset_dir, f"pretokenized-{label}")
    else:
        out_dir = os.path.join(dataset_dir, "pretokenized")

    # Run tokenization
    stats = tokenize_dataset(sessions, tokenizer, max_length, out_dir)

    # Save stats
    stats_path = os.path.join(out_dir, "tokenization-stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"[pretokenize] tokenized {stats['n_tokenized']} sessions")
    print(f"[pretokenize] {stats['total_tokens']} total tokens, "
          f"{stats['avg_tokens']} avg per session")
    print(f"[pretokenize] {stats['n_truncated']} truncated to {max_length}")
    print(f"[pretokenize] {stats['n_skipped']} skipped (too short)")
    print(f"[pretokenize] {stats['elapsed_seconds']}s ({stats['sessions_per_second']} sessions/sec)")
    print(f"[pretokenize] output: {stats['parquet_path']}")

    return 0
