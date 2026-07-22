"""Content-level leakage audit between a training mix and held-out benchmarks.

The corpus pipeline already excludes benchmark *sessions* from the train
mix (``analyze.benchmark_session_ids`` + ``strata --holdout``), so the
same session never trains and validates. But a *different* session can
still contain a near-identical instruction, file path, or command — a
soft leak that inflates the benchmark. This module checks for that by
comparing the normalized instruction text of every benchmark task against
every training example.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict


def _norm(text: str) -> str:
    """Lowercase, keep alphanumerics + spaces, collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _train_text(ex: dict) -> str:
    """Flatten a train-mix example (messages / conversations / instruction)."""
    parts: list[str] = []
    if ex.get("messages"):
        for m in ex["messages"]:
            parts.append(m.get("content", ""))
    elif ex.get("conversations"):
        for m in ex["conversations"]:
            parts.append(m.get("value", ""))
    elif ex.get("instruction") is not None:
        parts.append(ex["instruction"])
        if ex.get("output") is not None:
            parts.append(ex["output"])
    return _norm(" ".join(parts))


def _bench_instruction(ex: dict) -> str:
    return _norm(ex.get("instruction") or ex.get("prompt") or "")


def audit_leakage(train_rows: list[dict], bench_rows: list[dict],
                  min_len: int = 12) -> dict:
    """Return content-overlap hits between train mix and benchmark tasks.

    A hit is recorded when a benchmark instruction (normalized) appears as
    a substring of a training example's flattened text (and is long enough
    to be meaningful). Near-duplicate detection, not exact-session match.
    """
    train_texts = [(_train_text(r), r.get("task_id") or i)
                   for i, r in enumerate(train_rows)]
    hits: list[dict] = []
    per_bench: dict[str, int] = defaultdict(int)
    for b in bench_rows:
        bi = _bench_instruction(b)
        if len(bi) < min_len:
            continue
        for ttext, tid in train_texts:
            if bi and bi in ttext:
                hits.append({
                    "bench_task_id": b.get("task_id"),
                    "train_ref": tid,
                    "instruction": b.get("instruction"),
                })
                per_bench[b.get("task_id")] = per_bench.get(b.get("task_id"), 0) + 1
                break
    return {
        "n_train": len(train_rows),
        "n_bench": len(bench_rows),
        "n_hits": len(hits),
        "hit_rate": round(len(per_bench) / len(bench_rows), 3) if bench_rows else 0.0,
        "hits": hits,
    }


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(cfg, argv: list[str]) -> int:  # type: ignore[no-untyped-def]
    from src.cli import _parse_str_flag

    train_path = (_parse_str_flag(argv, "--train")
                  or os.path.join(cfg.path("analysis_dir"), "train.balanced.jsonl"))
    bench_path = (_parse_str_flag(argv, "--bench")
                  or os.path.join("eval", "tasks", "auto-verified.jsonl"))
    res = audit_leakage(_load_jsonl(train_path), _load_jsonl(bench_path))
    print(f"[audit] train={res['n_train']} bench={res['n_bench']} "
          f"hits={res['n_hits']} hit_rate={res['hit_rate']}")
    if res["hits"]:
        for h in res["hits"][:20]:
            print(f"  LEAK {h['bench_task_id']} <-> train#{h['train_ref']}: "
                  f"{str(h['instruction'])[:80]}")
    else:
        print("[audit] no content-level leakage detected")
    return 0
