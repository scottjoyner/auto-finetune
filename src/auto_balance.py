"""Automatic training data balancing from bucket analysis.

Uses the bucket analysis from analyze.py to create balanced training
datasets by upsampling minority classes and downsampling majority classes.

Usage:
    python -m src.cli auto-balance [--cap=500] [--out=<dir>]
    python -m src.cli auto-balance-status
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.config import Config


# Bucket priority weights (higher = more important to keep)
BUCKET_WEIGHTS = {
    "file-edit": 2.0,
    "multi-file-refactor": 2.5,
    "shell": 1.5,
    "debug": 1.0,
    "code-search": 1.5,
    "data-analysis": 2.0,
    "docs": 1.0,
    "web-research": 0.8,
    "reasoning": 0.5,
    "mixed": 0.7,
}


def load_bucket_map(bucket_map_path: str) -> dict:
    """Load the bucket map from analyze.py output."""
    with open(bucket_map_path) as f:
        return json.load(f)


def load_sessions(cleaned_dir: str) -> dict[str, dict]:
    """Load all sessions indexed by session_id."""
    sessions = {}
    for path in Path(cleaned_dir).rglob("*.json"):
        try:
            rec = json.loads(path.read_text())
            sid = rec.get("session_id", "")
            if sid:
                sessions[sid] = rec
        except Exception:
            continue
    return sessions


def compute_balance_stats(bucket_map: dict) -> dict:
    """Compute statistics for balancing."""
    bucket_counts = Counter()
    source_counts = Counter()
    difficulty_counts = Counter()
    keep_counts = Counter()

    for sid, meta in bucket_map.items():
        bucket = meta.get("bucket", "mixed")
        source = meta.get("source", "unknown")
        difficulty = meta.get("difficulty", "medium")
        keep = meta.get("keep", True)

        bucket_counts[bucket] += 1
        source_counts[source] += 1
        difficulty_counts[difficulty] += 1
        if keep:
            keep_counts["kept"] += 1
        else:
            keep_counts["dropped"] += 1

    return {
        "total": len(bucket_map),
        "by_bucket": dict(bucket_counts.most_common()),
        "by_source": dict(source_counts.most_common()),
        "by_difficulty": dict(difficulty_counts),
        "kept": keep_counts.get("kept", 0),
        "dropped": keep_counts.get("dropped", 0),
    }


def balance_buckets(
    bucket_map: dict,
    target_size: int = 500,
    weights: dict | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Balance sessions across buckets using weighted sampling.

    Args:
        bucket_map: session_id -> {bucket, source, ...}
        target_size: target size per bucket (or total if using weights)
        weights: optional bucket weights (default: BUCKET_WEIGHTS)
        seed: random seed for reproducibility

    Returns:
        dict of bucket -> list of session_ids to include
    """
    if weights is None:
        weights = BUCKET_WEIGHTS

    random.seed(seed)

    # Group sessions by bucket
    bucket_sessions: dict[str, list[str]] = defaultdict(list)
    for sid, meta in bucket_map.items():
        if not meta.get("keep", True):
            continue
        bucket = meta.get("bucket", "mixed")
        bucket_sessions[bucket].append(sid)

    if not bucket_sessions:
        return {}

    # Compute weighted sizes
    balanced: dict[str, list[str]] = {}
    total_weight = sum(weights.get(b, 1.0) for b in bucket_sessions)

    for bucket, sessions in bucket_sessions.items():
        weight = weights.get(bucket, 1.0)
        # Weighted proportion of target_size
        bucket_target = int(target_size * weight / total_weight)
        # Minimum of available sessions
        bucket_target = min(bucket_target, len(sessions))

        # Sample
        if bucket_target >= len(sessions):
            balanced[bucket] = sessions.copy()
        else:
            balanced[bucket] = random.sample(sessions, bucket_target)

    return balanced


def write_balanced_dataset(
    balanced: dict[str, list[str]],
    sessions: dict[str, dict],
    output_path: str,
) -> dict:
    """Write balanced dataset to JSONL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stats = {"total": 0, "by_bucket": {}}

    with open(output_path, "w") as f:
        for bucket, sids in sorted(balanced.items()):
            for sid in sids:
                rec = sessions.get(sid)
                if rec:
                    f.write(json.dumps(rec) + "\n")
                    stats["total"] += 1
                    stats["by_bucket"][bucket] = stats["by_bucket"].get(bucket, 0) + 1

    return stats


def main(cfg: Config, label: str | None = None, cap: int = 500,
         out_dir: str | None = None, seed: int = 42) -> int:
    """Auto-balance training data from bucket analysis."""
    analysis_dir = cfg.path("analysis_dir")
    cleaned_dir = cfg.path("cleaned_dir")
    dataset_dir = cfg.path("dataset_dir")

    # Load bucket map
    bucket_map_path = os.path.join(analysis_dir, "buckets.json")
    if not os.path.exists(bucket_map_path):
        print(f"[auto-balance] bucket map not found: {bucket_map_path}")
        print("[auto-balance] run `analyze` first")
        return 1

    bucket_map = load_bucket_map(bucket_map_path)
    print(f"[auto-balance] loaded {len(bucket_map)} sessions")

    # Compute stats
    stats = compute_balance_stats(bucket_map)
    print(f"[auto-balance] buckets: {stats['by_bucket']}")

    # Balance
    balanced = balance_buckets(bucket_map, target_size=cap, seed=seed)
    total_sampled = sum(len(sids) for sids in balanced.values())
    print(f"[auto-balance] sampled {total_sampled} sessions across {len(balanced)} buckets")
    for bucket, sids in sorted(balanced.items(), key=lambda x: -len(x[1])):
        print(f"  {bucket}: {len(sids)}")

    # Load sessions
    sessions = load_sessions(cleaned_dir)

    # Write output
    if out_dir is None:
        out_dir = os.path.join(analysis_dir, "balanced")

    output_path = os.path.join(out_dir, "train.balanced.jsonl")
    write_stats = write_balanced_dataset(balanced, sessions, output_path)

    print(f"[auto-balance] wrote {write_stats['total']} sessions to {output_path}")

    # Write metadata
    meta_path = os.path.join(out_dir, "balance-meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "cap": cap,
            "seed": seed,
            "source_stats": stats,
            "balanced_stats": write_stats,
            "bucket_weights": BUCKET_WEIGHTS,
        }, f, indent=2)

    print(f"[auto-balance] metadata written to {meta_path}")
    return 0
