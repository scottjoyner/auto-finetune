"""Multi-source data fusion for real-time pipeline.

Combines data from multiple sources (neo4j-traces, opencode, hermes)
into unified training datasets with deduplication and quality filtering.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config, load
from src.locking import atomic_write_json


@dataclass
class FusionStats:
    """Statistics from a fusion run."""
    sources: list[str] = field(default_factory=list)
    input_counts: dict[str, int] = field(default_factory=dict)
    output_count: int = 0
    duplicates_removed: int = 0
    quality_filtered: int = 0
    fusion_time: float = 0.0


def load_source(source: str, dataset_dir: str) -> list[dict]:
    """Load training examples from a source."""
    # Try various filename patterns
    patterns = [
        f"train.{source}.jsonl",
        f"train_{source}.jsonl",
        f"{source}.jsonl",
    ]
    for pattern in patterns:
        path = os.path.join(dataset_dir, pattern)
        if os.path.exists(path):
            sessions = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            sessions.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            return sessions
    return []


def fuse_sources(sources: list[str], dataset_dir: str,
                 output_path: str, dedup: bool = True,
                 min_quality: float = 0.0) -> FusionStats:
    """Fuse multiple data sources into a single dataset.

    Args:
        sources: List of source names to fuse
        dataset_dir: Directory containing source datasets
        output_path: Output path for fused dataset
        dedup: Whether to deduplicate by content hash
        min_quality: Minimum quality score (0 = no filtering)
    """
    stats = FusionStats(sources=sources, fusion_time=time.time())
    all_examples = []
    seen_hashes = set()

    for source in sources:
        examples = load_source(source, dataset_dir)
        stats.input_counts[source] = len(examples)

        for ex in examples:
            # Dedup by content hash
            if dedup:
                content_hash = hash(json.dumps(ex, sort_keys=True))
                if content_hash in seen_hashes:
                    stats.duplicates_removed += 1
                    continue
                seen_hashes.add(content_hash)

            # Quality filter
            if min_quality > 0:
                # Simple quality check: must have messages with content
                msgs = ex.get("messages", [])
                if not msgs or len(msgs) < 2:
                    stats.quality_filtered += 1
                    continue

            all_examples.append(ex)

    stats.output_count = len(all_examples)
    stats.fusion_time = time.time() - stats.fusion_time

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    return stats


def freshness_weighted_fuse(sources: list[str], dataset_dir: str,
                            output_path: str, weights: dict[str, float] | None = None) -> FusionStats:
    """Fuse sources with freshness weighting.

    More recent data gets higher weight (more copies in output).
    """
    if weights is None:
        weights = {s: 1.0 for s in sources}

    stats = FusionStats(sources=sources, fusion_time=time.time())
    all_examples = []
    seen_hashes = set()

    for source in sources:
        examples = load_source(source, dataset_dir)
        stats.input_counts[source] = len(examples)
        weight = weights.get(source, 1.0)
        copies = max(1, int(weight))

        for ex in examples:
            content_hash = hash(json.dumps(ex, sort_keys=True))
            if content_hash in seen_hashes:
                stats.duplicates_removed += 1
                continue
            seen_hashes.add(content_hash)

            for _ in range(copies):
                all_examples.append(ex)

    stats.output_count = len(all_examples)
    stats.fusion_time = time.time() - stats.fusion_time

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    return stats


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Multi-source data fusion")
    parser.add_argument("--sources", required=True, help="Comma-separated source names")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--dedup", action="store_true", default=True, help="Deduplicate")
    parser.add_argument("--min-quality", type=float, default=0.0, help="Min quality score")
    parser.add_argument("--weighted", action="store_true", help="Use freshness weighting")
    args = parser.parse_args(argv)

    cfg = load()
    sources = [s.strip() for s in args.sources.split(",")]
    dataset_dir = cfg.path("dataset_dir")

    if args.weighted:
        stats = freshness_weighted_fuse(sources, dataset_dir, args.output)
    else:
        stats = fuse_sources(sources, dataset_dir, args.output,
                             dedup=args.dedup, min_quality=args.min_quality)

    print(f"[fusion] Sources: {sources}")
    print(f"[fusion] Input counts: {stats.input_counts}")
    print(f"[fusion] Duplicates removed: {stats.duplicates_removed}")
    print(f"[fusion] Quality filtered: {stats.quality_filtered}")
    print(f"[fusion] Output: {stats.output_count} examples -> {args.output}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
