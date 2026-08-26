"""Blend tool-call corpus with general instruct data.

Showdown v2 (2026-08-25) showed pure tool-dialect SFT regresses general task
completion: stock LFM2.5-Instruct passed 21% of the exec suite while the
finetune dropped to 0% — correct dialect, no convergence. Mixing general
instruct data at a target ratio preserves task competence alongside dialect.

Output is deterministic for a given seed so runs stay comparable.
"""
from __future__ import annotations

import json
import random


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mix_corpus(
    tool_path: str,
    out_path: str,
    general_paths: list[str],
    general_ratio: float = 0.35,
    seed: int = 42,
) -> tuple[int, int]:
    """Write tool+general blend to out_path. Returns (n_tool, n_general)."""
    if not 0 < general_ratio < 1:
        raise ValueError(f"general_ratio must be in (0,1), got {general_ratio}")

    tool = _load_jsonl(tool_path)
    general: list[dict] = []
    for p in general_paths:
        general.extend(_load_jsonl(p))

    # n_general such that general/(tool+general) == ratio
    n_target = int(len(tool) * general_ratio / (1 - general_ratio))
    rng = random.Random(seed)
    sampled = general if n_target >= len(general) else rng.sample(general, n_target)

    blended = tool + sampled
    rng.shuffle(blended)
    with open(out_path, "w") as f:
        for row in blended:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[mix] {len(tool)} tool + {len(sampled)} general "
          f"(ratio {len(sampled)/max(len(blended),1):.0%}) -> {out_path}")
    return len(tool), len(sampled)
