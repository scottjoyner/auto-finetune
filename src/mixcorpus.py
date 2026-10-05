"""Blend tool-call corpus with general instruct data.

Showdown v2 (2026-08-25) showed pure tool-dialect SFT regresses general task
completion: stock LFM2.5-Instruct passed 21% of the exec suite while the
finetune dropped to 0% — correct dialect, no convergence. Mixing general
instruct data at a target ratio preserves task competence alongside dialect.

Output is deterministic for a given seed so runs stay comparable.
"""
from __future__ import annotations

import json
import os
import random


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _provenance_for(path: str, n_rows: int) -> list[str]:
    """Session id per row for a source dataset; blanks when unavailable.

    A dataset emitted before provenance existed still mixes fine -- it just
    cannot be audited for held-out benchmark sessions afterwards.
    """
    from src.format_dataset import read_provenance

    sids = read_provenance(path)
    if sids is None:
        return [""] * n_rows
    if len(sids) != n_rows:
        # Drift means the sidecar does not describe this file; refuse to
        # attach a mapping that would be wrong for some rows.
        return [""] * n_rows
    return sids


def mix_corpus(
    tool_path: str,
    out_path: str,
    general_paths: list[str],
    general_ratio: float = 0.35,
    seed: int = 42,
    exclude: set[str] | None = None,
) -> tuple[int, int]:
    """Write tool+general blend to out_path. Returns (n_tool, n_general).

    ``exclude`` holds out benchmark session ids. Provenance is carried through
    the sampling and shuffle so the blend stays auditable.
    """
    if not 0 < general_ratio < 1:
        raise ValueError(f"general_ratio must be in (0,1), got {general_ratio}")

    # Provenance has to survive the shuffle below, so rows travel as
    # (row, session_id) pairs rather than bare dicts.
    tool = _load_jsonl(tool_path)
    tool_sids = _provenance_for(tool_path, len(tool))
    general: list[dict] = []
    general_sids: list[str] = []
    for p in general_paths:
        rows = _load_jsonl(p)
        general.extend(rows)
        general_sids.extend(_provenance_for(p, len(rows)))

    # n_general such that general/(tool+general) == ratio
    n_target = int(len(tool) * general_ratio / (1 - general_ratio))
    rng = random.Random(seed)
    if n_target >= len(general):
        picked = list(zip(general, general_sids))
    else:
        idx = sorted(rng.sample(range(len(general)), n_target))
        picked = [(general[i], general_sids[i]) for i in idx]

    blended = list(zip(tool, tool_sids)) + picked
    if exclude:
        held = sum(1 for _, sid in blended if sid and sid in exclude)
        blended = [(r, s) for r, s in blended if not (s and s in exclude)]
        if held:
            print(f"[mix] held out {held} benchmark session(s)")
    rng.shuffle(blended)

    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        for row, _sid in blended:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    from src.format_dataset import write_provenance
    write_provenance(out_path, [sid for _, sid in blended])

    print(f"[mix] {len(tool)} tool + {len(picked)} general "
          f"(ratio {len(picked)/max(len(blended),1):.0%}) -> {out_path}")
    return len(tool), len(picked)
