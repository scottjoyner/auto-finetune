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
    explicit = ex.get("instruction") or ex.get("prompt")
    if explicit:
        return _norm(explicit)
    # Held-out training-format corpora use chat ``messages`` rather than the
    # mined-task instruction/prompt schema. Flatten the complete example so an
    # exact or contained training row cannot be mislabeled clean.
    return _train_text(ex)


def _bench_id(ex: dict, index: int) -> str | int:
    """Accept both mined-task (task_id/instruction) and suite (id/prompt)."""
    value = ex.get("task_id")
    if value is None:
        value = ex.get("id")
    return value if value is not None else index


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
    hit_bench: set[str | int] = set()
    eligible = 0
    for index, b in enumerate(bench_rows):
        bi = _bench_instruction(b)
        if len(bi) < min_len:
            continue
        eligible += 1
        bench_id = _bench_id(b, index)
        instruction = b.get("instruction") or b.get("prompt") or ""
        for ttext, tid in train_texts:
            if bi and bi in ttext:
                hits.append({
                    "bench_task_id": bench_id,
                    "train_ref": tid,
                    "instruction": instruction,
                })
                hit_bench.add(bench_id)
                break
    rate = len(hit_bench) / len(bench_rows) if bench_rows else 0.0
    status = ("not_evaluable" if bench_rows and eligible == 0 else
              "contaminated" if hits else "clean")
    return {
        "n_train": len(train_rows),
        "n_bench": len(bench_rows),
        "n_eligible": eligible,
        "n_hits": len(hits),
        "hit_rate": rate,
        "status": status,
        "hits": hits,
    }


def decontaminate(train_rows: list[dict], bench_rows: list[dict],
                  min_len: int = 12, max_rounds: int = 25
                  ) -> tuple[list[dict], list[int], dict]:
    """Drop training rows that contain a benchmark instruction verbatim.

    Session-level holdout (``benchmark_session_ids``) removes the sessions the
    tasks were mined from, but a *different* session can still quote them --
    in practice dev sessions about building the benchmark itself. Those rows
    are a small fraction of the mix, so dropping them is cheap; leaving them in
    would let the model train on eval text.

    This iterates to a fixpoint. ``audit_leakage`` reports only the first
    matching row per bench task, so one pass under-reports: dropping that row
    can expose another row carrying the same instruction. A single pass left 3
    of 6 hits behind on the ssd candidate.

    Only positional ``train_ref`` values are droppable. When a row carries a
    ``task_id`` the audit reports that instead of an index, and we refuse to
    guess -- we keep the contaminated rows and the caller fails closed.

    Returns ``(kept_rows, dropped_original_indices, final_audit_result)``.
    """
    kept = list(train_rows)
    # Original positional indices, so callers can trace what was removed.
    drop_original: set[int] = set()
    # Original index for each currently-kept row, for the same reason.
    origin = list(range(len(train_rows)))
    result = audit_leakage(kept, bench_rows, min_len=min_len)
    unresolvable: list = []
    if result["status"] != "contaminated":
        return train_rows, [], dict(result, n_dropped=0)

    for _ in range(max_rounds):
        if result["status"] != "contaminated":
            break
        drop: set[int] = set()
        for hit in result["hits"]:
            ref = hit["train_ref"]
            if isinstance(ref, int):
                drop.add(ref)
            else:
                unresolvable.append(ref)
        if unresolvable or not drop:
            # Either rows are identified by task_id, not position, or we made
            # no progress. Keep everything and let the caller fail closed.
            return train_rows, [], dict(result, n_dropped=0)
        drop_original.update(origin[i] for i in drop)
        kept = [r for i, r in enumerate(kept) if i not in drop]
        origin = [origin[i] for i in range(len(origin)) if i not in drop]
        result = audit_leakage(kept, bench_rows, min_len=min_len)

    if unresolvable:
        return train_rows, [], dict(result, n_dropped=0)
    # `result` is the audit of the KEPT rows, so it reads "clean" exactly when
    # rows were dropped. Callers that branch on status first will skip writing
    # the cleaned rows out, so surface the count in the result itself.
    return kept, sorted(drop_original), dict(result, n_dropped=len(drop_original))


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
