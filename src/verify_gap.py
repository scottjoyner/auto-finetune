"""Diagnose the benchmark tasks the static verifier cannot satisfy.

``verify`` replays recorded file writes and runs ``file_contains`` checks;
it deliberately executes no shell/code/web tool and materializes only
files under the sandbox root. That leaves a hard ceiling (the 61.3%
pass-rate) of tasks whose checks it can never satisfy. Before extending
the verifier we need to know *why* each of the failing tasks fails.

``categorize`` reads a ``verify-report.jsonl`` (rows from
``src.verify.verify_task``) and buckets every non-passing task by the
reason its checks fail:

  * ``no_source``        — source session missing from cleaned/
  * ``unsupported_kind`` — a check kind the harness doesn't implement
  * ``file_not_materialized`` — expected file wasn't replayed into the
                           sandbox (usually an absolute/remote path)
  * ``snippet_missing``  — file present but the expected snippet absent
  * ``other``            — anything else

This scopes exactly which verifier extension (new check kinds, sandbox
path-remap, etc.) would push the pass-rate past the ceiling.
"""
from __future__ import annotations

import json
import os
from collections import Counter


def _category(result: dict) -> str:
    if result.get("reason") == "source session not found":
        return "no_source"
    checks = result.get("checks", [])
    if not checks:
        return "other"
    details = [c.get("detail", "") for c in checks if not c.get("ok")]
    if any(d.startswith("unsupported") for d in details):
        return "unsupported_kind"
    if any(d.startswith("file not found") for d in details):
        return "file_not_materialized"
    if any(d.startswith("snippet not in") for d in details):
        return "snippet_missing"
    return "other"


def categorize(results: list[dict]) -> dict:
    counts: Counter = Counter()
    per_task: list[dict] = []
    for r in results:
        if r.get("ok"):
            continue
        cat = _category(r)
        counts[cat] += 1
        per_task.append({
            "task_id": r.get("task_id"),
            "category": cat,
            "bucket": r.get("bucket"),
            "reason": r.get("reason"),
        })
    return {
        "n_total": len(results),
        "n_passing": sum(1 for r in results if r.get("ok")),
        "n_failing": len(per_task),
        "counts": dict(counts),
        "tasks": per_task,
    }


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"not found: {path}")
    return [json.loads(l) for l in open(path) if l.strip()]


def main(cfg, argv: list[str]) -> int:  # type: ignore[no-untyped-def]
    from src.cli import _parse_str_flag

    report = (_parse_str_flag(argv, "--report")
               or os.path.join(cfg.path("analysis_dir"), "verify-report.jsonl"))
    res = categorize(_load_jsonl(report))
    print(f"[verify-gap] total={res['n_total']} passing={res['n_passing']} "
          f"failing={res['n_failing']}")
    for cat, n in sorted(res["counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<24} {n}")
    return 0
