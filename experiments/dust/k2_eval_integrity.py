"""Audit K2 evaluation prompt diversity/leakage without emitting raw text."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.dust.k2_data import normalized, read_pairs, tokenize_pair


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def summarize_hashes(hashes: list[str]) -> dict:
    counts: dict[str, int] = {}
    for value in hashes:
        counts[value] = counts.get(value, 0) + 1
    return {
        "rows": len(hashes),
        "unique_prompt_hashes": len(counts),
        "duplicate_prompt_rows": sum(n - 1 for n in counts.values() if n > 1),
        "max_rows_per_prompt": max(counts.values(), default=0),
        "prompt_sha256": sorted(counts),
    }


def audit_response_eval(path: Path, train_prompts: set[str], tokenizer,
                        max_tokens: int) -> dict:
    entries, stats = read_pairs(path)
    tokenizable = []
    for digest, (user_hash, pair) in sorted(entries.items()):
        try:
            row = tokenize_pair(tokenizer, pair, max_tokens)
        except (ValueError, TypeError, KeyError, AttributeError):
            row = None
        if row is not None:
            tokenizable.append((digest, user_hash, row))
    hashes = [h for _, h, _ in tokenizable]
    disjoint = [(d, h, r) for d, h, r in tokenizable if h not in train_prompts]
    disjoint_hashes = [h for _, h, _ in disjoint]
    return {
        "path_name": path.name,
        "source": stats,
        "tokenizable_pairs": len(tokenizable),
        "tokenizable_prompt_summary": summarize_hashes(hashes),
        "prompt_overlap_with_train_pairs": sum(h in train_prompts for h in hashes),
        "prompt_disjoint_pairs": len(disjoint),
        "prompt_disjoint_summary": summarize_hashes(disjoint_hashes),
        "prompt_disjoint_assistant_tokens": sum(r["assistant_tokens"] for _, _, r in disjoint),
        "raw_text_emitted": False,
    }


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def audit_probe(path: Path, train_prompts: set[str]) -> dict:
    hashes = []
    invalid = 0
    for row in _read_jsonl(path):
        messages = row.get("messages")
        if not isinstance(messages, list):
            invalid += 1
            continue
        prompts = [
            m.get("content") for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
            and isinstance(m.get("content"), str) and m.get("content").strip()
        ]
        if not prompts:
            invalid += 1
            continue
        hashes.append(prompt_hash(prompts[-1]))
    return {
        "path_name": path.name,
        **summarize_hashes(hashes),
        "prompt_overlap_with_train": sum(h in train_prompts for h in hashes),
        "prompt_disjoint_rows": sum(h not in train_prompts for h in hashes),
        "invalid_rows": invalid,
        "raw_text_emitted": False,
    }


def audit_tasks(path: Path, train_prompts: set[str]) -> dict:
    hashes = []
    invalid = 0
    ids = []
    for row in _read_jsonl(path):
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            invalid += 1
            continue
        hashes.append(prompt_hash(prompt))
        task_id = row.get("id")
        if isinstance(task_id, str):
            ids.append(hashlib.sha256(task_id.encode()).hexdigest())
    return {
        "path_name": path.name,
        **summarize_hashes(hashes),
        "prompt_overlap_with_train": sum(h in train_prompts for h in hashes),
        "prompt_disjoint_rows": sum(h not in train_prompts for h in hashes),
        "task_id_sha256": sorted(ids),
        "invalid_rows": invalid,
        "raw_text_emitted": False,
    }


def audit(*, train_file: Path, model_dir: Path, eval_files: list[Path],
          probe_files: list[Path], task_files: list[Path],
          max_tokens: int) -> dict:
    from transformers import AutoTokenizer

    train_entries, train_stats = read_pairs(train_file)
    train_prompts = {h for h, _ in train_entries.values()}
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    response_evals = [
        audit_response_eval(p, train_prompts, tokenizer, max_tokens)
        for p in eval_files
    ]
    curated_probes = [audit_probe(p, train_prompts) for p in probe_files]
    agentic_tasks = [audit_tasks(p, train_prompts) for p in task_files]
    independent_hashes = [
        value
        for result in [*curated_probes, *agentic_tasks]
        for value in result["prompt_sha256"]
    ]
    independent_disjoint = [
        value for value in independent_hashes if value not in train_prompts
    ]
    independent_union = {
        **summarize_hashes(independent_disjoint),
        "source_rows": sum(
            result["rows"] for result in [*curated_probes, *agentic_tasks]),
        "source_unique_prompt_entries": len(independent_hashes),
        "train_overlap_prompt_entries": sum(
            value in train_prompts for value in independent_hashes),
        "prompt_disjoint_unique_entries": len(independent_disjoint),
    }
    return {
        "schema": "auto-finetune.dust-k2-eval-integrity.v1",
        "research_only": True,
        "train_source": train_stats,
        "train_unique_prompt_hashes": len(train_prompts),
        "max_eval_tokens": max_tokens,
        "response_evals": response_evals,
        "curated_probes": curated_probes,
        "agentic_tasks": agentic_tasks,
        "independent_candidate_union": independent_union,
        "raw_text_emitted": False,
        "warning": (
            "Exact normalized-prompt audit only; semantic/near-duplicate "
            "contamination requires a separate review."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--eval-jsonl", type=Path, action="append", default=[])
    ap.add_argument("--probe-jsonl", type=Path, action="append", default=[])
    ap.add_argument("--task-jsonl", type=Path, action="append", default=[])
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args(argv)
    if not 16 <= args.max_tokens <= 2048:
        ap.error("max-tokens outside [16,2048]")
    if args.output is not None and args.output.exists():
        ap.error("Refusing to overwrite existing audit")
    report = audit(
        train_file=args.train_jsonl, model_dir=args.model_dir,
        eval_files=args.eval_jsonl, probe_files=args.probe_jsonl,
        task_files=args.task_jsonl, max_tokens=args.max_tokens)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
