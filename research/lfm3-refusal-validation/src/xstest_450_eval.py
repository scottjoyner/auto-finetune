"""Pinned, resumable XSTest evaluation for exactly three local LFM3 checkpoints.

The benchmark prompt texts and raw model responses stay private.
Published aggregates are exploratory until blinded manual grading.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.eval_contracts import refusal_proxy
from src.research_job_guard import require_research_write_capacity
from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.refusal_probe import load_local_model, generate

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = Path("/home/scott/lfm3-research/refusal-checkpoints/xstest-benchmark/xstest_prompts.csv")
SOURCE_SHA256 = "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a"
SOURCE_COMMIT = "d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d"
SOURCE_URL = ("https://raw.githubusercontent.com/paul-rottger/xstest/"
              + SOURCE_COMMIT + "/xstest_prompts.csv")
CATEGORIES = 18
ARM_ORDER = tuple(ARMS)


def read_verified_dataset(path=SOURCE_FILE):
    path = Path(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != SOURCE_SHA256:
        raise ValueError("XSTest source hash mismatch; refuse unpinned corpus")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 450:
        raise ValueError("XSTest must contain 450 rows")
    counts = Counter("unsafe" if x["type"].startswith("contrast_") else "safe"
                     for x in rows)
    if counts != {"safe": 250, "unsafe": 200}:
        raise ValueError("unexpected XSTest label balance")
    if len(set(x["id"] for x in rows)) != len(rows):
        raise ValueError("XSTest duplicate ids")
    if len(set(x["type"] for x in rows)) != CATEGORIES:
        raise ValueError("unexpected number of XSTest categories")
    for row in rows:
        row["class"] = "unsafe" if row["type"].startswith("contrast_") else "safe"
    return rows


def stratified_selection(rows, per_category, seed):
    """per_category=0 means all 450; otherwise seeded reproducible selection."""
    if per_category < 0:
        raise ValueError("per_category cannot be negative")
    groups = defaultdict(list)
    for row in rows:
        groups[row["type"]].append(row)
    rng = random.Random(seed)
    selected = []
    for name in sorted(groups):
        group = groups[name]
        if per_category == 0:
            chosen = group
        else:
            if per_category > len(group):
                raise ValueError("per_category larger than benchmark category")
            chosen = rng.sample(group, per_category)
        selected.extend(chosen)
    return selected


def manifest_for(rows, hashes, args):
    return {
        "study": "XSTest-three-checkpoint-v0",
        "source": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "checkpoint_sha256": hashes,
        "checkpoint_order": list(ARM_ORDER),
        "selection_ids": [x["id"] for x in rows],
        "category_counts": dict(Counter(x["type"] for x in rows)),
        "per_category": args.per_category,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "prompt_format": "tokenizer chat template, including pretrained base",
        "precision": "bf16",
        "device": "cpu",
        "real_actions_permitted": False,
        "human_safety_review": "PENDING",
    }


def verify_resume(selected, completed):
    # Prevent stale, duplicated, or out-of-scope records contaminating a run.
    known = set()
    selected_ids = set(x["id"] for x in selected)
    for item in completed:
        key = (item["arm"], item["id"])
        if item["arm"] not in ARM_ORDER or item["id"] not in selected_ids:
            raise ValueError("unknown model arm or XSTest case in existing output")
        if key in known:
            raise ValueError("duplicate resumed generation")
        known.add(key)
    return known


def record_summary(rows, selections):
    ids = {r["id"]:r for r in selections}
    groups = {}
    for arm in ARM_ORDER:
        by_class = {}
        for cls in ("safe", "unsafe"):
            matched = [r for r in rows if r["arm"] == arm and
                       ids[r["id"]]["class"] == cls]
            n = len(matched)
            by_class[cls] = {
                "complete_generations": n,
                "lexical_refusal_count": sum(bool(r["refusal_proxy"]) for r in matched),
                "lexical_refusal_rate": (sum(bool(r["refusal_proxy"]) for r in matched) / n if n else None),
            }
        groups[arm] = by_class
    return groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-category", type=int, default=1,
                   help="0 = all 450 prompts; 1..25 = stratified smoke samples")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results" / "xstest")
    args = p.parse_args()
    if args.max_new_tokens < 8 or args.max_new_tokens > 384:
        p.error("--max-new-tokens must be in 8..384")
    require_research_write_capacity(args.results_dir)
    os.umask(0o077)
    dataset = read_verified_dataset()
    selected = stratified_selection(dataset, args.per_category, args.seed)
    hashes = checked_arm_paths(ARMS)
    manifest = manifest_for(selected, hashes, args)
    folder = args.results_dir
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    manifest_file = folder / "private_run_manifest.json"
    output_file = folder / "private_generations.jsonl"
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        if previous != manifest:
            raise ValueError("existing manifest differs; use a separate --results-dir")
    elif output_file.exists():
        raise ValueError("private generations present without trusted manifest")
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
        os.chmod(manifest_file, 0o600)
    completed = []
    if output_file.exists():
        with output_file.open() as f:
            completed = [json.loads(x) for x in f if x.strip()]
    keys = verify_resume(selected, completed)
    import gc
    import torch
    for arm in ARM_ORDER:
        outstanding = [r for r in selected if (arm, r["id"]) not in keys]
        if not outstanding:
            continue
        require_research_write_capacity(folder)
        print(f"Loading {arm}; remaining={len(outstanding)}", flush=True)
        torch.manual_seed(args.seed)
        model, tokenizer = load_local_model(ARMS[arm])
        for i, row in enumerate(outstanding, 1):
            started = time.monotonic()
            text = generate(model, tokenizer, row["prompt"], args.max_new_tokens)
            result = {
                "arm": arm, "id": row["id"], "type": row["type"],
                "class": row["class"], "response": text,
                "refusal_proxy": refusal_proxy(text),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            require_research_write_capacity(folder)
            with output_file.open("a", encoding="utf-8") as fd:
                fd.write(json.dumps(result, ensure_ascii=False) + "\n")
                fd.flush()
                os.fsync(fd.fileno())
            os.chmod(output_file, 0o600)
            completed.append(result)
            if i == 1 or i % 5 == 0 or i == len(outstanding):
                print(f"{arm} {i}/{len(outstanding)} completed", flush=True)
        del model, tokenizer
        gc.collect()
    complete = len(completed) == len(selected) * len(ARM_ORDER)
    aggregate = {
        "experiment": "XSTest 450-prompt benchmark",
        "completion": "complete" if complete else "partial",
        "sampled_cases_per_arm": len(selected),
        "total_generations": len(completed),
        "dataset_sha256": SOURCE_SHA256,
        "category_counts": manifest["category_counts"],
        "arm_hashes": hashes,
        "model_results_proxy_only": record_summary(completed, selected),
        "manual_review": "PENDING",
        "limitations": [
            "No automated lexical proxy is a valid measure of malicious-action resistance",
            "Unreviewed model completions are private",
            "Base uses a chat template despite not being instruction-tuned",
            "Single-turn English-only XSTest cannot replace HarmBench and multi-turn agentic evaluations",
        ],
    }
    summary = folder / "safe_aggregate.json"
    summary.write_text(json.dumps(aggregate, indent=2) + "\n")
    os.chmod(summary, 0o600)
    print(json.dumps({k:v for k,v in aggregate.items() if k!="category_counts"},indent=2), flush=True)


if __name__ == "__main__":
    main()
