#!/usr/bin/env python3
"""Rebuild the training corpora with the benchmark holdout enforced.

Why a separate tree: `cli format` acquires a lease on the live `datasets/`
dir, which the running trainer (pid 9571) holds. Writing there would either be
refused or clobber the dataset mid-run. This builds into
`data/clean-rebuild/` instead, so the live corpus is untouched and the clean
one can be diffed before switching over.

Every emit path applies `exclude` (the 49 recovered benchmark sessions) and
writes a row-aligned provenance sidecar, then each artifact is verified before
being declared good.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/home/scott/git/auto-finetune")

from src.analyze import benchmark_session_ids
from src.config import Config, load
from src.eval import build_disjoint_partition
from src.format_dataset import combine, main as format_main, verify_dataset
from src.mixcorpus import mix_corpus

STAGING = "/media/scott/data/finetune-staging"
REBUILD = os.path.join(STAGING, "data", "clean-rebuild")
SUITE = "eval/tasks/auto-verified.jsonl"


def log(msg: str) -> None:
    print(f"[rebuild] {msg}", flush=True)


def main() -> int:
    held = benchmark_session_ids(SUITE)
    if not held:
        log("FATAL: no benchmark sessions recovered; refusing to build")
        return 2
    log(f"holding out {len(held)} benchmark sessions")

    src_dir = os.path.join(STAGING, "data", "cleaned")
    ds_dir = os.path.join(REBUILD, "datasets")
    os.makedirs(ds_dir, exist_ok=True)

    cfg = load("config.yaml")
    cfg.raw["paths"]["cleaned_dir"] = src_dir
    cfg.raw["paths"]["dataset_dir"] = ds_dir

    # 1. per-source + merged datasets, benchmark held out
    log("format --all-split ...")
    total = 0
    for source in ("hermes", "opencode", None):
        total += format_main(cfg, source=source, exclude=held)
    log(f"format wrote {total} examples")

    # 2. deduped union across the labels combine knows about
    log("combine ...")
    n_combined = combine(cfg, exclude=held)

    # 3. tool + general blend (the shape the qwen/ornith runs train on)
    general = os.path.join(STAGING, "data", "datasets", "general-norobots.jsonl")
    mixed = os.path.join(ds_dir, "train.mixed.jsonl")
    log("mixcorpus ...")
    mix_corpus(os.path.join(ds_dir, "train.combined.jsonl"), mixed, [general],
               general_ratio=0.35, seed=42, exclude=held)

    # 4. disjoint eval partition from the clean blend
    log("eval-split ...")
    part = build_disjoint_partition(mixed, os.path.join(REBUILD, "future-runs"),
                                    frac=0.1, seed=42, label="mixed",
                                    held_out=held)

    # 5. verify every artifact we are about to hand over
    log("verifying ...")
    checks = {
        "train.combined.jsonl": os.path.join(ds_dir, "train.combined.jsonl"),
        "train.mixed.jsonl": mixed,
        "partition train": part.train_path,
        "partition eval": part.eval_path,
    }
    bad = []
    for name, path in checks.items():
        v = verify_dataset(path, held)
        log(f"  {name:22} {v['status']:14} "
            f"{v['n_rows']:>7} rows / {v['n_sessions']:>6} sessions")
        if v["status"] != "clean":
            bad.append((name, path, v))

    manifest = {
        "held_out_sessions": len(held),
        "n_combined": n_combined,
        "n_mixed": sum(1 for _ in open(mixed)),
        "partition": {"train": str(part.train_path), "eval": str(part.eval_path),
                      "n_train": len(part.train_rows), "n_eval": len(part.eval_rows)},
        "checks": {name: verify_dataset(p, held)["status"] for name, p in checks.items()},
    }
    with open(os.path.join(REBUILD, "rebuild-manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    if bad:
        for name, path, v in bad:
            log(f"FAIL {name}: {v['status']} ({path})")
        return 3
    log("all artifacts clean")
    log(f"manifest -> {os.path.join(REBUILD, 'rebuild-manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())