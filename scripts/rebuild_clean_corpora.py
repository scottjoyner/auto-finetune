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

import glob
import json
import os
import sys

sys.path.insert(0, "/home/scott/git/auto-finetune")

from src.analyze import benchmark_session_ids
from src.audit import _load_jsonl, audit_leakage, decontaminate
from src.config import load
from src.eval import build_disjoint_partition
from src.format_dataset import (combine, main as format_main, read_provenance,
                                verify_dataset, write_provenance)
from src.mixcorpus import mix_corpus

STAGING = "/media/scott/data/finetune-staging"
REBUILD = os.path.join(STAGING, "data", "clean-rebuild")
SUITE = "eval/tasks/auto-verified.jsonl"


def log(msg: str) -> None:
    print(f"[rebuild] {msg}", flush=True)


def decontaminate_all(ds_dir: str) -> dict[str, int]:
    """Drop rows that carry benchmark text verbatim, across every dataset.

    Session holdout and content leakage are different problems: holding out the
    49 mined sessions does nothing about a dev session that quotes one of them.
    Rows are rewritten in place, keeping provenance aligned by blanking the
    dropped rows' session ids.
    """
    bench = _load_jsonl(SUITE)
    out: dict[str, int] = {}
    for f in sorted(glob.glob(os.path.join(ds_dir, "train*.jsonl"))):
        if f.endswith(".provenance.jsonl") or os.path.getsize(f) == 0:
            continue
        rows = _load_jsonl(f)
        kept, dropped, res = decontaminate(rows, bench)
        # Check `dropped`, not `res["status"]`: decontaminate returns the audit
        # of the *kept* rows, which reads "clean" precisely when it dropped
        # something. Testing status first silently discards the removed rows --
        # which is exactly what happened on the first run of this script.
        if not dropped:
            if res["status"] != "clean":
                log(f"  WARNING {os.path.basename(f)}: {res['status']} but "
                    "nothing droppable (rows keyed by task_id, not position)")
            out[os.path.basename(f)] = 0
            continue
        tmp = f + ".tmp"
        with open(tmp, "w") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, f)
        # Keep the sidecar row-aligned: blank the ids of dropped rows by
        # rebuilding it from what survives, which decontaminate preserves order for.
        sids = read_provenance(f)
        if sids is not None and len(sids) == len(rows):
            kept_idx = [i for i in range(len(rows)) if i not in set(dropped)]
            write_provenance(f, [sids[i] for i in kept_idx])
        out[os.path.basename(f)] = len(dropped)
    return out


def main() -> int:
    held = benchmark_session_ids(SUITE)
    if not held:
        log("FATAL: no benchmark sessions recovered; refusing to build")
        return 2
    log(f"holding out {len(held)} benchmark sessions")
    bench_rows = _load_jsonl(SUITE)

    src_dir = os.path.join(STAGING, "data", "cleaned")
    ds_dir = os.path.join(REBUILD, "datasets")
    os.makedirs(ds_dir, exist_ok=True)

    cfg = load("config.yaml")
    cfg.raw["paths"]["cleaned_dir"] = src_dir
    cfg.raw["paths"]["dataset_dir"] = ds_dir

    # 1. per-source + merged datasets, benchmark sessions held out
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

    # 4. content-level decontamination.
    #    Step 1-3 only remove the 49 sessions the tasks were mined from. Other
    #    sessions still quote benchmark text -- dev sessions about building the
    #    benchmark itself. Verified: after the session holdout, 37 bench tasks
    #    were still present in the blend, and a provenance check reads "clean"
    #    because it only asks whether those 49 session ids appear.
    log("decontaminating (content-level) ...")
    decontam = decontaminate_all(ds_dir)
    for name, n_dropped in sorted(decontam.items()):
        if n_dropped:
            log(f"  {name}: dropped {n_dropped} row(s) carrying bench text")

    # 5. disjoint eval partition from the clean blend
    log("eval-split ...")
    part = build_disjoint_partition(mixed, os.path.join(REBUILD, "future-runs"),
                                    frac=0.1, seed=42, label="mixed",
                                    held_out=held)

    # 6. verify with BOTH checks. The content-level audit is the stricter one
    #    and is what must pass; provenance is the weaker structural check.
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
        a = audit_leakage(_load_jsonl(path), bench_rows)
        ok = v["status"] == "clean" and a["status"] == "clean"
        log(f"  {name:22} prov={v['status']:6} content={a['status']:6} "
            f"hits={a['n_hits']}  rows={v['n_rows']}")
        if not ok:
            bad.append((name, path, v["status"], a["status"]))

    # every emitted dataset, not just the headline ones
    log("verifying all datasets (content-level) ...")
    for f in sorted(glob.glob(os.path.join(ds_dir, "train*.jsonl"))):
        if f.endswith(".provenance.jsonl") or os.path.getsize(f) == 0:
            continue
        a = audit_leakage(_load_jsonl(f), bench_rows)
        if a["status"] != "clean":
            log(f"  DIRTY {os.path.basename(f)}: {a['n_hits']} bench hit(s)")
            bad.append((os.path.basename(f), f, "?", a["status"]))

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
        for entry in bad:
            log(f"FAIL {entry[0]}: {' '.join(str(x) for x in entry[2:])} ({entry[1]})")
        return 3
    log("all artifacts clean")
    log(f"manifest -> {os.path.join(REBUILD, 'rebuild-manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())