"""Aggregate comparable K2 structured-training JSON manifests without raw data."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


def load(path: Path) -> dict:
    obj=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj,dict):
        raise ValueError(f"{path}: manifest is not an object")
    return obj


def fingerprint(x: dict) -> dict:
    d=x["dataset"]
    return {
        "schema": x.get("schema"),
        "model_weights_sha256": x.get("model_weights_sha256"),
        "model_config_sha256": x.get("model_config_sha256"),
        "rank": x.get("rank"),
        "steps": x.get("steps"),
        "population": x.get("population"),
        "sigma": x.get("sigma"),
        "learning_rate": x.get("learning_rate"),
        "train_source_sha256": d["train_source"]["sha256"],
        "eval_source_sha256": d["eval_source"]["sha256"],
        "train_selected_pair_sha256": d["train_selected_pair_sha256"],
        "eval_selected_pair_sha256": d["eval_selected_pair_sha256"],
        "train_rows": d["train_rows"],
        "eval_rows": d["eval_rows"],
        "train_max_seq_tokens": d["train_max_seq_tokens"],
        "eval_max_seq_tokens": d["eval_max_seq_tokens"],
    }


def summarize(paths: list[Path]) -> dict:
    if len(paths) < 2:
        raise ValueError("Need at least two seed manifests")
    rows=[load(p) for p in paths]
    base=fingerprint(rows[0])
    for p,row in zip(paths[1:],rows[1:]):
        if fingerprint(row) != base:
            raise ValueError(f"{p}: manifest is not comparable to first run")
    seeds=[int(r["seed"]) for r in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not allowed")
    def metrics(name: str):
        vals=[r[name] for r in rows]
        train=[v["train_ce_delta"] for v in vals]
        held=[v["heldout_ce_delta"] for v in vals]
        secs=[v["elapsed_seconds"] for v in vals]
        return {
            "mean_train_ce_delta": mean(train),
            "median_train_ce_delta": median(train),
            "train_improved_seeds": sum(v < 0 for v in train),
            "mean_heldout_ce_delta": mean(held),
            "median_heldout_ce_delta": median(held),
            "heldout_improved_seeds": sum(v < 0 for v in held),
            "mean_elapsed_seconds": mean(secs),
            "median_elapsed_seconds": median(secs),
        }
    a=[r["adapter_update_alignment"]["a"]["cosine"] for r in rows]
    b=[r["adapter_update_alignment"]["b"]["cosine"] for r in rows]
    out={
        "schema":"auto-finetune.dust-k2-structured-multiseed-summary.v1",
        "research_only":True,
        "production_promotion_authorized":False,
        "comparable_fingerprint":base,
        "seeds":sorted(seeds),
        "seed_count":len(seeds),
        "backprop":metrics("backprop"),
        "structured":metrics("structured"),
        "adapter_update_alignment":{
            "a_cosine_mean":mean(a),"a_cosine_min":min(a),
            "b_cosine_mean":mean(b),"b_cosine_min":min(b),
        },
        "structured_to_backprop_mean_time_ratio":(
            metrics("structured")["mean_elapsed_seconds"] /
            metrics("backprop")["mean_elapsed_seconds"]
        ),
        "source_manifest_names":[p.name for p in paths],
        "raw_user_data_in_report":False,
        "checkpoint_written":False,
        "warning":"Small research matrix; no model-quality or production-readiness claim.",
    }
    return out


def main(argv: list[str] | None=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifests",nargs="+",type=Path)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args(argv)
    if args.output is not None and args.output.exists():
        ap.error("Refusing to overwrite existing summary")
    report=summarize(args.manifests)
    rendered=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if args.output is None:
        print(rendered,end="")
    else:
        with args.output.open("x",encoding="utf-8") as f:
            f.write(rendered)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
