"""Aggregate comparable K2 tail-replay training manifests without raw data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def load(path: Path) -> dict:
    obj=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj,dict):
        raise ValueError(f"{path}: manifest must be an object")
    if obj.get("schema")!="auto-finetune.dust-k2-tail-train.v1":
        raise ValueError(f"{path}: unexpected schema")
    return obj


def fingerprint(x: dict) -> dict:
    d=x["dataset"]
    return {
        "model_weights_sha256":x["model_weights_sha256"],
        "model_config_sha256":x["model_config_sha256"],
        "rank":x["rank"],"steps":x["steps"],
        "population":x["population"],"sigma":x["sigma"],
        "learning_rate":x["learning_rate"],
        "direction_batch":x["direction_batch"],
        "train_source_sha256":d["train_source"]["sha256"],
        "eval_source_sha256":d["eval_source"]["sha256"],
        "train_selected_pair_sha256":d["train_selected_pair_sha256"],
        "eval_selected_pair_sha256":d["eval_selected_pair_sha256"],
        "train_rows":d["train_rows"],"eval_rows":d["eval_rows"],
        "train_max_seq_tokens":d["train_max_seq_tokens"],
        "eval_max_seq_tokens":d["eval_max_seq_tokens"],
    }


def aggregate(values):
    return {
        "mean":mean(values),
        "median":median(values),
        "min":min(values),
        "max":max(values),
    }


def summarize(paths: list[Path]) -> dict:
    if len(paths)<2:
        raise ValueError("Need at least two manifests")
    rows=[load(p) for p in paths]
    fp=fingerprint(rows[0])
    for p,row in zip(paths[1:],rows[1:]):
        if fingerprint(row)!=fp:
            raise ValueError(f"{p}: manifest is not comparable")
    seeds=[int(r["seed"]) for r in rows]
    if len(set(seeds))!=len(seeds):
        raise ValueError("Duplicate seeds are not allowed")

    train=[r["train_ce_delta"] for r in rows]
    held=[r["heldout_ce_delta"] for r in rows]
    training=[r["training_seconds"] for r in rows]
    total=[r["total_seconds"] for r in rows]
    cache=[r["prefix_cache_seconds"] for r in rows]

    reference_serial=[]
    reference_backprop=[]
    for row in rows:
        ref=row.get("reference")
        if not ref:
            raise ValueError("Every multiseed row requires immutable reference evidence")
        reference_serial.append(ref["serial_structured"])
        reference_backprop.append(ref["backprop"])

    return {
        "schema":"auto-finetune.dust-k2-tail-multiseed-summary.v1",
        "research_only":True,
        "production_promotion_authorized":False,
        "fingerprint":fp,
        "seeds":sorted(seeds),
        "seed_count":len(seeds),
        "tail":{
            "train_ce_delta":aggregate(train),
            "heldout_ce_delta":aggregate(held),
            "train_improved_seeds":sum(v<0 for v in train),
            "heldout_improved_seeds":sum(v<0 for v in held),
            "training_seconds":aggregate(training),
            "total_seconds":aggregate(total),
            "prefix_cache_seconds":aggregate(cache),
        },
        "reference_serial":{
            "train_ce_delta":aggregate(
                [x["train_ce_delta"] for x in reference_serial]),
            "heldout_ce_delta":aggregate(
                [x["heldout_ce_delta"] for x in reference_serial]),
            "elapsed_seconds":aggregate(
                [x["elapsed_seconds"] for x in reference_serial]),
            "train_improved_seeds":sum(
                x["train_ce_delta"]<0 for x in reference_serial),
            "heldout_improved_seeds":sum(
                x["heldout_ce_delta"]<0 for x in reference_serial),
        },
        "reference_backprop":{
            "train_ce_delta":aggregate(
                [x["train_ce_delta"] for x in reference_backprop]),
            "heldout_ce_delta":aggregate(
                [x["heldout_ce_delta"] for x in reference_backprop]),
            "elapsed_seconds":aggregate(
                [x["elapsed_seconds"] for x in reference_backprop]),
            "train_improved_seeds":sum(
                x["train_ce_delta"]<0 for x in reference_backprop),
            "heldout_improved_seeds":sum(
                x["heldout_ce_delta"]<0 for x in reference_backprop),
        },
        "source_manifest_names":[p.name for p in paths],
        "raw_user_data_in_report":False,
        "checkpoint_written":False,
        "warning":(
            "Cross-hardware tail vs reference timing is descriptive only; "
            "quality deltas remain a tiny research sample."),
    }


def main(argv=None) -> int:
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
