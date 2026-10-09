"""Bounded, split-aware read-only K2 probe collector on a pinned source cohort.

Only runs *existing* real-direction observation CLI. Every direction evaluated.
Never trains a classifier, modifies weights, dispatches production, or touches
NAS. Each run gets a unique x1 receiver HMAC receipt ledger. The receiver's
HMAC key remains on x1; local source HMAC key remains on the producer.
Outputs and per-run logs stay mode 600 on local SSD. Requires a separate x1
post-collection receipt/event/label join before data can be evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

from .k2_cohort_preflight import SCHEMA

COLLECTOR_SCHEMA = "auto-finetune.dust-k2-bounded-cohort-collection.v1"
ORDERED_SPLITS = ("train", "validation", "test")


def select_source_indices(manifest, quotas):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unrecognized cohort preflight")
    if manifest.get("max_tokens") != 128:
        raise ValueError("unexpected cohort tokenizer cap")
    if len(manifest.get("candidates", [])) > 256:
        raise ValueError("candidate count exceeded")
    if set(quotas) != set(ORDERED_SPLITS):
        raise ValueError("quotas must name all fixed partitions")
    if any(type(v) is not int or v < 0 or v > 4 for v in quotas.values()):
        raise ValueError("quota exceeds bounded 4 per split")
    if not 1 <= sum(quotas.values()) <= 8:
        raise ValueError("total cohort must be between 1 and 8 source groups")
    selected = []
    used_clusters = set()
    for split in ORDERED_SPLITS:
        if quotas[split] == 0:
            continue
        eligible = sorted((
            row for row in manifest["candidates"]
            if row.get("group_split") == split
            and row.get("disposition") == "ELIGIBLE"
        ), key=lambda row: row["sample_index"])
        added = 0
        for row in eligible:
            cluster = row.get("near_duplicate_cluster_sha256")
            if not (isinstance(cluster, str) and len(cluster) == 64):
                raise ValueError("missing cluster identity")
            if cluster in used_clusters:
                continue
            selected.append({
                "sample_index": row["sample_index"],
                "split": split,
                "episode_hmac_sha256": row["episode_hmac_sha256"],
                "near_duplicate_cluster_sha256": cluster,
            })
            used_clusters.add(cluster)
            added += 1
            if added == quotas[split]:
                break
        if added != quotas[split]:
            raise ValueError("not enough unique eligible " + split + " clusters")
    if len({r["near_duplicate_cluster_sha256"] for r in selected}) != len(selected):
        raise AssertionError("cluster-level independence violated")
    return selected


def run_cohort(*, manifest_path: Path, manifest_sha256: str,
               model_dir: Path, train_file: Path, key_file: Path,
               out_dir: Path, quotas: dict[str, int], expected_model_sha: str,
               python_executable: str, per_run_timeout: int = 90,
               population: int = 8):
    if population != 8:
        raise ValueError("first cohort limited to 8 complete directions/episode")
    if not 30 <= per_run_timeout <= 120:
        raise ValueError("bounded per-run timeout must be 30..120 seconds")
    if (not out_dir.is_dir() or out_dir.is_symlink()
            or out_dir.stat().st_mode & 0o077):
        raise PermissionError("cohort output directory must be private mode 700")
    blob = manifest_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest_sha256:
        raise ValueError("preflight digest mismatch")
    manifest = json.loads(blob)
    plan = select_source_indices(manifest, quotas)
    if len(expected_model_sha) != 64:
        raise ValueError("pinned expected model SHA is required")
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    results = []
    for item in plan:
        receiver_id = secrets.token_hex(16)
        base = (f"cohort128-{item['split']}-idx{item['sample_index']}-"
                f"{receiver_id}")
        event = out_dir / (base + ".events.jsonl")
        derived = out_dir / (base + ".derived.jsonl")
        summary = out_dir / (base + ".summary.json")
        local_log = out_dir / (base + ".stderr.txt")
        if any(p.exists() or p.is_symlink() for p in
               (event, derived, summary, local_log)):
            raise FileExistsError("refuse to overwrite earlier episode")
        command = [
            python_executable, "-m", "experiments.dust.k2_real_direction_probe",
            "--observe-only", "--model-dir", str(model_dir),
            "--train-jsonl", str(train_file),
            "--expected-sha256", expected_model_sha,
            "--episode-key-file", str(key_file),
            "--events", str(event), "--derived", str(derived),
            "--output", str(summary), "--device", "cuda", "--seed", "42",
            "--sample-index", str(item["sample_index"]),
            "--population", str(population), "--sigma", "0.25",
            "--direction-batch", "4", "--max-tokens", "128",
            "--receiver-run-id", receiver_id,
            "--preflight-manifest", str(manifest_path),
            "--expected-preflight-sha256", manifest_sha256,
        ]
        with local_log.open("x", encoding="utf-8") as err_stream:
            proc = subprocess.run(command, env=env, stdout=subprocess.DEVNULL,
                                  stderr=err_stream, check=False,
                                  timeout=per_run_timeout)
        if proc.returncode != 0:
            # Do not expose raw model/dataset text in an error message.
            raise RuntimeError("cohort probe failed; local restricted log "
                               f"retained for index {item['sample_index']}")
        reported = json.loads(summary.read_text())
        if (
            reported["population"] != population
            or reported["derived_evidence"]["rows"] != population
            or reported["derived_evidence"]["receiver_precommit_receipts"] != 2
            or reported["source_group_schema"] != "masked-prompt-prefix-v2"
            or reported["preflight_manifest_sha256"] != manifest_sha256
            or reported["source_episode_hmac_sha256"]
                != item["episode_hmac_sha256"]
            or reported["preflight_group_split"] != item["split"]
            or reported["optimizer_updates"] != 0
            or not reported["base_weights_unchanged"]
            or not reported["adapter_weights_unchanged"]
        ):
            raise AssertionError("real witness or source-group gate failed")
        results.append({
            **item,
            "receiver_run_id": receiver_id,
            "event_path": str(event),
            "derived_path": str(derived),
            "summary_path": str(summary),
            "events_sha256": hashlib.sha256(event.read_bytes()).hexdigest(),
            "derived_sha256": hashlib.sha256(derived.read_bytes()).hexdigest(),
            "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "source_row_count": population,
            "receiver_hmac_chain_joined_independently": False,
        })
    return {
        "schema": COLLECTOR_SCHEMA,
        "mode": "OBSERVATION_ONLY__AWAITING_INDEPENDENT_RECEIVER_JOIN",
        "preflight_sha256": manifest_sha256,
        "source_episode_count": len(results),
        "direction_rows": sum(r["source_row_count"] for r in results),
        "unique_clusters": len({r["near_duplicate_cluster_sha256"]
                                for r in results}),
        "split_counts": {
            s: sum(1 for r in results if r["split"] == s)
            for s in ORDERED_SPLITS},
        "source_samples": results,
        "classifier_training_authorized": False,
        "independent_receiver_join_verified": False,
        "model_or_adapter_updated": False,
        "hosted_provider_calls": 0,
        "nas_writes": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collect-readonly", action="store_true")
    p.add_argument("--preflight-manifest", type=Path, required=True)
    p.add_argument("--preflight-sha256", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--episode-key-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-model-sha", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--train-groups", type=int, default=4)
    p.add_argument("--validation-groups", type=int, default=2)
    p.add_argument("--test-groups", type=int, default=2)
    args = p.parse_args(argv)
    if not args.collect_readonly:
        p.error("explicit --collect-readonly required")
    os.umask(0o077)
    result = run_cohort(
        manifest_path=args.preflight_manifest,
        manifest_sha256=args.preflight_sha256,
        model_dir=args.model_dir, train_file=args.train_jsonl,
        key_file=args.episode_key_file, out_dir=args.output_dir,
        expected_model_sha=args.expected_model_sha,
        python_executable=args.python,
        quotas={
            "train": args.train_groups,
            "validation": args.validation_groups,
            "test": args.test_groups,
        })
    summary = args.output_dir / "cohort-collection-summary-v1.json"
    with summary.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    # Only aggregate statistics to stdout; no prompt-derived group HMACs.
    print(json.dumps({
        "source_episode_count": result["source_episode_count"],
        "direction_rows": result["direction_rows"],
        "unique_clusters": result["unique_clusters"],
        "split_counts": result["split_counts"],
        "independent_receiver_join_verified": False,
        "classifier_training_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
