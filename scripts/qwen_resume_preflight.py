#!/usr/bin/env python3
"""Read-only structural preflight. Not evidence of a successfully restored trainer."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import stat

STAGING = Path("/media/scott/data/finetune-staging")
REPO = Path("/home/scott/git/auto-finetune")
STOP = STAGING / "launch-next.stop"
LABEL = "qwen3-8b-sft-r1"
MANDATORY = (
    "adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
    "training_args.bin", "optimizer.pt", "scheduler.pt", "rng_state.pth",
    "tokenizer.json", "tokenizer_config.json",
)
NUMERIC = re.compile(r"checkpoint-([0-9]+)\Z")


def safe_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def safe_dir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def load_json(path: Path) -> dict | None:
    if not safe_file(path) or path.stat().st_size > 2 * 1024 * 1024:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def hash_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def train_invocations(source_path: Path) -> tuple[int, int]:
    """AST, not grep: detect trainer.train() calls and explicit resume args."""
    if not safe_file(source_path):
        return (0, 0)
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeError, OSError):
        return (0, 0)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (node.func.attr == "train"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "trainer"):
                calls.append(node)
    supports_resume = sum(
        any(kw.arg == "resume_from_checkpoint" for kw in call.keywords)
        or bool(call.args)
        for call in calls
    )
    return len(calls), supports_resume


def audit(staging: Path, repo: Path, *, hash_partition_source: bool = False) -> dict:
    reasons: list[str] = []
    labels = staging / "outputs/checkpoints" / LABEL
    if not safe_dir(labels):
        reasons.append("production_checkpoint_label_absent")
        ckpts = []
    else:
        ckpts = sorted(
            ((int(m.group(1)), d) for d in labels.iterdir()
             if (m := NUMERIC.fullmatch(d.name)) and safe_dir(d)),
            key=lambda a: a[0]
        )
    last = ckpts[-1][1] if ckpts else None
    if last is None:
        reasons.append("numeric_checkpoint_absent")
    checkpoint = {
        "step": ckpts[-1][0] if last is not None else None,
        "file_sizes": {},
        "missing": [],
        "global_step": None,
        "trainer_max_steps": None,
    }
    if last is not None:
        for name in MANDATORY:
            path = last / name
            if not safe_file(path):
                checkpoint["missing"].append(name)
            else:
                checkpoint["file_sizes"][name] = path.stat().st_size
                if path.stat().st_size == 0:
                    checkpoint["missing"].append(name)
        if checkpoint["missing"]:
            reasons.append("checkpoint_components_missing")
        trainer = load_json(last / "trainer_state.json")
        step = trainer.get("global_step") if trainer else None
        max_steps = trainer.get("max_steps") if trainer else None
        checkpoint["global_step"] = step
        checkpoint["trainer_max_steps"] = max_steps
        if not (type(step) is int and step == ckpts[-1][0]):
            reasons.append("trainer_state_step_mismatch")
        config = load_json(last / "adapter_config.json")
    else:
        config = None

    base_model = config.get("base_model_name_or_path") if config else None
    base = Path(base_model) if type(base_model) is str and base_model.startswith("/") else None
    model = {"base_name": base_model, "config_exists": False,
             "index_exists": False, "indexed_shards": 0,
             "present_shards": 0}
    if base and safe_dir(base):
        model["config_exists"] = safe_file(base / "config.json")
        model["index_exists"] = safe_file(base / "model.safetensors.index.json")
        index = load_json(base / "model.safetensors.index.json")
        names = set(index.get("weight_map", {}).values()) if index and isinstance(index.get("weight_map"), dict) else set()
        # Never accept external paths, traversal, symlinks or index-only evidence.
        names = {n for n in names if type(n) is str and
                 n.startswith("model-") and n.endswith(".safetensors")
                 and "/" not in n and "\\" not in n}
        model["indexed_shards"] = len(names)
        model["present_shards"] = sum(safe_file(base / name) and (base / name).stat().st_size > 0 for name in names)
    if not (model["config_exists"] and model["index_exists"]
            and model["indexed_shards"] > 0
            and model["present_shards"] == model["indexed_shards"]):
        reasons.append("base_model_structure_unverified")

    partition = load_json(staging / "data/future-runs/mixed-seed42/partition.json")
    clean = (isinstance(partition, dict)
             and isinstance(partition.get("contamination"), dict)
             and partition["contamination"].get("status") == "clean"
             and partition["contamination"].get("overlap_count") == 0)
    dataset = {"clean_partition_claim": bool(clean), "source_present": False,
               "train_present": False, "eval_present": False,
               "source_hash_matches": None, "manifest_source_hash": None,
               "source_bytes": 0, "train_bytes": 0, "eval_bytes": 0}
    if not clean:
        reasons.append("dataset_partition_not_clean")
    if partition:
        for key, display in (("source", "source"), ("train_path", "train"), ("eval_path", "eval")):
            value = partition.get(key)
            path = Path(value) if isinstance(value, str) and value.startswith("/") else None
            if path and safe_file(path):
                dataset[display + "_present"] = True
                dataset[display + "_bytes"] = path.stat().st_size
                if key == "source" and hash_partition_source:
                    expected = partition.get("source_sha256")
                    dataset["manifest_source_hash"] = expected
                    if isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected):
                        dataset["source_hash_matches"] = hash_file(path) == expected
    if not all(dataset[f + "_present"] for f in ("source", "train", "eval")):
        reasons.append("dataset_partition_files_missing")
    if hash_partition_source and dataset["source_hash_matches"] is not True:
        reasons.append("dataset_source_sha256_unverified")

    calls, supported = train_invocations(repo / "src/train.py")
    source = {"trainer_train_calls": calls, "calls_with_resume_argument": supported,
              "all_training_backends_support_resume": calls > 0 and calls == supported}
    if not source["all_training_backends_support_resume"]:
        reasons.append("training_code_has_no_resume_entrypoint")
    if not safe_file(staging / "launch-next.stop"):
        reasons.append("recovery_stop_file_missing")
    return {
        "status": "NO_GO", "may_resume": False, "may_train": False,
        "may_prune": False, "may_copy": False, "may_release_source": False,
        "restore_test_passed": False, "read_only": True,
        "stop_file_present": safe_file(staging / "launch-next.stop"),
        "checkpoint": checkpoint, "base_model": model, "dataset": dataset,
        "training_source": source, "reasons": sorted(set(reasons)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=STAGING)
    parser.add_argument("--repository", type=Path, default=REPO)
    parser.add_argument("--hash-partition-source", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.staging, args.repository,
                           hash_partition_source=args.hash_partition_source),
                     sort_keys=True))
    return 78


if __name__ == "__main__":
    raise SystemExit(main())
