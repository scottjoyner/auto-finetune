"""Read-only K2-Horizon/Dust inventory. Does not launch training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

DUST_UPSTREAM = "https://github.com/qlabs-eng/dust"
DUST_PIN = "b20f7c03eac630b6441ba6f254128c1761dc45ad"
UPSTREAM_FILES = ("dust.py", "model.py", "requirements.txt", "README.md", "LICENSE")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path, hash_file: bool = False) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError:
        return {"exists": False}
    if not path.is_file():
        return {"exists": False, "reason": "not_regular_file"}
    result: dict[str, Any] = {"exists": True, "bytes": info.st_size}
    if hash_file:
        result["sha256"] = digest(path)
    return result


def load_json(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return (obj, None) if isinstance(obj, dict) else ({}, "not_an_object")
    except (OSError, ValueError, UnicodeError) as exc:
        return {}, type(exc).__name__

def model_info(root: Path, hash_weights: bool = False) -> dict[str, Any]:
    config, config_error = load_json(root / "config.json")
    index_path = root / "model.safetensors.index.json"
    index, index_error = load_json(index_path) if index_path.exists() else ({}, None)
    if index:
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            weight_map = {}
        shards = sorted({name for name in weight_map.values() if isinstance(name, str)})
    else:
        shards = sorted(p.name for p in root.glob("*.safetensors") if p.is_file())
    shard_info = {name: file_info(root / name, hash_weights) for name in shards}
    missing = [name for name, info in shard_info.items() if not info["exists"]]
    return {
        "directory": str(root), "config_error": config_error,
        "index_error": index_error, "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "declared_total_bytes": index.get("metadata", {}).get("total_size") if index else None,
        "shards": shard_info, "shard_count": len(shards), "missing_shards": missing,
        "complete": config_error is None and bool(shards) and not missing and index_error is None,
        "config_sha256": digest(root / "config.json") if config_error is None else None,
    }


def adapter_info(root: Path, hash_weights: bool = False) -> dict[str, Any]:
    config, error = load_json(root / "adapter_config.json")
    files = sorted(root.glob("adapter_model*.safetensors"))
    files += sorted(root.glob("adapter_model*.bin"))
    weights = {p.name: file_info(p, hash_weights) for p in files if p.is_file()}
    return {
        "directory": str(root), "config_error": error,
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "peft_type": config.get("peft_type"), "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": sorted(config.get("target_modules") or []),
        "weights": weights, "complete": error is None and bool(weights),
        "config_sha256": digest(root / "adapter_config.json") if error is None else None,
    }


def dataset_info(path: Path) -> dict[str, Any]:
    """Only hash an explicitly requested split; never retain data records."""
    rows, size = 0, 0
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for line in f:
                rows += bool(line.strip())
                size += len(line)
                h.update(line)
        return {"path": str(path), "exists": True, "bytes": size,
                "nonblank_lines": rows, "sha256": h.hexdigest(),
                "warning": "line count does not prove valid JSONL or disjoint train/eval"}
    except OSError as exc:
        return {"path": str(path), "exists": False, "error": type(exc).__name__}


def git_head(directory: Path) -> tuple[str | None, bool | None]:
    try:
        head = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, check=True, text=True, timeout=8,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, check=True, text=True, timeout=8,
        ).stdout.strip())
        return head, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def upstream_info(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"checked": False, "expected_commit": DUST_PIN, "url": DUST_UPSTREAM}
    head, dirty = git_head(root)
    files = {name: file_info(root / name, name == "requirements.txt") for name in UPSTREAM_FILES}
    return {
        "checked": True, "directory": str(root), "url": DUST_UPSTREAM,
        "expected_commit": DUST_PIN, "actual_commit": head,
        "pinned": head == DUST_PIN and dirty is False,
        "tracked_dirty": dirty, "files": files,
        "all_files_present": all(x["exists"] for x in files.values()),
        "upstream_scope": "custom GPT pretraining; no K2 fine-tuning support claimed",
    }


def runtime_info(probe_torch: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "host": platform.node(), "os": platform.platform(),
        "python": platform.python_version(), "torch_probed": probe_torch,
    }
    if not probe_torch:
        return result
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["torch_hip"] = torch.version.hip
        result["torch_cuda"] = torch.version.cuda
        result["torch_device_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["torch_device_name"] = torch.cuda.get_device_name(0)
            result["bf16_supported"] = torch.cuda.is_bf16_supported()
    except (ImportError, RuntimeError, OSError, AssertionError) as exc:
        result["probe_error"] = type(exc).__name__
    return result


def mount_for(path: Path) -> dict[str, Any]:
    """Find backing mount from /proc/self/mountinfo, including CIFS."""
    try:
        target = os.path.realpath(str(path))
        records = []
        with open("/proc/self/mountinfo", encoding="utf-8") as f:
            for line in f:
                fields = line.split()
                sep = fields.index("-")
                location = fields[4].replace(r"\040", " ")
                if target == location or target.startswith(location.rstrip("/") + "/"):
                    records.append((len(location), location, fields[sep + 1], fields[sep + 2]))
        if records:
            _, location, fs_type, source = max(records)
            return {"path": str(path), "mountpoint": location,
                    "filesystem": fs_type, "source": source}
    except (OSError, ValueError, IndexError):
        pass
    return {"path": str(path), "mountpoint": None, "filesystem": None}


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    model = model_info(args.model_dir, args.sha256_weights)
    adapters = [adapter_info(path, args.sha256_weights) for path in args.adapter]
    upstream = upstream_info(args.upstream_dir)
    runtime = runtime_info(args.probe_torch)
    blockers = []
    if not model["complete"]:
        blockers.append("model_incomplete")
    if any(not a["complete"] for a in adapters):
        blockers.append("adapter_incomplete")
    if upstream["checked"] and not upstream["pinned"]:
        blockers.append("upstream_not_pinned")
    if upstream["checked"] and not upstream["all_files_present"]:
        blockers.append("upstream_missing_files")
    compatibility = "not_probed"
    if runtime.get("torch_hip"):
        compatibility = "rocm_unverified_upstream_nvidia_only"
    elif runtime.get("torch_cuda") and runtime.get("torch_device_available"):
        compatibility = "nvidia_cuda_candidate_not_executed"
    elif args.probe_torch:
        compatibility = "no_compatible_accelerator_detected"
    return {
        "schema": "auto-finetune.dust-k2-preflight.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "read_only_research", "training_authorized": False,
        "upstream": upstream, "runtime": runtime, "compatibility": compatibility,
        "model": model, "adapters": adapters,
        "datasets": [dataset_info(path) for path in args.dataset],
        "nas_mount": mount_for(args.nas_dir),
        "repo_commit": git_head(Path(__file__).resolve().parent.parent)[0],
        "blockers": blockers,
        "notes": [
            "No backward call, optimizer step, GPU workload or training process is started.",
            "Model weight SHA256 exists only when --sha256-weights is used.",
            "Train and held-out splits must be independently deduplicated and scored.",
            "Do not delete local weights until NAS checksum and restore/custody gates pass.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, action="append", default=[])
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--nas-dir", type=Path, default=Path("/nas"))
    parser.add_argument("--probe-torch", action="store_true")
    parser.add_argument("--sha256-weights", action="store_true")
    parser.add_argument("--strict-model", action="store_true",
                        help="Exit 2 when model, adapter or upstream fails preflight.")
    parser.add_argument("--output", type=Path,
                        help="Create NEW JSON file; never overwrite provenance.")
    args = parser.parse_args(argv)
    report = build_manifest(args)
    result = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as f:
                f.write(result)
        except FileExistsError:
            parser.error("output already exists; refusing to overwrite provenance")
    else:
        print(result, end="")
    if args.strict_model and report["blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
