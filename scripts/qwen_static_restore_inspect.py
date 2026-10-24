#!/usr/bin/env python3
"""Static Qwen trainer-state compatibility check; NEVER deserializes PyTorch.

Torch *.pt/*.pth/*.bin files may contain pickle. This inspector reads ONLY
ZIP central directory names and metadata, and bounded JSON config files.
It cannot perform or certify a trainer restoration, GPU admission or resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import stat
import zipfile

from scripts.qwen_resume_preflight import MANDATORY

DEFAULT_RUN = Path(
    "/media/scott/data/finetune-staging/outputs/checkpoints/qwen3-8b-sft-r1"
)
EXPECTED_BASE = "/media/scott/data/finetune-staging/models/Qwen3-8B"
PROTECTED = (100, 150)
ZIP_ARTIFACTS = {
    "optimizer.pt": "optimizer",
    "scheduler.pt": "scheduler",
    "rng_state.pth": "rng_state",
    "training_args.bin": "training_args",
}
MAX_ZIP_ENTRIES = 4096
MAX_ZIP_UNCOMPRESSED = 1 * 1024**3
MAX_JSON_BYTES = 1024 * 1024


def safe_dir(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode)
    except OSError:
        return False


def safe_file(path: Path) -> bool:
    try:
        state = path.lstat()
        return stat.S_ISREG(state.st_mode) and state.st_nlink == 1 and state.st_size > 0
    except OSError:
        return False


def safe_json(path: Path) -> dict | None:
    if not safe_file(path) or path.stat().st_size > MAX_JSON_BYTES:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def inspect_zip(path: Path, prefix: str) -> dict:
    result = {
        "present": safe_file(path),
        "torch_zip_structure": False,
        "archive_members": 0,
        "uncompressed_bytes": 0,
        "reasons": [],
    }
    if not result["present"]:
        result["reasons"].append("state_artifact_missing_or_untrusted")
        return result
    try:
        with zipfile.ZipFile(path, "r", allowZip64=True) as archive:
            entries = archive.infolist()
            result["archive_members"] = len(entries)
            if not entries or len(entries) > MAX_ZIP_ENTRIES:
                result["reasons"].append("archive_member_count_unbounded")
                return result
            seen = set()
            total = 0
            data_pickle = False
            version = False
            for info in entries:
                name = info.filename
                if name in seen:
                    result["reasons"].append("duplicate_archive_member")
                seen.add(name)
                parts = name.split("/")
                if (info.is_dir() or name.startswith("/")
                        or len(parts) < 2 or parts[0] != prefix
                        or any(part in ("", ".", "..") for part in parts)
                        or "\\" in name):
                    result["reasons"].append("untrusted_archive_member")
                if info.flag_bits & 0x1:
                    result["reasons"].append("encrypted_archive_member")
                if info.file_size < 0 or info.compress_size < 0:
                    result["reasons"].append("archive_size_invalid")
                total += info.file_size
                if total > MAX_ZIP_UNCOMPRESSED:
                    result["reasons"].append("archive_uncompressed_limit_exceeded")
                    break
                if name == prefix + "/data.pkl":
                    data_pickle = True
                if name == prefix + "/version":
                    version = True
            result["uncompressed_bytes"] = total
            if not (data_pickle and version):
                result["reasons"].append("torch_zip_metadata_incomplete")
            result["reasons"] = sorted(set(result["reasons"]))
            result["torch_zip_structure"] = not result["reasons"]
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError, ValueError):
        result["reasons"].append("zip_archive_unreadable")
    return result


def inspect(run: Path) -> dict:
    problems: list[str] = []
    rows: list[dict] = []
    if not safe_dir(run):
        problems.append("run_directory_missing_or_linked")
    references: list[tuple[dict, dict, dict]] = []
    for step in PROTECTED:
        path = run / f"checkpoint-{step}"
        reasons = []
        if not safe_dir(path):
            reasons.append("checkpoint_directory_missing_or_linked")
        state = safe_json(path / "trainer_state.json")
        adapter = safe_json(path / "adapter_config.json")
        tokenizer = safe_json(path / "tokenizer_config.json")
        if state is None:
            reasons.append("trainer_state_json_missing_or_invalid")
        else:
            if type(state.get("global_step")) is not int or state["global_step"] != step:
                reasons.append("checkpoint_global_step_mismatch")
            if type(state.get("max_steps")) is not int or state["max_steps"] <= step:
                reasons.append("trainer_max_steps_invalid")
            if type(state.get("num_train_epochs")) not in (int, float) or state["num_train_epochs"] <= 0:
                reasons.append("trainer_epochs_invalid")
        if adapter is None:
            reasons.append("adapter_config_json_missing_or_invalid")
        else:
            if adapter.get("base_model_name_or_path") != EXPECTED_BASE:
                reasons.append("adapter_base_model_mismatch")
            if (adapter.get("peft_type") != "LORA"
                    or adapter.get("task_type") != "CAUSAL_LM"
                    or type(adapter.get("r")) is not int or adapter["r"] < 1
                    or type(adapter.get("lora_alpha")) is not int or adapter["lora_alpha"] < 1):
                reasons.append("adapter_lora_configuration_invalid")
            targets = adapter.get("target_modules")
            if not isinstance(targets, list) or not targets or any(
                not isinstance(x,str) or not re.fullmatch(r"[A-Za-z0-9_]+",x)
                for x in targets
            ) or len(set(targets)) != len(targets):
                reasons.append("adapter_target_modules_invalid")
        if tokenizer is None or not isinstance(tokenizer.get("tokenizer_class"),str):
            reasons.append("tokenizer_config_missing_or_invalid")
        missing = [
            name for name in MANDATORY
            if not safe_file(path / name)
        ]
        if missing:
            reasons.append("required_checkpoint_files_missing")
        archives = {
            name: inspect_zip(path / name,prefix)
            for name,prefix in ZIP_ARTIFACTS.items()
        }
        for info in archives.values():
            reasons.extend(info["reasons"])
        rows.append({
            "step": step,
            "metadata": {
                "global_step":state.get("global_step") if state else None,
                "max_steps":state.get("max_steps") if state else None,
                "num_train_epochs":state.get("num_train_epochs") if state else None,
                "base_model":adapter.get("base_model_name_or_path") if adapter else None,
                "peft_type":adapter.get("peft_type") if adapter else None,
                "lora_rank":adapter.get("r") if adapter else None,
                "lora_alpha":adapter.get("lora_alpha") if adapter else None,
                "target_modules":sorted(adapter.get("target_modules"))
                    if adapter and isinstance(adapter.get("target_modules"),list)
                    and all(isinstance(x,str) for x in adapter["target_modules"])
                    else [],
            },
            "torch_archives":archives,
            "reasons":sorted(set(reasons)),
        })
        if state is not None and adapter is not None and tokenizer is not None:
            references.append((state,adapter,tokenizer))
        problems.extend(reasons)

    if len(references) == 2:
        first,second = references
        for field in ("max_steps", "num_train_epochs"):
            if first[0].get(field) != second[0].get(field):
                problems.append("trainer_schedule_differs_across_checkpoints")
        for field in ("base_model_name_or_path","peft_type","task_type",
                      "r","lora_alpha","lora_dropout","target_modules"):
            a,b=first[1].get(field),second[1].get(field)
            if field=="target_modules" and isinstance(a,list) and isinstance(b,list):
                a,b=set(a),set(b)
            if a!=b:
                problems.append("adapter_identity_differs_across_checkpoints")
        if first[2].get("tokenizer_class")!=second[2].get("tokenizer_class"):
            problems.append("tokenizer_identity_differs_across_checkpoints")

    return {
        "status":"NO_VERIFIED_RESTORE",
        "read_only":True,
        "untrusted_torch_pickle_deserialized":False,
        "metadata_structure_consistent":len(rows)==2 and not problems,
        "issues":sorted(set(problems)),
        "checkpoints":rows,
        "backup_copy_verified":False,
        "optimizer_restored":False,
        "scheduler_restored":False,
        "rng_restored":False,
        "trainer_restored":False,
        "may_copy":False,
        "may_resume":False,
        "may_train":False,
        "may_prune":False,
        "may_release_source":False,
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT_RUN)
    args=parser.parse_args()
    print(json.dumps(inspect(args.run),sort_keys=True))
    return 78


if __name__ == "__main__":
    raise SystemExit(main())
