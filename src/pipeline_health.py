"""Pipeline health checker and synergy validator.

Validates that the data preprocessing pipeline (extract → clean → format → train)
is consistent and that all components can work together. Provides a single
entry point for verifying pipeline integrity.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.config import Config


def check_raw_dir(cfg: Config) -> dict[str, Any]:
    """Validate the raw extraction output directory."""
    raw_dir = cfg.path("raw_dir")
    result = {"status": "ok", "path": raw_dir, "files": 0, "errors": []}
    if not os.path.isdir(raw_dir):
        result["status"] = "missing"
        result["errors"].append(f"raw_dir does not exist: {raw_dir}")
        return result
    try:
        files = []
        for root, _, fns in os.walk(raw_dir):
            for fn in fns:
                if fn.endswith(".json"):
                    files.append(os.path.join(root, fn))
        result["files"] = len(files)
        # Validate each file is parseable JSON
        bad = 0
        for fn in files[:100]:  # cap to avoid huge dirs
            try:
                with open(os.path.join(raw_dir, fn)) as f:
                    json.load(f)
            except Exception:
                bad += 1
        if bad:
            result["errors"].append(f"{bad} corrupt JSON files in raw_dir")
            result["status"] = "degraded"
    except OSError as e:
        result["status"] = "error"
        result["errors"].append(str(e))
    return result


def check_cleaned_dir(cfg: Config) -> dict[str, Any]:
    """Validate the cleaned output directory."""
    cleaned_dir = cfg.path("cleaned_dir")
    result = {"status": "ok", "path": cleaned_dir, "files": 0, "errors": []}
    if not os.path.isdir(cleaned_dir):
        result["status"] = "missing"
        result["errors"].append(f"cleaned_dir does not exist: {cleaned_dir}")
        return result
    try:
        files = []
        for root, _, fns in os.walk(cleaned_dir):
            for fn in fns:
                if fn.endswith(".json"):
                    files.append(os.path.join(root, fn))
        result["files"] = len(files)
        bad = 0
        for fp in files[:100]:
            try:
                with open(fp) as f:
                    rec = json.load(f)
                if not isinstance(rec, dict) or "messages" not in rec:
                    bad += 1
            except Exception:
                bad += 1
        if bad:
            result["errors"].append(f"{bad} invalid cleaned files")
            result["status"] = "degraded"
    except OSError as e:
        result["status"] = "error"
        result["errors"].append(str(e))
    return result


def check_dataset_dir(cfg: Config) -> dict[str, Any]:
    """Validate the dataset output directory."""
    dataset_dir = cfg.path("dataset_dir")
    result = {"status": "ok", "path": dataset_dir, "files": 0, "errors": []}
    if not os.path.isdir(dataset_dir):
        result["status"] = "missing"
        result["errors"].append(f"dataset_dir does not exist: {dataset_dir}")
        return result
    try:
        files = [f for f in os.listdir(dataset_dir) if f.endswith(".jsonl")]
        result["files"] = len(files)
        total_rows = 0
        for fn in files:
            fp = os.path.join(dataset_dir, fn)
            try:
                with open(fp) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            json.loads(line)
                            total_rows += 1
            except Exception as e:
                result["errors"].append(f"{fn}: {e}")
                result["status"] = "degraded"
        result["total_rows"] = total_rows
    except OSError as e:
        result["status"] = "error"
        result["errors"].append(str(e))
    return result


def check_pipeline(cfg: Config) -> dict[str, Any]:
    """Run a full pipeline health check.

    Returns a dict with status, per-stage results, and overall health.
    """
    raw = check_raw_dir(cfg)
    cleaned = check_cleaned_dir(cfg)
    dataset = check_dataset_dir(cfg)

    stages = {"raw": raw, "cleaned": cleaned, "dataset": dataset}
    errors = []
    for name, r in stages.items():
        if r["status"] != "ok":
            errors.append(f"{name}: {r['status']} - {'; '.join(r['errors'])}")

    overall = "healthy" if not errors else ("degraded" if len(errors) < 3 else "broken")
    return {
        "status": overall,
        "stages": stages,
        "errors": errors,
        "summary": f"Pipeline {overall}: raw={raw['files']} files, "
                   f"cleaned={cleaned['files']} files, dataset={dataset['files']} files",
    }


def validate_synergy(cfg: Config) -> list[str]:
    """Check that pipeline components are properly connected.

    Returns a list of issues found (empty = all good).
    """
    issues = []
    raw_dir = cfg.path("raw_dir")
    cleaned_dir = cfg.path("cleaned_dir")
    dataset_dir = cfg.path("dataset_dir")

    # Check that cleaned dir has corresponding raw sources
    if os.path.isdir(cleaned_dir) and os.path.isdir(raw_dir):
        raw_entries = set()
        for entry in os.listdir(raw_dir):
            if os.path.isdir(os.path.join(raw_dir, entry)):
                raw_entries.add(entry)
            elif entry.endswith(".json"):
                raw_entries.add("__flat__")

        cleaned_entries = set()
        for entry in os.listdir(cleaned_dir):
            if os.path.isdir(os.path.join(cleaned_dir, entry)):
                cleaned_entries.add(entry)
            elif entry.endswith(".json"):
                cleaned_entries.add("__flat__")

        # Account for derived cleaned subdirs (e.g. hermes-reasoning is produced
        # by cleaning hermes with keep_reasoning=True, not by a separate raw dir).
        derived = {"hermes-reasoning"}
        orphaned = (cleaned_entries - raw_entries) - derived
        if orphaned and "__flat__" not in orphaned:
            issues.append(f"cleaned_dir has entries not found in raw_dir: {orphaned}")

    # Check that dataset dir has corresponding cleaned data
    if os.path.isdir(dataset_dir) and os.path.isdir(cleaned_dir):
        has_jsonl = any(f.endswith(".jsonl") for f in os.listdir(dataset_dir))
        has_cleaned = any(
            f.endswith(".json") for f in os.listdir(cleaned_dir)
            if os.path.isfile(os.path.join(cleaned_dir, f))
        ) or any(
            os.path.isdir(os.path.join(cleaned_dir, d)) for d in os.listdir(cleaned_dir)
        )
        if has_cleaned and not has_jsonl:
            issues.append("cleaned data exists but no .jsonl dataset files found")

    return issues
