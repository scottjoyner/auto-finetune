"""Data freshness tracking for the real-time pipeline.

Monitors how stale the training data is and triggers alerts when data
exceeds freshness thresholds.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config, load
from src.locking import atomic_write_json


@dataclass
class FreshnessReport:
    """Freshness assessment for the pipeline data."""
    raw_newest_age_hours: float = 0.0
    cleaned_newest_age_hours: float = 0.0
    dataset_newest_age_hours: float = 0.0
    model_age_hours: float = 0.0
    overall_staleness: str = "fresh"  # fresh, aging, stale, critical
    recommendations: list[str] = field(default_factory=list)
    timestamp: float = 0.0


def _newest_file_age(directory: str, pattern: str = "*.json") -> float:
    """Return age in hours of the newest file matching pattern in directory."""
    import glob
    newest = 0.0
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(pattern.replace("*", "")):
                path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(path)
                    if mtime > newest:
                        newest = mtime
                except OSError:
                    pass
    if newest == 0:
        return float("inf")
    return (time.time() - newest) / 3600


def _newest_model_age(checkpoint_dir: str) -> float:
    """Return age in hours of the newest model checkpoint."""
    if not os.path.isdir(checkpoint_dir):
        return float("inf")
    newest = 0.0
    for root, _, files in os.walk(checkpoint_dir):
        for f in files:
            if f.endswith(".safetensors") or f.endswith(".bin"):
                path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(path)
                    if mtime > newest:
                        newest = mtime
                except OSError:
                    pass
    if newest == 0:
        return float("inf")
    return (time.time() - newest) / 3600


def assess_freshness(cfg: Config) -> FreshnessReport:
    """Assess data freshness across all pipeline stages."""
    report = FreshnessReport(timestamp=time.time())

    raw_dir = cfg.path("raw_dir")
    cleaned_dir = cfg.path("cleaned_dir")
    dataset_dir = cfg.path("dataset_dir")

    report.raw_newest_age_hours = _newest_file_age(raw_dir, ".json")
    report.cleaned_newest_age_hours = _newest_file_age(cleaned_dir, ".json")
    report.dataset_newest_age_hours = _newest_file_age(dataset_dir, ".jsonl")

    # Model age
    checkpoint_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
    report.model_age_hours = _newest_model_age(checkpoint_base)

    # Determine staleness
    ages = [report.raw_newest_age_hours, report.cleaned_newest_age_hours,
            report.dataset_newest_age_hours]
    max_age = max(a for a in ages if a != float("inf")) if any(a != float("inf") for a in ages) else float("inf")

    if max_age < 1:
        report.overall_staleness = "fresh"
    elif max_age < 24:
        report.overall_staleness = "aging"
    elif max_age < 168:
        report.overall_staleness = "stale"
    else:
        report.overall_staleness = "critical"

    # Recommendations
    if report.raw_newest_age_hours > 24:
        report.recommendations.append("Raw data is stale; run extraction")
    if report.cleaned_newest_age_hours > report.raw_newest_age_hours + 1:
        report.recommendations.append("Cleaned data lags behind raw; run cleaning")
    if report.dataset_newest_age_hours > report.cleaned_newest_age_hours + 1:
        report.recommendations.append("Dataset lags behind cleaned; run formatting")
    if report.model_age_hours > report.dataset_newest_age_hours + 24:
        report.recommendations.append("Model is stale; consider retraining")
    if report.overall_staleness == "critical":
        report.recommendations.append("CRITICAL: Data is very stale; full pipeline re-run recommended")

    # Save report
    report_path = os.path.join(cfg.path("analysis_dir"), "freshness-report.json")
    try:
        atomic_write_json(report_path, asdict(report))
    except Exception:
        pass

    return report


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Data freshness assessment")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    cfg = load()
    report = assess_freshness(cfg)

    if args.json:
        import json as j
        print(j.dumps(asdict(report), indent=2))
    else:
        print(f"[freshness] Overall: {report.overall_staleness}")
        print(f"[freshness] Raw data age: {report.raw_newest_age_hours:.1f}h")
        print(f"[freshness] Cleaned data age: {report.cleaned_newest_age_hours:.1f}h")
        print(f"[freshness] Dataset age: {report.dataset_newest_age_hours:.1f}h")
        print(f"[freshness] Model age: {report.model_age_hours:.1f}h")
        if report.recommendations:
            print("[freshness] Recommendations:")
            for r in report.recommendations:
                print(f"  - {r}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
