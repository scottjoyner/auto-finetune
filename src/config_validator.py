"""Configuration validation for auto-finetune.

Validates config.yaml on startup and reports issues.
"""
from __future__ import annotations

import os
from typing import Any

from src.config import Config, load


REQUIRED_PATHS = ["raw_dir", "cleaned_dir", "dataset_dir", "analysis_dir"]
VALID_BACKENDS = {"auto", "unsloth", "peft"}
VALID_FORMATS = {"chatml", "alpaca", "sharegpt", "hermes"}


def validate(cfg: Config) -> list[str]:
    """Validate configuration and return list of issues."""
    issues = []

    # Check paths
    for p in REQUIRED_PATHS:
        val = cfg.path(p) if hasattr(cfg, 'path') else None
        if val is None:
            issues.append(f"Missing path: {p}")
        elif not os.path.isabs(val):
            issues.append(f"Path not absolute: {p}={val}")

    # Check training config
    t = cfg.get("train", default={}) or {}
    backend = t.get("backend", "auto")
    if backend not in VALID_BACKENDS:
        issues.append(f"Invalid backend: {backend} (must be one of {VALID_BACKENDS})")

    model_name = t.get("model_name")
    if not model_name:
        issues.append("Missing train.model_name")

    max_seq = t.get("max_seq_length", 0)
    if not isinstance(max_seq, int) or max_seq < 1:
        issues.append(f"Invalid max_seq_length: {max_seq}")

    # Check sources
    sources = cfg.get("sources", default={}) or {}
    if not sources:
        issues.append("No sources configured")

    opencode = sources.get("opencode", {})
    if opencode.get("enabled", True):
        db_path = opencode.get("db_path")
        if db_path and not os.path.exists(db_path):
            issues.append(f"OpenCode DB not found: {db_path}")

    hermes = sources.get("hermes", {})
    if hermes.get("enabled", True):
        state_db = hermes.get("state_db")
        if state_db and not os.path.exists(state_db):
            issues.append(f"Hermes state.db not found: {state_db}")

    # Check extract config
    extract = cfg.get("extract", default={}) or {}
    min_msgs = extract.get("min_messages", 0)
    if not isinstance(min_msgs, int) or min_msgs < 0:
        issues.append(f"Invalid min_messages: {min_msgs}")

    # Check clean config
    clean = cfg.get("clean", default={}) or {}
    max_chars = clean.get("max_chars_per_message", 0)
    if not isinstance(max_chars, int) or max_chars < 1:
        issues.append(f"Invalid max_chars_per_message: {max_chars}")

    # Check format config
    fmt = cfg.get("format", default={}) or {}
    template = fmt.get("template", "chatml")
    if template not in VALID_FORMATS:
        issues.append(f"Invalid format template: {template}")

    return issues


def validate_and_report(cfg: Config) -> bool:
    """Validate config and print report. Returns True if valid."""
    issues = validate(cfg)
    if issues:
        print("[config] Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    print("[config] Configuration valid")
    return True


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Config validation")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on issues")
    args = parser.parse_args(argv)

    cfg = load()
    issues = validate(cfg)
    if issues:
        print(f"[config] {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        if args.strict:
            return 1
    else:
        print("[config] Configuration valid")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
