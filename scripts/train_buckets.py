#!/usr/bin/env python3
"""Train on all behavior buckets sequentially.

Iterates through the bucketed datasets created by src/bucket_dataset.py and
runs training (dry-run or real) on each bucket. Produces a summary report.

Usage:
    # Dry-run all buckets (CPU smoke test)
    python3 scripts/train_buckets.py --dry-run

    # Train on specific behavior group
    python3 scripts/train_buckets.py --group code_execution --live

    # Train on all buckets with GPU
    python3 scripts/train_buckets.py --live --max-examples 500
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


BUCKET_DIR = Path("/media/scott/data/finetune-staging/data/datasets/buckets")
MANIFEST = BUCKET_DIR / "manifest.json"

BEHAVIOR_GROUPS = {
    "code_execution": ["code_execution", "tool:terminal", "tool:execute_code", "tool:process", "tool:cronjob"],
    "filesystem": ["filesystem", "tool:read_file", "tool:write_file", "tool:patch", "tool:search_files"],
    "knowledge_graph": ["knowledge_graph", "tool:kg_query", "tool:kg_search", "tool:kg_index_docs", "tool:kg_remember_idea", "tool:mcp_neo4j_run_read_query", "tool:mcp_neo4j_run_query", "tool:mcp_neo4j_get_schema", "tool:mcp_neo4j_get_server_info", "tool:mcp_neo4j_get_labels"],
    "memory": ["memory", "tool:memory", "tool:skill_view", "tool:skill_manage", "tool:skills_list"],
    "planning": ["planning", "tool:todo", "tool:delegate_task", "tool:clarify"],
    "session": ["session", "tool:session_search"],
    "browser": ["browser", "tool:browser_console", "tool:browser_navigate"],
    "messaging": ["messaging", "tool:send_message"],
    "general": ["general"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train on behavior buckets")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, no GPU")
    parser.add_argument("--live", action="store_true", help="Actually train (requires GPU)")
    parser.add_argument("--group", type=str, default=None, help="Train only this behavior group")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples per bucket")
    parser.add_argument("--min-bucket-size", type=int, default=10, help="Skip buckets smaller than this")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated bucket names to skip")
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"[error] manifest not found: {MANIFEST}")
        return 1

    manifest = json.loads(MANIFEST.read_text())
    buckets = manifest["buckets"]

    skip_set = set(s.strip() for s in args.skip.split(",") if s.strip())

    # Determine which buckets to train
    to_train: list[str] = []
    if args.group:
        if args.group not in BEHAVIOR_GROUPS:
            print(f"[error] unknown group: {args.group}")
            print(f"Available groups: {list(BEHAVIOR_GROUPS.keys())}")
            return 1
        for b in BEHAVIOR_GROUPS[args.group]:
            if b in buckets and buckets[b]["count"] >= args.min_bucket_size and b not in skip_set:
                to_train.append(b)
    else:
        for b in buckets:
            if buckets[b]["count"] >= args.min_bucket_size and b not in skip_set:
                to_train.append(b)

    to_train.sort(key=lambda b: -buckets[b]["count"])

    print(f"[train-buckets] training {len(to_train)} buckets")
    print(f"[train-buckets] mode: {'dry-run' if args.dry_run else 'LIVE' if args.live else 'dry-run (default)'}")
    print()

    results: list[dict[str, Any]] = []
    for i, bucket in enumerate(to_train, 1):
        count = buckets[bucket]["count"]
        print(f"[{i}/{len(to_train)}] {bucket} ({count} examples)...")

        env = os.environ.copy()
        env["TRAIN_ALLOW_UNVERIFIED"] = "1"
        env["AF_SKIP_UNMANAGED_CHECK"] = "1"

        cmd = [sys.executable, "-m", "src.cli", "train", f"--source={bucket}"]
        if args.dry_run or not args.live:
            cmd.append("--dry-run")
        if args.max_examples:
            cmd.append(f"--max-examples={args.max_examples}")

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd="/media/scott/data/git/auto-finetune",
                timeout=300,
            )
            ok = proc.returncode == 0
            # Extract key line
            summary = "OK" if ok else "FAILED"
            for line in proc.stdout.splitlines():
                if "dry-run OK" in line or "examples loaded" in line:
                    summary = line.strip()
                    break
            print(f"  {'✓' if ok else '✗'} {summary}")
            results.append({"bucket": bucket, "count": count, "ok": ok, "summary": summary})
        except subprocess.TimeoutExpired:
            print("  ✗ TIMEOUT")
            results.append({"bucket": bucket, "count": count, "ok": False, "summary": "TIMEOUT"})
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({"bucket": bucket, "count": count, "ok": False, "summary": str(e)})

    # Summary
    print()
    print("=" * 60)
    passed = sum(1 for r in results if r["ok"])
    failed = sum(1 for r in results if not r["ok"])
    print(f"[train-buckets] {passed} passed, {failed} failed out of {len(results)}")
    if failed:
        print("Failed buckets:")
        for r in results:
            if not r["ok"]:
                print(f"  - {r['bucket']}: {r['summary']}")
    print("=" * 60)

    # Write report
    report_path = Path("/media/scott/data/finetune-staging/data/datasets/buckets/training_report.json")
    report = {
        "mode": "dry-run" if (args.dry_run or not args.live) else "live",
        "group": args.group,
        "max_examples": args.max_examples,
        "results": results,
        "passed": passed,
        "failed": failed,
        "total": len(results),
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[train-buckets] report written to {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
