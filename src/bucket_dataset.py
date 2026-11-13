"""Bucket the decontaminated dataset by behavior/tool for targeted finetuning.

Reads train.decontaminated.jsonl and writes per-tool bucketed JSONL files plus
a manifest. Each example is assigned to one or more behavior buckets based on
the tools it invokes. Examples with no tool calls go to a "general" bucket.

Usage:
    python3 -m src.bucket_dataset \
        --input data/datasets/train.decontaminated.jsonl \
        --out data/datasets/buckets \
        [--min-per-bucket 5] [--overlap]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.locking import atomic_write_json


def _tool_names(messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for m in messages:
        for tc in m.get("tool_calls", []) or []:
            func = tc.get("function", tc) if isinstance(tc, dict) else tc
            if isinstance(func, dict):
                name = func.get("name", "unknown")
            else:
                name = "unknown"
            names.add(name)
    return names


# Behavior-group mappings: group label -> set of tool names
BEHAVIOR_GROUPS: dict[str, set[str]] = {
    "filesystem": {
        "read_file", "write_file", "patch", "search_files", "list_dir",
        "mkdir", "rm", "cp", "mv",
    },
    "code_execution": {
        "execute_code", "process", "terminal", "cronjob",
    },
    "knowledge_graph": {
        "kg_query", "kg_search", "kg_index_docs", "kg_remember_idea",
        "kg_status", "mcp_neo4j_run_read_query", "mcp_neo4j_run_query",
        "mcp_neo4j_get_server_info", "mcp_neo4j_get_schema",
        "mcp_neo4j_get_labels",
    },
    "memory": {
        "memory", "skill_view", "skill_manage", "skills_list",
    },
    "planning": {
        "todo", "delegate_task", "clarify",
    },
    "session": {
        "session_search",
    },
    "browser": {
        "browser_navigate", "browser_console",
    },
    "voice": {
        "sophia_voice_status",
    },
    "messaging": {
        "send_message",
    },
}


def bucket_example(example: dict[str, Any], overlap: bool = False) -> list[str]:
    """Return the list of bucket labels for an example."""
    names = _tool_names(example.get("messages", []))
    if not names:
        return ["general"]

    matched: set[str] = set()
    for group_label, group_tools in BEHAVIOR_GROUPS.items():
        if names & group_tools:
            matched.add(group_label)

    # Per-tool buckets (each tool gets its own bucket)
    per_tool = [f"tool:{n}" for n in sorted(names)]

    if overlap:
        return sorted(matched | set(per_tool))
    # Non-overlap: assign to the most specific single bucket
    if per_tool:
        return [per_tool[0]]  # first tool alphabetically as primary
    return sorted(matched) or ["general"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bucket decontaminated dataset by behavior/tool"
    )
    parser.add_argument(
        "--input",
        default="/media/scott/data/finetune-staging/data/datasets/train.decontaminated.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--out",
        default="/media/scott/data/finetune-staging/data/datasets/buckets",
        help="Output bucket directory",
    )
    parser.add_argument(
        "--min-per-bucket",
        type=int,
        default=5,
        help="Minimum examples per bucket (else merged into 'misc')",
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="Allow examples to appear in multiple buckets",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    input_path = Path(args.input)
    # Never turn absent, malformed or empty input into a new empty dataset,
    # and never silently overwrite a prior signed research receipt.
    if (args.min_per_bucket < 1 or not input_path.is_file()
            or out_dir.exists() or out_dir.is_symlink()):
        print("[bucket] HOLD: invalid input/threshold or existing output")
        return 78

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total = 0
    try:
        with input_path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                ex = json.loads(line)
                if (not isinstance(ex, dict)
                        or not isinstance(ex.get("messages"), list)
                        or not ex["messages"]
                        or not all(isinstance(m, dict) for m in ex["messages"])):
                    raise ValueError("invalid session record")
                labels = bucket_example(ex, overlap=args.overlap)
                for lbl in labels:
                    # Tool names are untrusted; labels become local filenames.
                    if not isinstance(lbl, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", lbl) or ".." in lbl:
                        raise ValueError("untrusted bucket filename")
                    buckets[lbl].append(ex)
                total += 1
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        print("[bucket] HOLD: input record or label invalid")
        return 78
    if total == 0:
        print("[bucket] HOLD: zero valid training records")
        return 78

    # Merge small buckets into misc, retaining the original source total.
    small = [label for label, records in buckets.items()
             if len(records) < args.min_per_bucket]
    for label in small:
        buckets["misc"].extend(buckets.pop(label))

    manifest: dict[str, Any] = {
        "source": str(input_path),
        "total_examples": total,
        "min_per_bucket": args.min_per_bucket,
        "overlap": args.overlap,
        "buckets": {},
    }

    # Stage all outputs on the destination filesystem before one atomic
    # directory rename. A failed write never publishes a partial manifest.
    stage = None
    try:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".bucket-stage-", dir=out_dir.parent))
        for label in sorted(buckets):
            bucket_path = stage / f"{label}.jsonl"
            with bucket_path.open("w") as output:
                for ex in buckets[label]:
                    output.write(json.dumps(ex, ensure_ascii=False) + "\n")
            manifest["buckets"][label] = {
                "count": len(buckets[label]),
                "file": str(out_dir / f"{label}.jsonl"),
            }
        atomic_write_json(stage / "manifest.json", manifest)
        if out_dir.exists() or out_dir.is_symlink():
            raise FileExistsError("output appeared during stage")
        stage.rename(out_dir)
        stage = None
    except (OSError, ValueError, TypeError):
        print("[bucket] HOLD: atomic dataset publication unavailable")
        return 78
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)

    manifest_path = out_dir / "manifest.json"
    print(f"[bucket] total examples: {total}")
    print(f"[bucket] buckets created: {len(manifest['buckets'])}")
    print(f"[bucket] output dir: {out_dir}")
    print(f"[bucket] manifest: {manifest_path}")
    print()
    print("Top buckets:")
    for lbl in sorted(manifest["buckets"], key=lambda k: -manifest["buckets"][k]["count"])[:15]:
        info = manifest["buckets"][lbl]
        print(f"  {info['count']:5d}  {lbl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
