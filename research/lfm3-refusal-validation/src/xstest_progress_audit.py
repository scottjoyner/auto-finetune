"""Read-only, fail-closed audit of a live three-checkpoint XSTest journal.

Does not load models, change journal files, or infer behavioral safety.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

from src.three_checkpoint_eval import ARMS
from src.xstest_450_eval import (
    SOURCE_SHA256, read_verified_dataset, stratified_selection,
)

ARMS_ORDER = tuple(ARMS)


def snapshot_complete_lines(path):
    data = Path(path).read_bytes() if Path(path).exists() else b""
    complete, separator, partial = data.rpartition(b"\n")
    if not separator:
        if data:
            return [], True, hashlib.sha256(b"").hexdigest()
        return [], False, hashlib.sha256(b"").hexdigest()
    raw = complete + b"\n"
    try:
        rows = [json.loads(s) for s in raw.decode("utf-8").splitlines() if s]
    except (ValueError, UnicodeError) as exc:
        raise ValueError("malformed completed JSONL record") from exc
    return rows, bool(partial), hashlib.sha256(raw).hexdigest()


def validate_records(manifest, records, corpus):
    expected_ids = manifest["selection_ids"]
    catalog = {r["id"]: r for r in corpus}
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate ID in manifest")
    if not set(expected_ids).issubset(catalog):
        raise ValueError("manifest contains unknown benchmark IDs")
    if manifest["checkpoint_order"] != list(ARMS_ORDER):
        raise ValueError("not the frozen three-arm comparison")
    if len(set(manifest["checkpoint_sha256"].values())) != 3:
        raise ValueError("checkpoint aliases or duplicates")
    if set(manifest["checkpoint_sha256"]) != set(ARMS_ORDER):
        raise ValueError("missing original, edited or base checkpoint")
    if manifest["source_sha256"] != SOURCE_SHA256:
        raise ValueError("dataset checksum mismatch")
    groups = collections.Counter()
    ids = set()
    last_position = -1
    for row in records:
        arm, case_id = row.get("arm"), row.get("id")
        if arm not in ARMS_ORDER or case_id not in expected_ids:
            raise ValueError("out-of-scope checkpoint or benchmark case")
        key = (arm, case_id)
        if key in ids:
            raise ValueError("duplicate model and case")
        ids.add(key)
        case = catalog[case_id]
        if row.get("type") != case["type"] or row.get("class") != case["class"]:
            raise ValueError("tampered benchmark label or type")
        if not isinstance(row.get("response"), str):
            raise ValueError("generation is not a string")
        if not isinstance(row.get("refusal_proxy"), bool):
            raise ValueError("proxy field is invalid")
        elapsed = row.get("elapsed_seconds")
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid generation timing")
        position = ARMS_ORDER.index(arm) * len(expected_ids) + expected_ids.index(case_id)
        if position <= last_position:
            raise ValueError("out-of-order journal; requires manual provenance review")
        last_position = position
        groups[(arm, row["class"])] += 1
    expected = len(expected_ids) * len(ARMS_ORDER)
    return {
        "study": "XSTest three-checkpoint journal",
        "journal_integrity": "passed",
        "progress_only_not_safety": True,
        "expected_generations": expected,
        "recorded_generations": len(records),
        "completion_fraction": round(len(records) / expected, 6) if expected else 0,
        "per_arm": {
            arm: {
                "recorded": sum(groups[(arm, c)] for c in ("safe", "unsafe")),
                "safe_recorded": groups[(arm, "safe")],
                "unsafe_recorded": groups[(arm, "unsafe")],
                "remaining": len(expected_ids) - sum(groups[(arm, c)] for c in ("safe", "unsafe")),
            } for arm in ARMS_ORDER
        },
        "manual_review": "NOT STARTED",
        "evidence_status": "INCOMPLETE" if len(records) < expected else "GENERATIONS COMPLETE; HUMAN REVIEW PENDING",
    }


def audit(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "private_run_manifest.json").read_text())
    corpus = read_verified_dataset()
    selection = stratified_selection(corpus, manifest["per_category"], manifest["seed"])
    if [x["id"] for x in selection] != manifest["selection_ids"]:
        raise ValueError("manifest selection differs from pinned stratification")
    rows, pending_partial_line, digest = snapshot_complete_lines(folder / "private_generations.jsonl")
    summary = validate_records(manifest, rows, corpus)
    summary["journal_complete_lines_sha256"] = digest
    summary["ignored_unterminated_tail"] = pending_partial_line
    summary["benchmark_sha256"] = SOURCE_SHA256
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.results_dir), indent=2))


if __name__ == "__main__":
    main()
