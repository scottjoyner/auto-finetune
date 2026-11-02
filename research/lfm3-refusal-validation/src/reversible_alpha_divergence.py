"""Read-only full-text divergence audit on pinned, private alpha-control journal.

Exports aggregate counts only; does NOT score actionability or safety.
"""
import argparse
import json
import os
from pathlib import Path

from src.reversible_alpha_controls import (
    CONDITIONS, EXPECTED_MODEL, EXPECTED_VECTOR, EXPECTED_PAIRS,
    read_journal, record_key, sha256, summary, write_exclusive,
)


def changes_against_baseline(records, pair_ids):
    by_key = {record_key(r): r for r in records}
    if len(by_key) != len(records):
        raise ValueError("duplicate responses")
    changes = {}
    for category in ("benign", "adversarial"):
        changes[category] = {}
        for condition in CONDITIONS:
            changes[category][condition] = sum(
                by_key[(case, category, condition)]["response"] !=
                by_key[(case, category, "baseline")]["response"]
                for case in pair_ids)
    return changes
def audit_folder(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "private_manifest.json").read_text())
    if (manifest.get("study") != "lfm3-reversible-alpha-control-smoke-v1"
            or manifest.get("checkpoint_sha256") != EXPECTED_MODEL
            or manifest.get("direction_sha256") != EXPECTED_VECTOR
            or manifest.get("pair_dataset_sha256") != EXPECTED_PAIRS
            or manifest.get("conditions") != list(CONDITIONS)
            or manifest.get("layer") != 10 or manifest.get("seed") != 17
            or manifest.get("device") != "cpu"
            or manifest.get("tool_execution") is not False
            or manifest.get("max_new_tokens") not in range(8, 129)):
        raise ValueError("unrecognized experimental protocol")
    pair_ids = manifest.get("holdout_ids")
    if not isinstance(pair_ids, list) or not 1 <= len(pair_ids) <= 6:
        raise ValueError("unrecognized holdout roster")
    if pair_ids != [f"p{i:02d}" for i in range(3, 3 + len(pair_ids))]:
        raise ValueError("holdout selection changed")
    journal = folder / "private_generations.jsonl"
    rows = read_journal(journal, set(pair_ids))
    aggregate = summary(rows, pair_ids, manifest, sha256(journal))
    return {
        "evidence_class": "private-response text divergence; no safety grading",
        "n_holdout_pairs": len(pair_ids), "n_generations": len(rows),
        "checkpoint_sha256": aggregate["checkpoint_sha256"],
        "direction_sha256": aggregate["direction_sha256"],
        "private_journal_sha256": aggregate["private_journal_sha256"],
        "exact_response_changed_vs_baseline": changes_against_baseline(rows, pair_ids),
        "keyword_positive_counts": aggregate["keyword_positive_counts"],
        "baseline_vs_alpha_zero_exact_match": True,
        "human_safety_reviewed": False, "checkpoint_promotion_approved": False,
        "limitations": [
            "Different text may be harmless paraphrase, not changed refusal or usefulness",
            "Identical text does not generalize to other prompts or settings",
            "Keyword matches do not establish appropriate refusal or harmful actionability",
            "Only two contrast-calibration pairs and 128-token-or-shorter local smoke",
        ],
    }
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("immutable aggregate only: choose new evidence name")
    report = audit_folder(args.results_dir)
    os.umask(0o077)
    write_exclusive(args.out, report)
    print(json.dumps({
        "n_generations": report["n_generations"],
        "exact_response_changed_vs_baseline": report["exact_response_changed_vs_baseline"],
        "keyword_positive_counts": report["keyword_positive_counts"],
        "behavioral_safety_verified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
