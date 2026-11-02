"""Observed XSTest elapsed-time sensitivity to overlapping answer word lengths.

Uses 1350 frozen responses and only publishes aggregates. This is NOT
controlled output-length inference and whitespace words are NOT model tokens.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.xstest_450_eval import ARM_ORDER, read_verified_dataset
from src.xstest_latency_repetition import median, summarize
from src.xstest_progress_audit import audit, snapshot_complete_lines

C = ARM_ORDER[2]
COMPARATORS = ARM_ORDER[:2]
WORD_GAPS = (0, 5, 10, 20)
EXPECTED_GROUPS = {
    ("safe", "repeated"): 29,
    ("safe", "other"): 221,
    ("unsafe", "repeated"): 127,
    ("unsafe", "other"): 73,
}


def analyze(rows, dataset):
    """Paired output-length overlap, not a causal or token-speed comparison."""
    source_check = summarize(rows, dataset)
    if source_check["n_generations"] != 1350:
        raise ValueError("requires the complete frozen three-arm XSTest panel")
    by_key = {(row["id"], row["arm"]): row for row in rows}
    sources = {row["id"]: row for row in dataset}
    frequencies = Counter(by_key[(source_id, C)]["response"]
                          for source_id in sources)
    mode, count = frequencies.most_common(1)[0]
    if count != 156:
        raise ValueError("frozen C repetition count changed")
    groups = {}
    for (category, response_group), expected in EXPECTED_GROUPS.items():
        matched = [
            sid for sid, meta in sources.items()
            if meta["class"] == category
            and (by_key[(sid, C)]["response"] == mode) ==
                (response_group == "repeated")
        ]
        if len(matched) != expected:
            raise ValueError("frozen C modal grouping changed")
        groups[(category, response_group)] = matched
    results = []
    for (category, group), members in groups.items():
        for other in COMPARATORS:
            for gap in WORD_GAPS:
                eligible = [
                    sid for sid in members
                    if abs(len(by_key[(sid, C)]["response"].split()) -
                           len(by_key[(sid, other)]["response"].split())) <= gap
                ]
                n = len(eligible)
                seconds = [
                    by_key[(sid, C)]["elapsed_seconds"] -
                    by_key[(sid, other)]["elapsed_seconds"]
                    for sid in eligible
                ]
                results.append({
                    "prompt_class": category,
                    "C_output_group": group,
                    "comparator": other,
                    "max_absolute_whitespace_word_gap": gap,
                    "source_group_count": len(members),
                    "matched_length_overlap_count": n,
                    "overlap_fraction_of_group": round(n / len(members), 6),
                    "C_faster_count": sum(d < 0 for d in seconds),
                    "C_slower_count": sum(d > 0 for d in seconds),
                    "recorded_time_ties": sum(d == 0 for d in seconds),
                    "median_paired_C_minus_comparator_elapsed_seconds":
                        median(seconds) if seconds else None,
                    "descriptive_only_not_statistical_inference": True,
                })
    if len(results) != 32:
        raise AssertionError("expected 4 groups x 2 comparators x 4 length thresholds")
    for row in results:
        assert (row["C_faster_count"] + row["C_slower_count"] +
                row["recorded_time_ties"] ==
                row["matched_length_overlap_count"])
    return {
        "evidence_class": "post-hoc whitespace-word-length overlap sensitivity",
        "source_prompts": 450,
        "frozen_generations": 1350,
        "mode_C_exact_response_count": 156,
        "word_gap_thresholds": list(WORD_GAPS),
        "results": results,
        "tokenizer_token_counts_recorded": False,
        "generated_token_throughput_measured": False,
        "causal_speed_advantage_proven": False,
        "human_refusal_or_safety_evaluated": False,
        "release_approved": False,
        "limitations": [
            "Whitespace words are not tokenizer tokens; no per-token decode timing exists",
            "Conditioning on C's modal output is post hoc and not randomized",
            "Length overlap is also post hoc and may remove most repeated C cases",
            "Elapsed seconds mix prefill, generation and runtime overhead",
            "Overlap windows 0, 5, 10 and 20 are sensitivity thresholds not preregistered hypotheses",
            "No statistical hypothesis tests, population correction or causal effects are implied",
            "Unreviewed lexical proxies are not justified-refusal labels",
        ],
    }
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("immutable aggregate must not be overwritten")
    progress = audit(args.results_dir)
    if (progress["recorded_generations"] != 1350 or
            progress["journal_integrity"] != "passed" or
            progress["ignored_unterminated_tail"] or
            progress["checkpoint_sha256"] != checked_arm_paths(ARMS)):
        raise ValueError("original study provenance or checkpoints changed")
    records, partial, digest = snapshot_complete_lines(
        args.results_dir / "private_generations.jsonl")
    if partial or digest != progress["journal_complete_lines_sha256"]:
        raise ValueError("journal changed during audit")
    output = analyze(records, read_verified_dataset())
    output["private_journal_sha256"] = digest
    output["benchmark_sha256"] = progress["benchmark_sha256"]
    output["checkpoint_sha256"] = progress["checkpoint_sha256"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "frozen_generations": output["frozen_generations"],
        "length_overlap_rows": len(output["results"]),
        "decode_tokens_per_second_proven": False,
        "human_refusal_verified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
