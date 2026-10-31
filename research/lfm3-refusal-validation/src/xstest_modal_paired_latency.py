"""Paired latency by C's repeated vs nonrepeated output on frozen XSTest.

Uses existing 1350 CPU BF16 responses. Grouping by *observed C output*
is post-hoc and cannot prove a causal speed benefit, throughput or safety.
Only aggregates leave the original private journal.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from src.arc_paired_exact_inference import (
    exact_two_sided_discordant_p, holm_bonferroni,
)
from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.xstest_450_eval import ARM_ORDER, read_verified_dataset
from src.xstest_latency_repetition import median, summarize
from src.xstest_progress_audit import audit, snapshot_complete_lines

SOURCE_COUNTS = {"safe": (29, 221), "unsafe": (127, 73)}
CONTRASTS = (ARM_ORDER[0], ARM_ORDER[1])
C = ARM_ORDER[2]


def evaluate(rows, dataset):
    """Validate full input first; count prompt-matched C-vs-A/B time signs."""
    baseline = summarize(rows, dataset)
    if baseline["n_generations"] != 1350:
        raise ValueError("requires the complete original XSTest panel")
    keyed = {(item["id"], item["arm"]): item for item in rows}
    source = {case["id"]: case for case in dataset}
    counts = Counter(keyed[(case, C)]["response"] for case in source)
    repeated_text, repetition_count = counts.most_common(1)[0]
    if repetition_count != 156:
        raise ValueError("changed repeated C output")
    diagnostics = []
    for label in ("safe", "unsafe"):
        for mode_index, modal in enumerate((True, False)):
            ids = [case for case, metadata in source.items()
                   if metadata["class"] == label and
                   (keyed[(case, C)]["response"] == repeated_text) == modal]
            expected = SOURCE_COUNTS[label][mode_index]
            if len(ids) != expected:
                raise ValueError("C mode membership or source class changed")
            for other in CONTRASTS:
                deltas = [
                    keyed[(case, C)]["elapsed_seconds"] -
                    keyed[(case, other)]["elapsed_seconds"]
                    for case in ids
                ]
                faster = sum(x < 0 for x in deltas)
                slower = sum(x > 0 for x in deltas)
                tied = len(ids) - faster - slower
                if faster + slower + tied != expected:
                    raise AssertionError("matched prompt accounting mismatch")
                diagnostics.append({
                    "prompt_class": label,
                    "C_response_group": "modal_repeated" if modal else "other",
                    "paired_comparison": f"{C}_minus_{other}",
                    "n_matched_prompts": expected,
                    "C_faster_count": faster,
                    "C_slower_count": slower,
                    "recorded_time_ties": tied,
                    "C_faster_fraction": round(faster / expected, 6),
                    "median_paired_delta_seconds": median(deltas),
                    "C_median_recorded_elapsed_seconds": median([
                        keyed[(case, C)]["elapsed_seconds"] for case in ids
                    ]),
                    "other_median_recorded_elapsed_seconds": median([
                        keyed[(case, other)]["elapsed_seconds"] for case in ids
                    ]),
                    "C_median_whitespace_words_per_response": median([
                        len(keyed[(case, C)]["response"].split()) for case in ids
                    ]),
                    "other_median_whitespace_words_per_response": median([
                        len(keyed[(case, other)]["response"].split()) for case in ids
                    ]),
                    "exploratory_two_sided_sign_p": exact_two_sided_discordant_p(
                        faster, slower),
                })
    if len(diagnostics) != 8:
        raise AssertionError("four strata and two contrasts required")
    adjusted = holm_bonferroni([
        item["exploratory_two_sided_sign_p"] for item in diagnostics])
    for item, p in zip(diagnostics, adjusted):
        item["holm_adjusted_p_across_eight_posthoc_tests"] = p
    return {
        "evidence_class":
            "post-hoc C-output-conditioned paired elapsed-time diagnostics",
        "source_prompts": 450,
        "frozen_generations": 1350,
        "modal_C_total_count": 156,
        "modal_C_safe_count": 29,
        "modal_C_unsafe_count": 127,
        "comparisons": diagnostics,
        "response_group_posthoc_selected_on_C_output": True,
        "population_refusal_or_safety_rate_proven": False,
        "causal_speed_advantage_proven": False,
        "decode_tokens_per_second_measured": False,
        "human_behavioral_safety_review_complete": False,
        "checkpoint_release_authorized": False,
        "limitations": [
            "C mode groups were selected after observing C outputs; no causal estimates",
            "Counts are from the SAME prompts across A, B and C, not randomized trials",
            "Elapsed time includes prompt processing, generation and overhead",
            "No original token counts, finish reasons or per-token timings were recorded",
            "C is pretrained while A/B are instruction variants and prompts differ in produced length",
            "Exact sign tests are exploratory, post-hoc and Holm-adjusted within eight contrasts",
            "Identical keyword-positive response is not an appropriate-refusal human label",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("immutable aggregate cannot be replaced")
    progress = audit(args.results_dir)
    if (progress["recorded_generations"] != 1350 or
        progress["journal_integrity"] != "passed" or
        progress["ignored_unterminated_tail"] or
        progress["checkpoint_sha256"] != checked_arm_paths(ARMS)):
        raise ValueError("frozen checkpoint or journal integrity failed")
    rows, partial, sha = snapshot_complete_lines(
        args.results_dir / "private_generations.jsonl")
    if partial or sha != progress["journal_complete_lines_sha256"]:
        raise ValueError("complete source journal changed during snapshot")
    record = evaluate(rows, read_verified_dataset())
    record["private_journal_sha256"] = sha
    record["benchmark_sha256"] = progress["benchmark_sha256"]
    record["checkpoint_sha256"] = progress["checkpoint_sha256"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "evidence_class": record["evidence_class"],
        "n_comparisons": len(record["comparisons"]),
        "source_prompts": record["source_prompts"],
        "model_safety_verified": False,
        "causal_speed_verified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
