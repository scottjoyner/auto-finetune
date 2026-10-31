"""Descriptive completion-length confounding audit for frozen A/B/C XSTest.

Word/character lengths and keyword cross-tabs are NOT behavioral refusal,
safe compliance, helpfulness, or causal effects. Private text stays local.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from src.arc_paired_exact_inference import (
    exact_two_sided_discordant_p, holm_bonferroni,
)
from src.eval_contracts import refusal_proxy
from src.three_checkpoint_eval import checked_arm_paths, ARMS
from src.xstest_450_eval import read_verified_dataset, ARM_ORDER
from src.xstest_progress_audit import audit, snapshot_complete_lines

STRATA = {"safe": 250, "unsafe": 200}
PAIRS = ((ARM_ORDER[0], ARM_ORDER[1]), (ARM_ORDER[0], ARM_ORDER[2]),
         (ARM_ORDER[1], ARM_ORDER[2]))
SHORT_CHARS = 50


def percentile(values, fraction):
    """Discrete nearest-rank statistic without interpolating between samples."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("invalid percentile input")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def word_count(response):
    return len(response.split())
def summarize(records, dataset):
    if len(dataset) != 450 or len(records) != 1350:
        raise ValueError("incomplete 450 by three journal")
    cases = {}
    for item in dataset:
        ident = item["id"]
        if ident in cases or item.get("class") not in STRATA:
            raise ValueError("duplicate ID or bad source class")
        cases[ident] = item
    if Counter(x["class"] for x in cases.values()) != STRATA:
        raise ValueError("unexpected source class totals")
    by_key = {}
    for item in records:
        ident = item.get("id")
        arm = item.get("arm")
        if ident not in cases or arm not in ARM_ORDER:
            raise ValueError("unknown model/case")
        key = (ident, arm)
        if key in by_key:
            raise ValueError("duplicate model/case generation")
        source = cases[ident]
        if item.get("class") != source["class"] or item.get("type") != source["type"]:
            raise ValueError("untrusted source class or type")
        response = item.get("response")
        if not isinstance(response, str) or type(item.get("refusal_proxy")) is not bool:
            raise ValueError("missing completion or lexical indicator")
        if refusal_proxy(response) != item["refusal_proxy"]:
            raise ValueError("stored lexical keyword flag differs from source regex")
        by_key[key] = item
    if len(by_key) != len(cases) * len(ARM_ORDER):
        raise ValueError("missing matched model arm")
    result = {}
    comparisons = []
    for group, expected in STRATA.items():
        case_ids = [case for case, item in cases.items() if item["class"] == group]
        if len(case_ids) != expected:
            raise ValueError("wrong stratum denominator")
        per_arm = {}
        for arm in ARM_ORDER:
            rows = [by_key[(case, arm)] for case in case_ids]
            characters = [len(row["response"]) for row in rows]
            words = [word_count(row["response"]) for row in rows]
            short = [row for row in rows if len(row["response"]) < SHORT_CHARS]
            long = [row for row in rows if len(row["response"]) >= SHORT_CHARS]
            per_arm[arm] = {
                "n": expected,
                "median_characters": statistics.median(characters),
                "p10_characters": percentile(characters, .1),
                "p90_characters": percentile(characters, .9),
                "median_words": statistics.median(words),
                "p10_words": percentile(words, .1),
                "p90_words": percentile(words, .9),
                "short_response_less_than_50_characters": len(short),
                "short_response_keyword_positive": sum(row["refusal_proxy"] for row in short),
                "long_response_at_least_50_characters": len(long),
                "long_response_keyword_positive": sum(row["refusal_proxy"] for row in long),
                "total_keyword_positive": sum(row["refusal_proxy"] for row in rows),
                "empty_response": sum(not row["response"].strip() for row in rows),
            }
        result[group] = {"n_prompts": expected, "per_arm": per_arm}
        for left, right in PAIRS:
            diffs = [
                word_count(by_key[(case, right)]["response"]) -
                word_count(by_key[(case, left)]["response"])
                for case in case_ids
            ]
            greater = sum(delta > 0 for delta in diffs)
            lower = sum(delta < 0 for delta in diffs)
            ties = sum(delta == 0 for delta in diffs)
            comparisons.append({
                "stratum": group, "left": left, "right": right, "n_paired": expected,
                "right_longer_words": greater, "right_shorter_words": lower,
                "word_count_ties": ties,
                "median_paired_word_delta_right_minus_left": statistics.median(diffs),
                "exploratory_exact_sign_p": exact_two_sided_discordant_p(greater, lower),
                "exact_string_matches": sum(
                    by_key[(case, left)]["response"] == by_key[(case, right)]["response"]
                    for case in case_ids),
            })
    adjusted = holm_bonferroni([x["exploratory_exact_sign_p"] for x in comparisons])
    for row, p in zip(comparisons, adjusted):
        row["holm_six_adjusted_p"] = p
    return {
        "evidence_class": "full XSTest completion-length and lexical-indicator confounding audit",
        "n_prompts": 450, "n_generations": 1350,
        "short_response_definition": "strictly fewer than 50 Unicode code points (Python len)",
        "word_definition": "whitespace-delimited tokens, NOT model-generated tokenizer tokens",
        "strata": result,
        "paired_word_length_diagnostics": comparisons,
        "human_behavioral_refusals_verified": False,
        "causal_length_effect_established": False,
        "checkpoint_release_approved": False,
        "limitations": [
            "Post-generation length is an outcome, not a causal adjustment covariate",
            "Length and keyword incidence cannot grade actual refusal or harmful actionability",
            "All results reuse 128-max-token single-turn XSTest English completions",
            "Sign tests measure length direction only, not quality or magnitude",
            "Pretrained C is not instruction-tuned, so arm contrasts remain confounded",
            "Two blinded independent human reviewers are not completed",
        ],
    }
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("do not overwrite earlier aggregate")
    progress = audit(a.results_dir)
    if (progress["recorded_generations"] != 1350
            or progress["journal_integrity"] != "passed"
            or progress["ignored_unterminated_tail"]):
        raise ValueError("not a complete, authenticated journal")
    hashes = checked_arm_paths(ARMS)
    if hashes != progress["checkpoint_sha256"]:
        raise ValueError("checkpoint digest differs from frozen manifest")
    rows, partial, digest = snapshot_complete_lines(
        a.results_dir / "private_generations.jsonl")
    if partial or digest != progress["journal_complete_lines_sha256"]:
        raise ValueError("journal changed during snapshot")
    outcome = summarize(rows, read_verified_dataset())
    outcome["private_journal_sha256"] = digest
    outcome["benchmark_sha256"] = progress["benchmark_sha256"]
    outcome["checkpoint_sha256"] = hashes
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(outcome, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "n_generations": outcome["n_generations"],
        "strata": outcome["strata"],
        "paired_word_length_diagnostics": outcome["paired_word_length_diagnostics"],
        "human_behavioral_refusals_verified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
