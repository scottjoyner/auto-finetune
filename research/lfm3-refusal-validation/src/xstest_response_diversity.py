"""Private-journal, public-aggregate audit of exact response repetition.

Counts repeated strings within and between XSTest strata without exposing
response text, per-response hashes, sample IDs, or a behavioral safety grade.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from src.eval_contracts import refusal_proxy
from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.xstest_450_eval import ARM_ORDER, read_verified_dataset
from src.xstest_progress_audit import audit, snapshot_complete_lines

EXPECTED = {"safe": 250, "unsafe": 200}


def summarize(rows, dataset):
    if len(rows) != 1350 or len(dataset) != 450:
        raise ValueError("requires complete 450 x 3 frozen journal")
    cases = {}
    for item in dataset:
        ident = item.get("id")
        if (not isinstance(ident, str) or not ident or ident in cases
                or item.get("class") not in EXPECTED):
            raise ValueError("duplicate or invalid benchmark case")
        cases[ident] = item
    if Counter(c["class"] for c in cases.values()) != EXPECTED:
        raise ValueError("wrong benchmark source classes")
    by_key = {}
    for item in rows:
        ident, arm = item.get("id"), item.get("arm")
        if ident not in cases or arm not in ARM_ORDER:
            raise ValueError("unexpected checkpoint or case")
        key = (ident, arm)
        if key in by_key:
            raise ValueError("duplicated checkpoint/case")
        if (item.get("class") != cases[ident]["class"]
                or item.get("type") != cases[ident]["type"]):
            raise ValueError("untrusted class or category")
        response, flag = item.get("response"), item.get("refusal_proxy")
        if not isinstance(response, str) or type(flag) is not bool:
            raise ValueError("missing response or keyword flag")
        if refusal_proxy(response) != flag:
            raise ValueError("keyword proxy mismatches current pinned regex")
        by_key[key] = item
    if len(by_key) != 1350:
        raise ValueError("incomplete paired panel")
    report = {}
    for arm in ARM_ORDER:
        records = [by_key[(ident, arm)] for ident in cases]
        frequency = Counter(x["response"] for x in records)
        max_count = max(frequency.values())
        modes = [text for text, n in frequency.items() if n == max_count]
        # Ambiguous multimodal ties must never be resolved by response text order.
        overall = {
            "n_prompts": 450,
            "distinct_exact_responses": len(frequency),
            "repeated_response_types": sum(n > 1 for n in frequency.values()),
            "responses_belonging_to_repeated_types": sum(
                n for n in frequency.values() if n > 1),
            "largest_identical_response_group": max_count,
            "most_common_group_is_unique": len(modes) == 1,
            "exact_response_collision_probability_with_replacement": round(
                sum(n * n for n in frequency.values()) / (450 * 450), 8),
            "total_keyword_positive": sum(x["refusal_proxy"] for x in records),
        }
        strata = {}
        for group, count in EXPECTED.items():
            subset = [x for x in records if x["class"] == group]
            freq = Counter(x["response"] for x in subset)
            keyword = [x["response"] for x in subset if x["refusal_proxy"]]
            strata[group] = {
                "n": count, "unique_exact_responses": len(freq),
                "largest_identical_group": max(freq.values()),
                "repeated_response_types": sum(n > 1 for n in freq.values()),
                "responses_in_repeated_groups": sum(
                    n for n in freq.values() if n > 1),
                "keyword_positive_responses": len(keyword),
                "distinct_keyword_positive_responses": len(set(keyword)),
                "short_under_50_characters": sum(
                    len(x["response"]) < 50 for x in subset),
                "short_under_50_keyword_positive": sum(
                    len(x["response"]) < 50 and x["refusal_proxy"] for x in subset),
            }
        shared = set(x["response"] for x in records if x["class"] == "safe") & set(
            x["response"] for x in records if x["class"] == "unsafe")
        overall["distinct_responses_shared_between_strata"] = len(shared)
        overall["responses_matching_cross_stratum_strings"] = sum(
            frequency[text] for text in shared)
        if len(modes) == 1:
            modal = modes[0]
            modal_counts = {
                group: sum(x["response"] == modal for x in records
                           if x["class"] == group)
                for group in EXPECTED
            }
            overall["top_mode"] = {
                "safe_count": modal_counts["safe"],
                "unsafe_count": modal_counts["unsafe"],
                "keyword_positive": refusal_proxy(modal),
                "shorter_than_50_characters": len(modal) < 50,
                "total_occurrences": max_count,
            }
        else:
            overall["top_mode"] = None
        if len(frequency) != overall["distinct_exact_responses"]:
            raise AssertionError("frequency accounting mismatch")
        report[arm] = {"overall": overall, "strata": strata}
    return {
        "evidence_class": "frozen XSTest exact output diversity and repetition (not refusal safety)",
        "n_prompts": 450, "n_generations": 1350,
        "checkpoint_data": report,
        "human_behavioral_refusal_verified": False,
        "mode_repetition_implies_refusal_quality": False,
        "checkpoint_promotion_approved": False,
        "limitations": [
            "Exact text repetition is descriptive and may be legitimate in some contexts",
            "A shared refusal-like string across safe/unsafe prompts does not prove safety",
            "Lexical keyword matches never establish justified refusal or harm actionability",
            "Pretrained C is not instruction-tuned, so A/B/C response-style comparisons are confounded",
            "The fixed single-turn generation cap was 128 model new tokens",
            "Mode selection is post hoc; no p-value is inferred for repeated response counts",
            "The report contains no raw model text or per-response digests",
        ],
    }
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("output already exists; evidence is immutable")
    progress = audit(args.results_dir)
    if (progress["recorded_generations"] != 1350
            or progress["journal_integrity"] != "passed"
            or progress["ignored_unterminated_tail"]):
        raise ValueError("invalid or incomplete XSTest source journal")
    if checked_arm_paths(ARMS) != progress["checkpoint_sha256"]:
        raise ValueError("checkpoint content differs from frozen run manifest")
    records, partial, digest = snapshot_complete_lines(
        args.results_dir / "private_generations.jsonl")
    if partial or digest != progress["journal_complete_lines_sha256"]:
        raise ValueError("source journal mismatch")
    output = summarize(records, read_verified_dataset())
    output["checkpoint_sha256"] = progress["checkpoint_sha256"]
    output["benchmark_sha256"] = progress["benchmark_sha256"]
    output["private_journal_sha256"] = digest
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "n_generations": output["n_generations"],
        "checkpoint_data": output["checkpoint_data"],
        "human_behavioral_refusal_verified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
