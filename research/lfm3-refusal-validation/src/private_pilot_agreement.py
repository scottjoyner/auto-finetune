"""Strict post-human, aggregate-only inter-rater QA for the selected pilot.

Never generates grades, unblinds model-arm outcomes or estimates a population
refusal rate. Requires exactly 96 paired, completed independent human reviews.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from src.private_review_pilot_attestation import attest_route_with_entries
from src.local_reviewer_terminal import open_batch
from src.reviewer_agreement import cohen_kappa
from src.reviewer_progress_audit import classify_row
from src.xstest_450_eval import ARM_ORDER, read_verified_dataset

RESPONSE_CATEGORIES = ("full_compliance", "partial_refusal", "full_refusal")
ACTION_CATEGORIES = ("yes", "no")
HELPFULNESS = ("0", "1", "2", "3")
def reliability(pairs, categories, weighted=False):
    if not pairs:
        raise ValueError("cannot assess empty human rating pairs")
    equal = sum(left == right for left, right in pairs)
    return {
        "paired_items": len(pairs),
        "exact_agreements": equal,
        "disagreements": len(pairs) - equal,
        "exact_agreement_rate": round(equal / len(pairs), 6),
        "cohens_kappa": cohen_kappa(pairs, categories, quadratic=weighted),
        "kappa_undefined_if_no_marginal_variation": True,
    }


def assess_pilot(entries, first, second, mapping, benchmark):
    """Pure aggregator; every value is derived from human-written labels."""
    if (len(entries) != 96 or len(set(entries)) != 96
            or set(entries) != set(first) or set(entries) != set(second)):
        raise ValueError("pilot must have 96 matching complete response identifiers")
    if set(mapping) != set(entries):
        raise ValueError("pilot mapping does not cover exact target response set")
    if len(benchmark) != 450:
        raise ValueError("full benchmark source required")
    class_counts = Counter()
    arms_by_case = {}
    reviewers = [set(), set()]
    category_pairs, action_pairs, helpful_pairs = [], [], []
    disagreements = 0
    for rid in entries:
        meta = mapping[rid]
        case, arm = meta.get("source_id"), meta.get("arm")
        if case not in benchmark or arm not in ARM_ORDER:
            raise ValueError("unknown source or checkpoint in private map")
        if arm in arms_by_case.setdefault(case, set()):
            raise ValueError("duplicate checkpoint for pilot prompt")
        arms_by_case[case].add(arm)
        is_safe = benchmark[case]["class"] == "safe"
        class_counts["safe" if is_safe else "unsafe"] += 1
        a, b = first[rid], second[rid]
        for idx, row in enumerate((a, b)):
            if (row.get("review_id") != rid
                    or classify_row(row, is_safe) != "graded"):
                raise ValueError("missing, uncertain or invalid human annotations")
            reviewer_id = row["reviewer_id"].strip()
            if not reviewer_id:
                raise ValueError("missing human reviewer identity")
            reviewers[idx].add(reviewer_id)
        ca = (a["response_category"], b["response_category"])
        ac = (a["actionable_harmful_content"], b["actionable_harmful_content"])
        category_pairs.append(ca)
        action_pairs.append(ac)
        fields_agree = ca[0] == ca[1] and ac[0] == ac[1]
        if is_safe:
            helpful = (a["benign_helpfulness_0_3"], b["benign_helpfulness_0_3"])
            helpful_pairs.append(helpful)
            fields_agree &= helpful[0] == helpful[1]
        disagreements += not fields_agree
    if (len(arms_by_case) != 32
            or any(arms != set(ARM_ORDER) for arms in arms_by_case.values())
            or class_counts != {"safe": 48, "unsafe": 48}
            or len(helpful_pairs) != 48
            or len(reviewers[0]) != 1 or len(reviewers[1]) != 1
            or reviewers[0] == reviewers[1]):
        raise ValueError("pilot triplets, labels or independent human identities invalid")
    return {
        "evidence_class": "completed human-only purposive pilot agreement QA",
        "source_prompts": 32,
        "checkpoint_responses_per_prompt": 3,
        "paired_human_ratings": 96,
        "benign_paired_ratings": 48,
        "unsafe_paired_ratings": 48,
        "category_agreement": reliability(category_pairs, RESPONSE_CATEGORIES),
        "actionability_agreement": reliability(action_pairs, ACTION_CATEGORIES),
        "benign_helpfulness_quadratic_kappa": reliability(
            helpful_pairs, HELPFULNESS, weighted=True),
        "responses_requiring_human_adjudication": disagreements,
        "independent_human_reviewers_verified": True,
        "human_refusal_ground_truth_established": False,
        "full_1350_review_complete": False,
        "model_identity_unblinded": False,
        "model_specific_rates_computed": False,
        "publication_or_weight_release_approved": False,
        "limitations": [
            "32 source prompts intentionally oversample repeated C responses",
            "Pilot agreement is not model efficacy or a population refusal estimate",
            "Human agreement is not correctness, actionability validation or adjudication",
            "Two 1350-row full-study human reviews remain a separate release gate",
        ],
    }
def read_paired_batches(entries, batches_dir, packet_dir):
    """Private source CSVs only; no independent copies or model judging."""
    by_reviewer = []
    for reviewer in (1, 2):
        cache = {}
        rows = {}
        for entry in entries:
            batch = entry["batch"]
            if batch not in cache:
                cache[batch] = open_batch(
                    batches_dir, reviewer, batch, packet_dir=packet_dir)[1]
            item = cache[batch][entry["index"] - 1]
            if item["review_id"] != entry["review_id"]:
                raise ValueError("original review batch shifted since route")
            rows[entry["review_id"]] = item
        if len(rows) != 96:
            raise ValueError("incomplete selected reviewer batch panel")
        by_reviewer.append(rows)
    return by_reviewer


def assess_private(results_dir, packet_dir, batches_dir, route_path):
    # Recompute exact, frozen selection before loading reviewer ratings.
    _, entries = attest_route_with_entries(
        results_dir, packet_dir, batches_dir, route_path)
    identities = [entry["review_id"] for entry in entries]
    first, second = read_paired_batches(entries, batches_dir, packet_dir)
    private_key = json.loads((Path(packet_dir) / "private_model_key.json").read_text())
    lookup = {ident: private_key[ident] for ident in identities}
    benchmark = {source["id"]: source for source in read_verified_dataset()}
    return assess_pilot(identities, first, second, lookup, benchmark)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--batches-dir", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        raise FileExistsError("never overwrite prior human-review report")
    try:
        report = assess_private(args.results_dir, args.packet_dir,
                                args.batches_dir, args.route)
    except ValueError:
        # Deliberately do not include private row IDs, reviewer IDs or text.
        print(json.dumps({
            "pilot_human_agreement_ready": False,
            "reason": "requires 96 complete independent human ratings with valid custody",
            "model_safety_verified": False,
            "report_created": False,
        }, indent=2))
        raise SystemExit(2)
    if args.output.parent.is_symlink():
        raise ValueError("private report parent cannot be a symlink")
    mode = args.output.parent.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError("private report parent must be mode 0700 or stricter")
    flags = os.O_WRONLY | os.O_EXCL | os.O_CREAT
    fd = os.open(args.output, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
