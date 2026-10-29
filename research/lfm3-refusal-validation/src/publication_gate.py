"""Read-only release-readiness checklist for the frozen 3-checkpoint research.

Never authorizes model publication, deployment, or scientific claims itself.
"""
import argparse
import json
from pathlib import Path

from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.xstest_progress_audit import audit
from src.xstest_450_eval import SOURCE_SHA256
from src.blinded_xstest_review import read_verified_dataset
from src.reviewed_xstest_metrics import read_review
from src.paired_reviewed_effects import consensus_rows, paired_report

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARC_TEST_COUNT = 1172
ARC_SHA256 = "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9"


def capability_evidence_valid(report, hashes):
    """Evidence format validation, not an independent scientific certification."""
    if not isinstance(report, dict):
        return False
    if report.get("evaluation") != "ARC-Challenge official full test":
        return False
    if report.get("dataset_sha256") != ARC_SHA256:
        return False
    if report.get("source_split") != "test":
        return False
    if report.get("checkpoint_sha256") != hashes:
        return False
    per_arm = report.get("per_arm", {})
    if set(per_arm) != set(ARMS):
        return False
    for v in per_arm.values():
        if (not isinstance(v, dict) or v.get("total") != EXPECTED_ARC_TEST_COUNT or
                not isinstance(v.get("correct"), int) or isinstance(v.get("correct"), bool) or
                not (0 <= v["correct"] <= v["total"])):
            return False
    if report.get("scoring_method") not in ("normalized conditional loglikelihood",
                                             "verified standard ARC evaluation"):
        return False
    return True


def human_review_check(review_dir, benchmark):
    review_dir = Path(review_dir)
    names = ("blinded_reviewer_1.csv", "blinded_reviewer_2.csv", "private_model_key.json")
    if not all((review_dir / x).is_file() for x in names):
        return {"status":"pending", "reason":"full XSTest human-review packet and grades missing"}
    try:
        a = read_review(review_dir / names[0])
        b = read_review(review_dir / names[1])
        key = json.loads((review_dir / names[2]).read_text())
        consensus = consensus_rows(a, b, key, benchmark, require_full=True)
        # Validate that the estimator can operate before approving the checklist.
        paired_report(consensus, benchmark, resamples=200, require_full=True)
    except (ValueError, KeyError, TypeError) as exc:
        return {"status":"blocked", "reason":"human review incomplete or unadjudicated"}
    return {"status":"complete", "reviewed_responses":len(key),
            "two_independent_consensus_reviewers":True}


def readiness_status(full_results_dir, review_dir, capability_report, hashes, benchmark):
    progress = audit(full_results_dir)
    model_match = (progress.get("checkpoint_sha256") == hashes and len(hashes) == 3)
    full = (model_match and
            progress["recorded_generations"] == 1350 and
            progress["evidence_status"] == "GENERATIONS COMPLETE; HUMAN REVIEW PENDING" and
            progress["journal_integrity"] == "passed" and
            not progress.get("ignored_unterminated_tail", False))
    results = {
        "checkpoint_identity": {
            "status":"complete" if model_match else "blocked",
            "checkpoint_count":len(hashes),
            "journal_matches_local_model_sha256":model_match,
        },
        "xstest_inference": {
            "status":"complete" if full else "incomplete",
            "complete_generations":progress["recorded_generations"],
            "required_generations":1350,
            "journal_integrity":progress["journal_integrity"],
        },
        "human_review": {"status":"blocked",
            "reason":"full inference must be complete before review evidence can count"},
        "capability_validation": {"status":"pending",
            "reason":"official ARC 1172-question three-arm evaluation missing"},
        "responsible_publication_review": {"status":"requires separate independent approval"},
    }
    if full:
        results["human_review"] = human_review_check(review_dir, benchmark)
    p = Path(capability_report)
    if p.is_file():
        try:
            if capability_evidence_valid(json.loads(p.read_text()),hashes):
                results["capability_validation"]={"status":"evidence_present",
                    "reason":"requires independent verification and order-robustness check"}
            else:
                results["capability_validation"]={"status":"blocked",
                    "reason":"full capability evidence provenance invalid"}
        except (ValueError, OSError):
            results["capability_validation"]={"status":"blocked",
                "reason":"capability evidence cannot be parsed"}
    results["publication_approved"] = False
    results["release_approved"] = False
    results["model_count_frozen"] = 3
    results["paper_claims_allowed"] = "methods and clearly qualified preliminary evidence only"
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-results-dir", type=Path,
                   default=ROOT/"results/xstest_full_20261029_v1")
    p.add_argument("--review-dir", type=Path,
                   default=ROOT/"results/xstest_full_review_private")
    p.add_argument("--capability-report", type=Path,
                   default=ROOT/"results/arc_official_full_private.json")
    args = p.parse_args()
    hashes=checked_arm_paths(ARMS)
    benchmark={r["id"]:r for r in read_verified_dataset()}
    summary=readiness_status(args.full_results_dir,args.review_dir,
                             args.capability_report,hashes,benchmark)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
