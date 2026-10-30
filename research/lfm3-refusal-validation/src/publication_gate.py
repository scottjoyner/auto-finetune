"""Read-only release-readiness checklist for the frozen 3-checkpoint research.

Never authorizes model publication, deployment, or scientific claims itself.
"""
import argparse
import json
import math
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



def local_full_arc_check(path, hashes):
    """Recognize full *locally adapted* ARC scores without releasing model gates."""
    path=Path(path) if path is not None else ROOT/"evidence/20261030_arc_local_full1172_cpu_aggregate.json"
    if not path.is_file():
        return {"status":"pending",
                "reason":"complete local-parquet ARC three-arm aggregate not present"}
    try:
        report=json.loads(path.read_text())
        if (report.get("evidence_class") !=
                "actual full 1172-case locally pinned ARC-Challenge multiple-choice likelihood scoring"
                or report.get("n_questions_per_arm")!=EXPECTED_ARC_TEST_COUNT
                or report.get("checkpoint_count")!=3
                or report.get("checkpoint_sha256")!=hashes
                or report.get("dataset_sha256")!=ARC_SHA256
                or report.get("local_parquet_adapter_used") is not True
                or report.get("canonical_unmodified_hub_task_used") is not False
                or report.get("source_documents_verified_against_pinned_parquet") is not True
                or report.get("human_refusal_review_complete") is not False
                or report.get("model_release_approved") is not False
                or len(report.get("matched_question_comparisons",[]))!=6):
            raise ValueError("full local ARC aggregate contract mismatch")
        scores=report["metrics"]
        if set(scores)!=set(ARMS):
            raise ValueError("full local ARC arm identities incomplete")
        counts={}
        for arm in ARMS:
            counts[arm]={}
            for metric in ("acc","acc_norm"):
                val=scores[arm][metric]
                correct=val["correct"]
                fraction=val.get("fraction")
                if (type(correct) is not int or not 0<=correct<=EXPECTED_ARC_TEST_COUNT
                        or val.get("n")!=EXPECTED_ARC_TEST_COUNT
                        or type(fraction) not in (float,int)
                        or not math.isfinite(fraction)
                        or abs(fraction-correct/EXPECTED_ARC_TEST_COUNT)>1e-10):
                    raise ValueError("inconsistent full local ARC denominators")
                counts[arm][metric]=correct
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError):
        return {"status":"blocked","reason":"invalid or inconsistent local full ARC aggregate"}
    return {"status":"complete","questions_per_checkpoint":EXPECTED_ARC_TEST_COUNT,
            "n_checkpoints":3,"correct_case_counts":counts,
            "canonical_official_benchmark":False,
            "independent_external_replication":False,
            "human_safety_review":False,
            "release_approved":False,
            "reason":"full source-pinned local-parquet ARC scoring, not canonical Hub-task validation"}


def canonical_full_arc_check(path, hashes):
    """Recognize separately audited canonical ARC evidence, never approve safety."""
    path=Path(path) if path is not None else ROOT/"evidence/20261030_arc_canonical_full1172_cpu_aggregate.json"
    if not path.is_file():
        return {"status":"pending","reason":"unmodified built-in ARC three-arm report is absent"}
    try:
        from src.arc_canonical_full_report import CANONICAL_ROWS_HASH
        report=json.loads(path.read_text())
        if (report.get("evidence_class") !=
                "real full canonical EleutherAI arc_challenge 3-arm zero-shot evaluation"
                or report.get("harness_version")!="0.4.13"
                or report.get("n_questions_per_arm")!=EXPECTED_ARC_TEST_COUNT
                or report.get("checkpoint_sha256")!=hashes
                or report.get("dataset_content_sha256")!=CANONICAL_ROWS_HASH
                or report.get("canonical_builtin_task_used") is not True
                or report.get("hub_dataset_revision_content_matches_pinned_source") is not True
                or report.get("human_refusal_review_complete") is not False
                or report.get("model_release_approved") is not False
                or report.get("publication_approved") is not False
                or len(report.get("matched_question_comparisons",[]))!=6
                or set(report.get("metrics",{}))!=set(ARMS)):
            raise ValueError("unverified canonical benchmark provenance")
        per_arm={}
        for arm in ARMS:
            per_arm[arm]={}
            for metric in ("acc","acc_norm"):
                item=report["metrics"][arm][metric]
                k=item["correct"]
                value=item.get("fraction")
                stderr=item.get("stderr")
                if (type(k) is not int or not 0<=k<=EXPECTED_ARC_TEST_COUNT
                    or item.get("n")!=EXPECTED_ARC_TEST_COUNT
                    or type(value) not in (int,float) or not math.isfinite(value)
                    or abs(value-k/EXPECTED_ARC_TEST_COUNT)>1e-10
                    or type(stderr) not in (int,float) or not math.isfinite(stderr)
                    or not 0<=stderr<=1):
                    raise ValueError("invalid canonical arm metrics")
                per_arm[arm][metric]=k
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"status":"blocked","reason":"inconsistent canonical ARC evidence"}
    return {
        "status":"evidence_present",
        "n_questions_per_checkpoint":EXPECTED_ARC_TEST_COUNT,
        "correct_case_counts":per_arm,
        "canonical_builtin_task":True,
        "independent_external_replication":False,
        "human_refusal_review_complete":False,
        "release_approved":False,
        "reason":"full canonical task aggregate audited; separate reproducibility and human safety signoff pending",
    }

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


def custom_arc_check(results_dir, hashes):
    """Recognize finished custom ARC likelihood, never call it official validation."""
    if results_dir is None or not (Path(results_dir)/"private_manifest.json").is_file():
        return {"status":"pending","reason":"three-arm custom ARC likelihood evidence missing"}
    try:
        from src.arc_likelihood_progress_audit import audit as arc_audit
        state=arc_audit(results_dir,verify_live_weights=False)
        if (state.get("checkpoint_sha256")!=hashes or
                state.get("journal_integrity")!="passed" or
                state.get("recorded_cases")!=3516 or
                state.get("expected_cases")!=3516 or
                state.get("completion")!="generations_complete" or
                state.get("unterminated_tail_present") or
                any(state["per_arm"][arm]["recorded"]!=1172 for arm in ARMS)):
            raise ValueError("incomplete or mismatched custom ARC run")
    except (OSError, ValueError, KeyError, TypeError):
        return {"status":"blocked","reason":"custom likelihood run not complete or provenance mismatch"}
    return {
        "status":"complete",
        "evaluated_case_model_pairs":3516,
        "questions_per_checkpoint":1172,
        "official_benchmark":False,
        "reason":"custom answer-choice likelihood; independent standard benchmarking still needed"
    }


def readiness_status(full_results_dir, review_dir, capability_report, hashes, benchmark, custom_arc_dir=None, local_arc_report=None, canonical_arc_report=None):
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
    results["arc_custom_likelihood"] = custom_arc_check(custom_arc_dir, hashes)
    results["arc_full_local_pinned"] = local_full_arc_check(
        local_arc_report,hashes)
    results["arc_full_canonical"] = canonical_full_arc_check(
        canonical_arc_report,hashes)
    if results["arc_full_canonical"]["status"]=="evidence_present":
        results["capability_validation"]={
            "status":"evidence_present",
            "reason":"canonical ARC result verified structurally; independent external replication still required",
        }
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
    p.add_argument("--arc-custom-dir",type=Path,
                   default=ROOT/"results/arc_choice_likelihood_full_private")
    p.add_argument("--arc-local-full-report",type=Path,
                   default=ROOT/"evidence/20261030_arc_local_full1172_cpu_aggregate.json")
    p.add_argument("--arc-canonical-full-report",type=Path,
                   default=ROOT/"evidence/20261030_arc_canonical_full1172_cpu_aggregate.json")
    args = p.parse_args()
    hashes=checked_arm_paths(ARMS)
    benchmark={r["id"]:r for r in read_verified_dataset()}
    summary=readiness_status(args.full_results_dir,args.review_dir,
                             args.capability_report,hashes,benchmark,
                             args.arc_custom_dir,args.arc_local_full_report,
                             args.arc_canonical_full_report)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
