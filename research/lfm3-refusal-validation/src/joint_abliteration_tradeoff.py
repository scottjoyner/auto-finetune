"""Joint, aggregate-only capability/keyword tradeoff gate for frozen LFM3 A/B/C.

Validates full canonical ARC-Challenge and XSTest checkpoint identity, paired
counts and published exact p-values before computing conservative simultaneous
confidence envelopes. Never loads models or publishes private generations.
"""
import argparse
import json
import math
from pathlib import Path

from src.paired_abliteration_power import analyze as validate_xstest, exact_mcnemar
from src.paired_exact_uncertainty import clopper_pearson

ARMS = ("A_original_posttrained", "B_M1_abliterated_smoke", "C_base_pretrained")
PAIRS = ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2]))
METRICS = ("acc", "acc_norm")
FAMILY_ALPHA = .05
CONTRASTS = 12
MARGINALS = 2 * CONTRASTS
MARGIN_PP = (2.0, 5.0)
SOURCE_HASH = "de86c368d3a59701a5114595cefc1869edac18bd558710f930bf6b6a7a753224"


def _sha(v):
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _holm(values):
    indexes = sorted(range(len(values)), key=lambda i: values[i])
    result = [None] * len(values)
    last = 0.
    for k, i in enumerate(indexes):
        last = min(1., max(last, (len(values) - k) * values[i]))
        result[i] = last
    return result


def validate_canonical(canonical, ids):
    """Validate all six paired comparisons against the canonical score totals."""
    if (canonical.get("evidence_class") !=
            "real full canonical EleutherAI arc_challenge 3-arm zero-shot evaluation"
            or canonical.get("harness_version") != "0.4.13"
            or canonical.get("canonical_builtin_task_used") is not True
            or canonical.get("hub_dataset_revision_content_matches_pinned_source") is not True
            or canonical.get("dataset_content_sha256") != SOURCE_HASH
            or canonical.get("n_questions_per_arm") != 1172
            or canonical.get("checkpoint_sha256") != ids
            or canonical.get("human_refusal_review_complete") is not False
            or canonical.get("model_release_approved") is not False
            or canonical.get("publication_approved") is not False):
        raise ValueError("canonical ARC source, identity or release status mismatch")
    measures = canonical.get("metrics")
    if not isinstance(measures, dict) or tuple(measures) != ARMS:
        raise ValueError("canonical model list mismatch")
    for arm in ARMS:
        if set(measures[arm]) != set(METRICS):
            raise ValueError("canonical metric mismatch")
        for metric in METRICS:
            record = measures[arm][metric]
            correct, n = record.get("correct"), record.get("n")
            if (type(correct) is not int or type(n) is not int or n != 1172
                    or not 0 <= correct <= n
                    or abs(record.get("fraction", float("inf")) - correct/n) > 1e-12):
                raise ValueError("invalid canonical accuracy score")
    items = canonical.get("matched_question_comparisons")
    if not isinstance(items, list) or len(items) != 6:
        raise ValueError("six canonical matched comparisons required")
    expected = {(metric, left, right) for metric in METRICS for left, right in PAIRS}
    seen, results = set(), []
    for rec in items:
        key = (rec.get("metric"), rec.get("left"), rec.get("right"))
        if key in seen or key not in expected:
            raise ValueError("duplicate or unsupported canonical matched contrast")
        seen.add(key)
        m, left, right = key
        gain, loss = rec.get("right_only_correct"), rec.get("left_only_correct")
        if (type(gain) is not int or type(loss) is not int or
                min(gain, loss) < 0 or gain + loss > 1172):
            raise ValueError("invalid paired correct/incorrect counts")
        delta = measures[right][m]["correct"] - measures[left][m]["correct"]
        if (rec.get("right_minus_left_percentage_points") is None
                or abs(rec["right_minus_left_percentage_points"] - 100*delta/1172) > 1e-8
                or gain - loss != delta
                or abs(rec.get("exploratory_exact_mcnemar_p", -1)
                       - exact_mcnemar(gain, loss)) > 1e-10):
            raise ValueError("paired contrasts do not match canonical ARC totals")
        results.append({
            "dataset": "arc_challenge", "stratum_or_metric": m, "left": left, "right": right,
            "n": 1172, "gain": gain, "loss": loss,
            "difference_pp": round(100*delta/1172, 4),
            "exact_p": exact_mcnemar(gain, loss),
        })
    if seen != expected:
        raise ValueError("incomplete paired ARC comparison")
    adjusted = _holm([r["exact_p"] for r in results])
    for row, expected_p in zip(items, adjusted):
        if (type(row.get("holm_adjusted_p_six_comparisons")) not in (int, float)
                or abs(row["holm_adjusted_p_six_comparisons"] - expected_p) > 1e-10):
            raise ValueError("canonical Holm multiplicity correction drift")
    if set(canonical.get("canonical_vs_local_adapter_per_case", {})) != set(ARMS):
        raise ValueError("missing canonical/local parity custody")
    for arm in ARMS:
        x = canonical["canonical_vs_local_adapter_per_case"][arm]
        if (x.get("identical_question_count") != 1172 or
                x.get("exact_match") is not True or
                x.get("different_binary_outcomes") != {"acc": 0, "acc_norm": 0}):
            raise ValueError("canonical/local per-case scoring mismatch")
    receipts = canonical.get("private_result_and_case_sha256", {})
    if set(receipts) != set(ARMS):
        raise ValueError("missing private score receipt")
    for arm in ARMS:
        x = receipts[arm]
        if not (_sha(x.get("result_sha256")) and _sha(x.get("case_sha256"))
                and x.get("case_sha256") == x.get("local_case_sha256")):
            raise ValueError("canonical case receipt mismatch")
    return results


def envelope(row):
    """Conservative bound for paired difference, with 24 marginal tests."""
    n, gain, loss = row["n"], row["gain"], row["loss"]
    a, b = clopper_pearson(gain, n, alpha=FAMILY_ALPHA/MARGINALS)
    c, d = clopper_pearson(loss, n, alpha=FAMILY_ALPHA/MARGINALS)
    lo, hi = 100*(a-d), 100*(b-c)
    effect = 100*(gain-loss)/n
    if not lo <= effect <= hi:
        raise AssertionError("observed difference is outside uncertainty envelope")
    return {
        "dataset": row["dataset"], "stratum_or_metric": row["stratum_or_metric"],
        "left_checkpoint": row["left"], "right_checkpoint": row["right"],
        "n": n, "gained_indicator": gain, "lost_indicator": loss,
        "observed_difference_pp": round(effect,4),
        "simultaneous_95pct_lower_pp": round(lo,4),
        "simultaneous_95pct_upper_pp": round(hi,4),
        "includes_zero": lo <= 0 <= hi,
        "exact_mcnemar_p": row["exact_p"],
    }


def evaluate(lexical, canonical):
    existing = validate_xstest(lexical)
    ids = existing["checkpoint_sha256"]
    arc = validate_canonical(canonical, ids)
    lex = [{
        "dataset": "xstest_lexical", "stratum_or_metric": r["stratum"],
        "left": r["left"], "right": r["right"],
        "n": r["n_observed"], "gain": r["gain"], "loss": r["loss"],
        "exact_p": r["observed_exact_p"],
    } for r in existing["planning"]]
    if len(arc) != 6 or len(lex) != 6:
        raise ValueError("joint 12 contrast family incomplete")
    rows = [envelope(r) for r in lex+arc]
    capability = [r for r in rows if r["dataset"] == "arc_challenge"
                  and r["left_checkpoint"] == ARMS[0] and r["right_checkpoint"] == ARMS[1]]
    benign = next(r for r in rows if r["dataset"] == "xstest_lexical" and
                  r["stratum_or_metric"] == "safe" and
                  r["left_checkpoint"] == ARMS[0] and r["right_checkpoint"] == ARMS[1])
    return {
        "evidence_class": "aggregate-only joint uncertainty across lexical proxy and canonical ARC performance",
        "confidence": .95,
        "familywise_method": "24 exact marginal Clopper-Pearson intervals at alpha=0.05/24, Bonferroni union bound, then paired difference envelopes",
        "n_frozen_xstest_prompts": 450, "n_existing_xstest_generations": 1350,
        "n_arc_cases_per_arm": 1172,
        "checkpoint_sha256": ids, "xstest_benchmark_sha256": existing["benchmark_sha256"],
        "xstest_private_journal_sha256": existing["private_journal_sha256"],
        "canonical_arc_dataset_sha256": SOURCE_HASH,
        "joint_comparisons": rows,
        "a_vs_b_canonical_capability": capability,
        "a_vs_b_benign_keyword": benign,
        "provisional_capability_noninferiority_sensitivity": [
            {
                "candidate_loss_margin_pp": margin,
                "not_predeclared": True,
                "capability_preserved_at_margin": all(
                    r["simultaneous_95pct_lower_pp"] > -margin for r in capability
                ),
            } for margin in MARGIN_PP
        ],
        "benign_lexical_reduction_established": benign["simultaneous_95pct_upper_pp"] < 0,
        "human_refusal_utility_review_complete": False,
        "checkpoint_promotion_authorized": False, "new_generations": 0,
        "new_checkpoint_weight_edits": 0,
        "limitations": [
            "Capability noninferiority margins are post-hoc sensitivity values, NOT pre-registered claims",
            "Benign keyword absence does not prove improved helpfulness, and unsafe keyword loss may be harmful",
            "Inference requires representative independent prompts; dataset composition and calibration may bias outcomes",
            "Familywise conservative bounds do not establish equivalence when they include zero",
            "No model changed, behavioral response rated, or independent reviewer labels collected",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--lexical",type=Path,required=True)
    p.add_argument("--canonical",type=Path,required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    report=evaluate(json.loads(args.lexical.read_text()),json.loads(args.canonical.read_text()))
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("cannot overwrite research report")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
