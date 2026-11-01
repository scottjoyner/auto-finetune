"""Conservative familywise exact-binomial bounds for frozen paired keyword data.

This does not score behavioral refusal or generate model responses. It uses
only prepublished aggregate paired counts after the original hash checks.
"""
import argparse
import json
import math
from pathlib import Path

from src.paired_abliteration_power import analyze

FAMILY_ALPHA = 0.05
N_COMPARISONS = 6
N_MARGINALS = 2 * N_COMPARISONS


def _cdf(n, k, p):
    """P[Binomial(n,p) <= k], using a log-sum-exp for small tails."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
             +i*lp+(n-i)*lq for i in range(k+1)]
    peak = max(terms)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(t-peak) for t in terms))


def _survival(n, k, p):
    """P[Binomial(n,p) >= k], avoiding cancellation of small upper tails."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
             +i*lp+(n-i)*lq for i in range(k,n+1)]
    peak = max(terms)
    return min(1.0, math.exp(peak)*math.fsum(math.exp(t-peak) for t in terms))


def clopper_pearson(count, total, *, alpha):
    """Exact two-sided marginal binomial interval at nominal 1-alpha."""
    if (type(count) is not int or type(total) is not int
            or not 0 <= count <= total or total < 1
            or type(alpha) not in (float, int)
            or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("invalid binomial interval inputs")
    tail = alpha/2
    if count == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(55):
            mid = (lo+hi)/2
            if _survival(total, count, mid) < tail:
                lo = mid
            else:
                hi = mid
        lower = (lo+hi)/2
    if count == total:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(55):
            mid = (lo+hi)/2
            if _cdf(total, count, mid) > tail:
                lo = mid
            else:
                hi = mid
        upper = (lo+hi)/2
    return lower, upper


def evaluate(aggregate):
    """Derive simultaneous conservative paired-difference intervals."""
    original = analyze(aggregate)
    nominal_alpha_each = FAMILY_ALPHA/N_MARGINALS
    results = []
    for row in original["planning"]:
        n = row["n_observed"]
        gain, loss = row["gain"], row["loss"]
        gain_lo, gain_hi = clopper_pearson(gain, n, alpha=nominal_alpha_each)
        loss_lo, loss_hi = clopper_pearson(loss, n, alpha=nominal_alpha_each)
        point = 100*(gain-loss)/n
        lower, upper = 100*(gain_lo-loss_hi), 100*(gain_hi-loss_lo)
        if not lower <= point <= upper:
            raise AssertionError("paired effect outside conservative envelope")
        results.append({
            "stratum":row["stratum"],"left_checkpoint":row["left"],
            "right_checkpoint":row["right"],"n":n,
            "newly_keyword_positive":gain,"no_longer_keyword_positive":loss,
            "paired_difference_pp":round(point,4),
            "familywise_conservative_lower_pp":round(lower,4),
            "familywise_conservative_upper_pp":round(upper,4),
            "interval_includes_zero":lower <= 0 <= upper,
            "marginal_gain_lower":round(gain_lo,7),
            "marginal_gain_upper":round(gain_hi,7),
            "marginal_loss_lower":round(loss_lo,7),
            "marginal_loss_upper":round(loss_hi,7),
        })
    return {
        "evidence_class":"conservative 95% simultaneous familywise paired lexical difference envelopes",
        "method":"Clopper-Pearson intervals for 12 correlated marginal binomial rates; Bonferroni union bound; difference of intervals",
        "family_alpha":FAMILY_ALPHA,
        "marginal_interval_alpha":nominal_alpha_each,
        "comparison_count":N_COMPARISONS,
        "benchmark_sha256":original["benchmark_sha256"],
        "private_journal_sha256":original["private_journal_sha256"],
        "checkpoint_sha256":original["checkpoint_sha256"],
        "cases_existing":450,"responses_existing":1350,
        "new_model_generations":0,"model_weights_changed":False,
        "comparisons":results,
        "all_comparisons_include_zero":all(x["interval_includes_zero"] for x in results),
        "human_refusals_evaluated":False,"release_approved":False,
        "limitations":[
            "Intervals apply only to marginal keyword indicator transitions under independent prompt sampling; not behavioral refusal",
            "Conservative familywise bounds are wider than exact paired-specific alternatives and do not imply equivalence",
            "Exploratory post-hoc effect selection, dependence and 2 calibration pairs limit external validity",
            "Different training regimes for pretrained C prohibit causal attribution to abliteration",
            "No modified checkpoints, private prompts or grader ratings included",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--aggregate",type=Path,required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=evaluate(json.loads(args.aggregate.read_text()))
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite evidence")
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
