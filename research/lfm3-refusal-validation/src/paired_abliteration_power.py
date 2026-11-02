"""Model-free, paired-power planning for frozen LFM3 A/B/C proxy results.

Uses aggregate-only 450-prompt evidence. Normal approximation to the
paired score-test power is a PLANNING heuristic, not inferential proof.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

ARMS = ("A_original_posttrained", "B_M1_abliterated_smoke", "C_base_pretrained")
PAIRS = ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2]))
STRATA = {"safe": 250, "unsafe": 200}
NORMAL = NormalDist()
FAMILY_ALPHA = 0.05 / 6


def _valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def exact_mcnemar(gain, loss):
    n = gain + loss
    if not n:
        return 1.0
    k = min(gain, loss)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def power_approx(n, gain_rate, loss_rate, *, alpha=FAMILY_ALPHA):
    """Normal planning approximation for paired binary difference detection."""
    if type(n) is not int or n < 1:
        raise ValueError("invalid sample size")
    if not (0 < alpha < 1) or not all(
            math.isfinite(x) and x >= 0 for x in (gain_rate, loss_rate)
    ) or gain_rate + loss_rate > 1:
        raise ValueError("invalid paired transition probabilities")
    q = gain_rate + loss_rate
    d = gain_rate - loss_rate
    if q == 0 or d == 0:
        return 0.0
    variance = q - d * d
    if variance <= 0:
        return 1.0
    center = math.sqrt(n) * d / math.sqrt(q)
    spread = math.sqrt(variance / q)
    threshold = NORMAL.inv_cdf(1 - alpha / 2)
    p = NORMAL.cdf((-threshold - center) / spread) + (
        1 - NORMAL.cdf((threshold - center) / spread))
    return max(0.0, min(1.0, p))


def minimum_planning_n(gain_rate, loss_rate, *, alpha=FAMILY_ALPHA,
                       target=0.8, cap=100000):
    if gain_rate == loss_rate:
        return None
    if power_approx(cap, gain_rate, loss_rate, alpha=alpha) < target:
        return None
    lo, hi = 1, cap
    while lo < hi:
        mid = (lo + hi) // 2
        if power_approx(mid, gain_rate, loss_rate, alpha=alpha) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def analyze(evidence):
    if (evidence.get("evidence_class") !=
            "exploratory matched exact inference on pre-existing weak refusal keyword proxy only"
            or evidence.get("checkpoint_count") != 3
            or evidence.get("n_unique_prompts") != 450
            or evidence.get("n_generated_responses") != 1350
            or evidence.get("human_refusal_scoring_complete") is not False
            or evidence.get("supports_modified_model_release") is not False):
        raise ValueError("unexpected frozen evidence identity or release state")
    ids = evidence.get("checkpoint_sha256")
    if (not isinstance(ids, dict) or tuple(ids) != ARMS or
            len(set(ids.values())) != 3 or
            not all(_valid_sha(s) for s in ids.values()) or
            not _valid_sha(evidence.get("private_journal_sha256")) or
            not _valid_sha(evidence.get("benchmark_sha256"))):
        raise ValueError("missing, aliased, or unexpected checkpoint/source digest")
    entries = evidence.get("comparisons")
    expected = {(stratum, a, b) for stratum in STRATA for a, b in PAIRS}
    if not isinstance(entries, list) or len(entries) != 6:
        raise ValueError("six matched comparisons required")
    observed, records = set(), []
    for row in entries:
        key = (row.get("stratum"), row.get("left_checkpoint"), row.get("right_checkpoint"))
        if key not in expected or key in observed:
            raise ValueError("duplicate or unexpected arm/stratum")
        observed.add(key)
        n = STRATA[key[0]]
        keys = ("lexical_keyword_newly_present", "lexical_keyword_no_longer_present",
                "both_keyword_positive", "both_keyword_negative")
        counts = [row.get(x) for x in keys]
        if any(type(x) is not int or x < 0 for x in counts) or sum(counts) != n:
            raise ValueError("invalid paired 2x2 contingency table")
        gain, loss = counts[:2]
        delta = (gain - loss) * 100 / n
        if (row.get("n_paired_prompts") != n
                or row.get("net_keyword_count_change") != gain - loss
                or abs(row.get("difference_percentage_points", float("inf")) - delta) > 1e-6
                or abs(row.get("exploratory_exact_conditional_mcnemar_p", -1)
                       - exact_mcnemar(gain, loss)) > 1e-10):
            raise ValueError("published paired effect/test drift")
        rate_gain, rate_loss = gain/n, loss/n
        records.append({
            "stratum": key[0], "left": key[1], "right": key[2],
            "n_observed": n, "gain": gain, "loss": loss,
            "observed_net_difference_pp": round(delta, 4),
            "observed_exact_p": exact_mcnemar(gain, loss),
            "planning_discordance_rate": round((gain + loss)/n, 6),
            "approx_power_at_observed_n": round(power_approx(n, rate_gain, rate_loss), 4),
            "approx_n_for_80pct_power": minimum_planning_n(rate_gain, rate_loss),
        })
    if observed != expected:
        raise ValueError("incomplete matched comparisons")
    for key in STRATA:
        for left, right in PAIRS:
            row = next(x for x in records if (x["stratum"], x["left"], x["right"]) ==
                       (key, left, right))
            original = next(x for x in entries if
                            (x["stratum"], x["left_checkpoint"], x["right_checkpoint"]) ==
                            (key, left, right))
            p = original.get("holm_familywise_adjusted_p_six_exploratory_tests")
            if not isinstance(p, (int, float)) or not 0 <= p <= 1:
                raise ValueError("invalid previously published multiplicity adjustment")
    ranked = sorted(records, key=lambda x: x["observed_exact_p"])
    corrected, prior = {}, 0.
    for index, row in enumerate(ranked):
        prior = min(1., max(prior, row["observed_exact_p"]*(6-index)))
        corrected[(row["stratum"], row["left"], row["right"])] = prior
    for row in entries:
        key = (row["stratum"], row["left_checkpoint"], row["right_checkpoint"])
        if abs(row["holm_familywise_adjusted_p_six_exploratory_tests"]
               - corrected[key]) > 1e-10:
            raise ValueError("published Holm correction drift")
    return {
        "evidence_class": "model-free paired proxy power-planning sensitivity; not a new inference experiment",
        "model_checkpoints": list(ARMS), "n_unique_existing_prompts": 450,
        "model_weights_changed": False, "new_generations": 0,
        "benchmark_sha256": evidence["benchmark_sha256"],
        "private_journal_sha256": evidence["private_journal_sha256"],
        "checkpoint_sha256": ids,
        "assumed_two_sided_family_alpha_per_contrast": FAMILY_ALPHA,
        "method": "normal approximation to paired binary score-test power; six-test Bonferroni design allowance",
        "assumption": "future independent cases replicate observed gain/loss rates; retrospective and unstable",
        "planning": records,
        "human_behavioral_grades_completed": False,
        "refusal_reduction_validated": False,
        "release_approved": False,
        "limitations": [
            "Normal approximation is unreliable for sparse discordant transitions",
            "Observed effect sizes and discordance were estimated on the same frozen data",
            "Effect sizes are lexical proxy changes; power is not evidence of harmful compliance",
            "C is an unmatched training regime; do not interpret its contrasts as an ablation",
            "No human grades or new model generations were produced",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = analyze(json.loads(args.evidence.read_text()))
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("refuse to overwrite existing evidence")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()