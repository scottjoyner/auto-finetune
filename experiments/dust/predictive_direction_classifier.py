"""Shadow-only predictor for useful stochastic activation-perturbation directions.

Research toy experiment, NOT connected to Dust/K2 training dispatch. A direction
is 'useful' if it reduces a *synthetic quadratic* after a hypothetical step.
Candidate features are measured or estimated BEFORE the candidate's label.
Nothing here imports the model, changes sampling probabilities, or runs backprop.

Three bootstrap logistic heads estimate agreement/uncertainty. Train/validate/
test are split by synthetic episode, not candidate row. Validation determines
the classification threshold; the untouched test split is reported once.

python -m experiments.dust.predictive_direction_classifier --synthetic-only
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, pstdev

FEATURE_NAMES = (
    "momentum_projection", "recent_projection", "older_projection",
    "curvature_proxy", "step_times_curvature", "direction_disagreement",
    "step_size", "momentum_abs_projection",
)
SPLIT_EPISODES = {"train": tuple(range(1100, 1164)),
                  "validation": tuple(range(2100, 2116)),
                  "test": tuple(range(3100, 3116))}
CANDIDATES_PER_EPISODE = 16
DIMENSIONS = 8
ENSEMBLE_SEEDS = (7, 42, 1337)
TOP_K = 4
# Explicit research boundary. No code path here is allowed to call a trainer.
RESEARCH_ONLY = True
TRAINER_ADMISSION = False


@dataclass(frozen=True)
class Candidate:
    episode: int
    candidate: int
    features: tuple[float, ...]
    useful: int
    actual_gain: float
    momentum_score: float
    curvature_score: float


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def make_episode(episode: int) -> list[Candidate]:
    if episode < 0:
        raise ValueError("invalid seed")
    rng = random.Random(episode)
    gradient = [rng.gauss(0, 1) for _ in range(DIMENSIONS)]
    diagonal = [rng.uniform(0.2, 2.7) for _ in range(DIMENSIONS)]
    # Both historical probes and noisy curvature proxy precede the candidate
    # outcome. They are NOT exact gradients or future-loss observations.
    old = [g + rng.gauss(0, 0.7) for g in gradient]
    new = [g + rng.gauss(0, 0.7) for g in gradient]
    curvature_estimate = [max(0.01, h + rng.gauss(0, 0.8))
                          for h in diagonal]
    eta = rng.uniform(0.25, 0.8)
    rows = []
    for idx in range(CANDIDATES_PER_EPISODE):
        raw = [rng.gauss(0, 1) for _ in range(DIMENSIONS)]
        magnitude = math.sqrt(dot(raw, raw))
        direction = [v / magnitude for v in raw]
        old_projection = dot(old, direction)
        new_projection = dot(new, direction)
        momentum = 0.65 * new_projection + 0.35 * old_projection
        curvature = dot(curvature_estimate, [v * v for v in direction])
        true_curvature = dot(diagonal, [v * v for v in direction])
        true_delta = eta * dot(gradient, direction) + (
            0.5 * eta * eta * true_curvature)
        gain = -true_delta  # positive == improvement
        # A historical-gradient heuristic does NOT know the true label.
        momentum_score = -momentum
        curvature_score = -momentum - 0.5 * eta * curvature
        features = (
            -momentum, -new_projection, -old_projection, curvature,
            eta * curvature, abs(new_projection - old_projection), eta,
            abs(momentum),
        )
        if not all(math.isfinite(x) for x in features + (gain,)):
            raise ArithmeticError("nonfinite synthetic observation")
        rows.append(Candidate(episode, idx, features, int(gain > 0),
                              gain, momentum_score, curvature_score))
    return rows


def split_fixture() -> dict[str, list[Candidate]]:
    groups = [set(v) for v in SPLIT_EPISODES.values()]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("episode leakage across splits")
    return {split: [r for seed in seeds for r in make_episode(seed)]
            for split, seeds in SPLIT_EPISODES.items()}


@dataclass(frozen=True)
class Standardizer:
    center: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def from_training(cls, rows: list[Candidate]):
        if not rows:
            raise ValueError("empty train")
        columns = list(zip(*(r.features for r in rows)))
        return cls(tuple(mean(v) for v in columns),
                   tuple(max(pstdev(v), 1e-6) for v in columns))

    def transform(self, features):
        if len(features) != len(FEATURE_NAMES):
            raise ValueError("unexpected predictive feature schema")
        if not all(math.isfinite(v) for v in features):
            raise ValueError("nonfinite features")
        return (1.0,) + tuple((x - m) / s for x, m, s
                              in zip(features, self.center, self.scale, strict=True))


def logistic(value):
    if value < -40:
        return 0.0
    if value > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def train_head(rows: list[Candidate], normalizer: Standardizer, seed: int,
               epochs: int = 100, learning_rate: float = .10):
    """Bootstrap whole episodes; never bootstrap candidate rows independently."""
    by_episode = {}
    for r in rows:
        by_episode.setdefault(r.episode, []).append(r)
    if set(by_episode) != set(SPLIT_EPISODES["train"]):
        raise ValueError("training contains non-train episodes")
    rnd = random.Random(seed)
    episode_ids = sorted(by_episode)
    boot = [rnd.choice(episode_ids) for _ in episode_ids]
    selected = [r for k in boot for r in by_episode[k]]
    rng = random.Random(seed + 101)
    weights = [0.0] * (len(FEATURE_NAMES) + 1)
    for epoch in range(epochs):
        # Batch SGD on a fixed training-only bootstrapped episode cohort.
        rng.shuffle(selected)
        gradient = [0.0] * len(weights)
        for row in selected:
            vec = normalizer.transform(row.features)
            residual = logistic(dot(weights, vec)) - row.useful
            for j, x in enumerate(vec):
                gradient[j] += residual * x
        for j in range(len(weights)):
            reg = .005 * weights[j] if j else 0.0
            weights[j] -= learning_rate * (
                gradient[j] / len(selected) + reg)
    return tuple(weights)


def predict_probability(heads, normalizer, candidate: Candidate):
    vec = normalizer.transform(candidate.features)
    probs = [logistic(dot(w, vec)) for w in heads]
    return mean(probs), pstdev(probs)


def choose_threshold(heads, normalizer, validation):
    """Frozen grid, optimized ONLY on validation. Never inspect test here."""
    if any(r.episode not in SPLIT_EPISODES["validation"]
           for r in validation):
        raise ValueError("threshold selection must use validation only")
    options = (.35, .40, .45, .50, .55, .60, .65)
    scores = []
    for cut in options:
        tp = fp = fn = 0
        for row in validation:
            prediction = predict_probability(heads, normalizer, row)[0] >= cut
            tp += int(prediction and row.useful)
            fp += int(prediction and not row.useful)
            fn += int(not prediction and row.useful)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        scores.append((f1, -abs(cut - .5), cut))
    return max(scores)[2]


def brier_score(rows, predict):
    return mean((predict(r) - r.useful) ** 2 for r in rows)


def selected_metrics(rows, scored, *, top_k=TOP_K):
    episodes = {}
    for row in rows:
        episodes.setdefault(row.episode, []).append(row)
    precision = []
    gain = []
    for episode in sorted(episodes):
        candidates = episodes[episode]
        best = sorted(candidates,
                      key=lambda row: (-scored(row), row.candidate))[:top_k]
        precision.append(sum(r.useful for r in best) / len(best))
        gain.append(mean(r.actual_gain for r in best))
    return {"precision_at_4": mean(precision),
            "mean_true_gain_at_4": mean(gain)}


def evaluate(rows, heads, normalizer, threshold):
    allowed = set(SPLIT_EPISODES["test"])
    if any(row.episode not in allowed for row in rows):
        raise ValueError("test evaluation contains non-test episodes")
    model_score = lambda r: predict_probability(heads, normalizer, r)[0]
    # Same fixed direction candidates, no selectively observed labels.
    random_score = lambda r: random.Random(
        r.episode * 100_003 + r.candidate * 997 + 991).random()
    result = {
        "classifier": selected_metrics(rows, model_score),
        "momentum": selected_metrics(rows, lambda r: r.momentum_score),
        "curvature_aware_momentum": selected_metrics(
            rows, lambda r: r.curvature_score),
        "random": selected_metrics(rows, random_score),
        "classifier_brier": brier_score(rows, model_score),
        "mean_ensemble_disagreement": mean(
            predict_probability(heads, normalizer, r)[1] for r in rows),
        "classification_threshold_val_only": threshold,
        "classifier_accuracy_at_val_threshold": mean(
            (model_score(r) >= threshold) == bool(r.useful) for r in rows),
        "test_positive_fraction": mean(r.useful for r in rows),
    }
    baseline = result["curvature_aware_momentum"]
    proposed = result["classifier"]
    result["shadow_recommendation"] = (
        "CONTINUE_SHADOW" if (
            proposed["precision_at_4"] > baseline["precision_at_4"]
            and proposed["mean_true_gain_at_4"] > baseline["mean_true_gain_at_4"]
        ) else "HOLD_NO_BASELINE_ADVANTAGE"
    )
    return result


def run_synthetic():
    dataset = split_fixture()
    training, validation, test = (dataset[n] for n in
                                  ("train", "validation", "test"))
    norm = Standardizer.from_training(training)
    heads = [train_head(training, norm, seed) for seed in ENSEMBLE_SEEDS]
    threshold = choose_threshold(heads, norm, validation)
    result = evaluate(test, heads, norm, threshold)
    source_split_hashes = {
        name: hashlib.sha256(",".join(str(seed) for seed in seeds).encode()
                             ).hexdigest()
        for name, seeds in SPLIT_EPISODES.items()
    }
    model_parameters = {
        "features": FEATURE_NAMES,
        "center": norm.center, "scale": norm.scale,
        "head_weights": heads, "threshold": threshold,
    }
    digest = hashlib.sha256(
        json.dumps(model_parameters, sort_keys=True).encode()).hexdigest()
    return {
        "schema": "auto-finetune.dust-predictive-direction.synthetic.v1",
        "research_only": RESEARCH_ONLY,
        "trainer_admission_authorized": TRAINER_ADMISSION,
        "model_checkpoint_promotion": False,
        "pretrained_model_loaded": False,
        "hosted_provider_calls": 0, "gpu_seconds": 0,
        "optimizer_updates": 0, "backward_calls": 0,
        "data": "synthetic quadratic, synthetic history/proxy, no tokens",
        "training_episodes": len(SPLIT_EPISODES["train"]),
        "validation_episodes": len(SPLIT_EPISODES["validation"]),
        "test_episodes": len(SPLIT_EPISODES["test"]),
        "candidates_per_episode": CANDIDATES_PER_EPISODE,
        "episode_split_sha256": source_split_hashes,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "model_coefficients_sha256": digest,
        "model_coefficients_are_synthetic_only": True,
        "metrics": result,
        "limitations": [
            "No real K2 per-direction loss labels collected",
            "No changes to candidate sampling or production routing",
            "No held-out language-model CE or agentic-task quality",
            "No unbiased gradient correction for classifier-based selection",
            "Tiny toy quadratic does not reproduce neural loss curvature",
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    if not args.synthetic_only:
        p.error("research training disabled unless --synthetic-only is set")
    output = json.dumps(run_synthetic(), sort_keys=True, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as file:
            file.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
