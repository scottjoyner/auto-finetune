"""Causal, backprop-free direction-attention predictor for LFM3 Dust trials.

This is a small online kernel-attention memory over *pre-probe* direction
features and previously measured scalar gains. It is not a trained transformer
attention layer. A future task label or held-out metric may never enter fit().
An optional model wrapper must independently evaluate proposed directions.
No model weights, training admission, checkpoints or optimizer API are touched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

SCHEMA = "sensorgraft.lfm3.dust-direction-attention.v1"
DIM = 8
CAPACITY = 32
TEMPERATURE = 4.0
DECAY = 0.94


def _feature(values):
    if (not isinstance(values, (tuple, list)) or len(values) != DIM
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   for x in values)):
        raise ValueError("invalid_preprobe_direction_features")
    norm = math.sqrt(sum(float(v) ** 2 for v in values))
    if not 1e-9 <= norm <= 1e5:
        raise ValueError("zero_or_unbounded_direction_features")
    return tuple(float(v) / norm for v in values)


@dataclass(frozen=True)
class Observation:
    episode: int
    features: tuple[float, ...]
    gain: float


class DirectionAttentionHead:
    """Online attention on previous confirmed train-only perturbation outcomes."""
    def __init__(self, *, capacity=CAPACITY):
        if type(capacity) is not int or not 1 <= capacity <= CAPACITY:
            raise ValueError("invalid_memory_cap")
        self.capacity = capacity
        self._history: list[Observation] = []
        self._last_episode = -1

    @property
    def size(self):
        return len(self._history)

    def predict(self, features, *, episode):
        q = _feature(features)
        if type(episode) is not int or episode < 0:
            raise ValueError("invalid_prediction_episode")
        visible = [h for h in self._history if h.episode < episode]
        if not visible:
            return {"predicted_gain": 0.0, "evidence": 0,
                    "status": "cold_start_uninformative"}
        raw = [
            TEMPERATURE * sum(a * b for a, b in zip(q, h.features, strict=True))
            + (episode - h.episode) * math.log(DECAY)
            for h in visible
        ]
        center = max(raw)
        weights = [math.exp(x - center) for x in raw]
        weight_sum = sum(weights)
        prediction = sum(w * h.gain for w, h in zip(weights, visible, strict=True)) / weight_sum
        if not math.isfinite(prediction):
            raise ValueError("nonfinite_direction_prediction")
        return {"predicted_gain": prediction, "evidence": len(visible),
                "status": "historical_training_probe_attention"}

    def rank(self, candidates, *, episode):
        if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 16):
            raise ValueError("invalid_direction_candidates")
        scored = [(self.predict(c, episode=episode)["predicted_gain"], i)
                  for i, c in enumerate(candidates)]
        # Stable order; no future outcome used. Cold-start is candidate 0.
        return sorted(range(len(scored)), key=lambda i: (-scored[i][0], i))

    def observe(self, features, gain, *, episode, split="train", probed=True):
        q = _feature(features)
        if (type(episode) is not int or episode < 0
                or episode < self._last_episode or split != "train"
                or probed is not True or type(gain) not in (float, int)
                or not math.isfinite(gain) or abs(gain) > 1e4):
            raise ValueError("untrusted_or_postselection_direction_label")
        self._last_episode = episode
        self._history.append(Observation(episode, q, float(gain)))
        del self._history[:-self.capacity]
        return {"status": "historical_train_only_memory_updated",
                "memory_size": len(self._history),
                "backward_calls": 0, "weight_promotion_permitted": False}
