"""Episode-local causal history features for K2 direction utility research.

All values returned by preview() are functions of *previously completed*
direction probes only. commit() is called AFTER the current antithetic ±
loss has been observed. Histories are never shared between source groups.

This module does not connect to the active estimator or train a model.
It defines the frozen real-probe history contract for a future v2 witness.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

SCHEMA = "auto-finetune.dust-k2-direction-history.v2"
FEATURES = (
    "prior_completed_count_fraction",
    "prior_beneficial_rate",
    "prior_plus_gain_mean",
    "prior_plus_gain_std",
    "prior_antithetic_slope_mean",
    "prior_antithetic_slope_std",
    "prior_absolute_slope_mean",
    "prior_latest_plus_gain",
)


def finite_float(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("invalid numeric history value: " + name)
    value = float(value)
    if abs(value) > 1000:
        raise ValueError("out-of-range history value: " + name)
    return value


@dataclass(frozen=True)
class CompletedProbe:
    index: int
    clean_loss: float
    plus_loss: float
    minus_loss: float
    sigma: float

    @property
    def plus_gain(self):
        return self.clean_loss - self.plus_loss

    @property
    def signed_slope(self):
        return (self.plus_loss - self.minus_loss) / (2 * self.sigma)


class CausalProbeHistory:
    def __init__(self, *, expected_population: int, episode_hmac_sha256: str):
        if type(expected_population) is not int or not 1 <= expected_population <= 1024:
            raise ValueError("invalid expected population")
        if (not isinstance(episode_hmac_sha256, str)
                or len(episode_hmac_sha256) != 64
                or any(c not in "0123456789abcdef"
                       for c in episode_hmac_sha256)):
            raise ValueError("invalid episode group digest")
        self.episode = episode_hmac_sha256
        self.population = expected_population
        self._completed = []
        self._preview_cursor = 0

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def preview(self, *, before_candidate: int) -> tuple[float, ...]:
        """No candidate's own label may enter any feature.

        The expected sequential sampler must call preview for the next batch
        before any candidate in that batch is committed. A PRE batch shares
        exactly the same previous-history snapshot.
        """
        if (type(before_candidate) is not int
                or not 0 <= before_candidate < self.population
                or before_candidate < len(self._completed)):
            raise ValueError("invalid historical preview index")
        if before_candidate != self._preview_cursor:
            raise ValueError("PRE previews must be contiguous and in candidate order")
        self._preview_cursor += 1
        if not self._completed:
            return (0.,) * len(FEATURES)
        gains = [p.plus_gain for p in self._completed]
        slopes = [p.signed_slope for p in self._completed]
        def mean_std(vals):
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))
            return mu, sd
        gain_avg, gain_std = mean_std(gains)
        slope_avg, slope_std = mean_std(slopes)
        result = (
            len(gains) / self.population,
            sum(g > 0 for g in gains) / len(gains),
            gain_avg, gain_std, slope_avg, slope_std,
            sum(abs(x) for x in slopes) / len(slopes),
            gains[-1],
        )
        if not all(math.isfinite(x) for x in result):
            raise ArithmeticError("nonfinite causal-history feature")
        return tuple(result)

    def commit_batch(self, *, first_index: int,
                     clean_loss: float, plus_losses: Sequence[float],
                     minus_losses: Sequence[float], sigma: float) -> None:
        """Append completed candidate losses ONLY after PRE preview occurred.

        Refuses partial or out-of-order population, duplicate probes, invalid
        σ or nonfinite CE. Never stores prompt contents or direction vectors.
        """
        sigma = finite_float(sigma, "sigma")
        if not 0 < sigma <= .5:
            raise ValueError("invalid perturbation sigma")
        clean = finite_float(clean_loss, "clean")
        if not 0 <= clean <= 100:
            raise ValueError("invalid clean CE")
        if type(first_index) is not int or first_index != len(self._completed):
            raise ValueError("out-of-order or duplicated commit index")
        if (not plus_losses or len(plus_losses) != len(minus_losses)
                or first_index + len(plus_losses) > self.population):
            raise ValueError("invalid completed batch length")
        if self._preview_cursor != first_index + len(plus_losses):
            raise ValueError("cannot commit labels without exactly matching prior PRE previews")
        prepared = []
        for offset, (plus, minus) in enumerate(
            zip(plus_losses, minus_losses, strict=True)
        ):
            p = finite_float(plus, "plus")
            m = finite_float(minus, "minus")
            if not 0 <= p <= 100 or not 0 <= m <= 100:
                raise ValueError("CE outside bounds")
            prepared.append(CompletedProbe(first_index + offset, clean, p, m, sigma))
        self._completed.extend(prepared)

    def snapshot(self) -> dict:
        """Aggregate only. Never expose individual labels/feature histories."""
        return {
            "schema": SCHEMA,
            "source_group_hmac_sha256": self.episode,
            "completed_count": len(self._completed),
            "expected_population": self.population,
            "no_training_authority": True,
            "no_source_text_or_vectors": True,
        }
