"""Offline paired LFM3 candidate gate; self-reported data can never promote weights."""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import stat
from pathlib import Path

SCHEMA = "sensorgraft.lfm3.paired-holdout.v1"
MAX_BYTES = 65536
MAX_RUNS = 16
MIN_RUNS = 3
HEX64 = re.compile(r"[a-f0-9]{64}\Z")
ENVELOPE = frozenset({"schema", "base_sha256", "dataset_sha256", "eval_sha256",
                      "benchmark_sha256", "baseline_optimizer", "candidate_optimizer",
                      "baseline", "candidate"})
ROW = frozenset({"seed", "checkpoint_sha256", "quality",
                 "p95_latency_ms", "peak_memory_bytes"})
OPTIMIZERS = frozenset({"adamw", "muon", "hybrid"})


class InvalidEvidence(ValueError):
    """Evidence has not met the strictly numeric comparison contract."""


def _pairs(items):
    obj = {}
    for k, v in items:
        if k in obj:
            raise InvalidEvidence("duplicate_key")
        obj[k] = v
    return obj


def _reject_constant(_value):
    raise InvalidEvidence("nonfinite_literal")


def _digest(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _row(item):
    if not isinstance(item, dict) or set(item) != ROW:
        raise InvalidEvidence("row_fields")
    if type(item["seed"]) is not int or item["seed"] < 0 or item["seed"] > 2**31:
        raise InvalidEvidence("seed")
    if not _digest(item["checkpoint_sha256"]):
        raise InvalidEvidence("checkpoint_digest")
    if not _finite(item["quality"]) or not 0 <= item["quality"] <= 1:
        raise InvalidEvidence("quality")
    if not _finite(item["p95_latency_ms"]) or not 0 < item["p95_latency_ms"] <= 1e9:
        raise InvalidEvidence("latency")
    n = item["peak_memory_bytes"]
    if type(n) is not int or not 0 < n <= 2**50:
        raise InvalidEvidence("memory")
    return item


def assess(value):
    """Suggest independent review at best; never attest or promote a checkpoint."""
    if not isinstance(value, dict) or set(value) != ENVELOPE or value["schema"] != SCHEMA:
        raise InvalidEvidence("envelope")
    digests = [value[k] for k in ("base_sha256", "dataset_sha256", "eval_sha256", "benchmark_sha256")]
    if any(not _digest(x) for x in digests) or len(set(digests)) != 4:
        raise InvalidEvidence("provenance")
    if any(value[k] not in OPTIMIZERS if isinstance(value[k], str) else True
           for k in ("baseline_optimizer", "candidate_optimizer")):
        raise InvalidEvidence("optimizer")
    sides = []
    for key in ("baseline", "candidate"):
        rows = value[key]
        if not isinstance(rows, list) or not MIN_RUNS <= len(rows) <= MAX_RUNS:
            raise InvalidEvidence("replicate_count")
        by_seed = {}
        for row in rows:
            r = _row(row)
            if r["seed"] in by_seed:
                raise InvalidEvidence("duplicate_seed")
            by_seed[r["seed"]] = r
        sides.append(by_seed)
    a, b = sides
    if set(a) != set(b):
        raise InvalidEvidence("unpaired_seeds")
    if any(a[s]["checkpoint_sha256"] == b[s]["checkpoint_sha256"] for s in a):
        raise InvalidEvidence("identical_checkpoint")
    gains = [b[s]["quality"] - a[s]["quality"] for s in sorted(a)]
    latency_ratios = [b[s]["p95_latency_ms"] / a[s]["p95_latency_ms"] for s in sorted(a)]
    memory_ratios = [b[s]["peak_memory_bytes"] / a[s]["peak_memory_bytes"] for s in sorted(a)]
    mean_gain = sum(gains) / len(gains)
    gates = {
        "paired_runs_at_least_3": True,
        "mean_quality_gain_ge_0p02": mean_gain >= 0.02 - 1e-12,
        "no_seed_quality_regression_gt_0p01": min(gains) >= -0.01 - 1e-12,
        "p95_latency_ratio_le_1p25": max(latency_ratios) <= 1.25,
        "peak_memory_ratio_le_1p15": max(memory_ratios) <= 1.15,
    }
    return {
        "schema": "sensorgraft.lfm3.paired-holdout-review.v1",
        "status": "candidate_for_independent_review" if all(gates.values()) else "hold",
        "gates": gates,
        "summary": {
            "paired_seed_count": len(a),
            "mean_quality_delta": round(mean_gain, 8),
            "worst_quality_delta": round(min(gains), 8),
            "worst_latency_ratio": round(max(latency_ratios), 8),
            "worst_memory_ratio": round(max(memory_ratios), 8),
            "base_sha256": value["base_sha256"],
            "eval_sha256": value["eval_sha256"],
        },
        "source_identity_verified": False,
        "evaluation_independence_verified": False,
        "checkpoint_integrity_verified": False,
        "weight_promotion_permitted": False,
    }


def load(path: Path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_BYTES or st.st_nlink != 1:
            raise InvalidEvidence("unsafe_file")
        with os.fdopen(os.dup(fd), "rb") as stream:
            payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise InvalidEvidence("oversized_input")
        return json.loads(payload, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        result = assess(load(args.manifest))
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        result = {"schema": "sensorgraft.lfm3.paired-holdout-review.v1",
                  "status": "blocked", "reason": type(exc).__name__,
                  "source_identity_verified": False,
                  "evaluation_independence_verified": False,
                  "checkpoint_integrity_verified": False,
                  "weight_promotion_permitted": False}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["status"] == "candidate_for_independent_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
