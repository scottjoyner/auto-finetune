"""Strict privacy-minimal contract for *completed* forward-only direction probes.

The classifier is not trained from these records in this PR. This validator
prepares the acceptance boundary for a separately approved real K2 dataset:
features recorded before evaluating a direction, subsequent +/- forward loss
observations, and episode-level partitioning with deterministic hashing.

A pseudonymous episode ID MUST be keyed/HMAC-based upstream, not a plain
hash of an individual prompt. Raw prompt/token/activation values are never
accepted by this module. Provenance/timing must be independently attested
before real records can train anything; a JSON field cannot prove causality.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from .predictive_direction_classifier import FEATURE_NAMES

SCHEMA = "auto-finetune.dust-predictive-probe.v1"
FIELDS = frozenset({
    "schema", "episode_hmac_sha256", "model_revision_sha256",
    "candidate_index", "features_pre_probe", "sigma", "loss_clean",
    "loss_plus", "loss_minus", "probe_time_order_attested",
})
MAX_ROWS = 20000
MAX_BYTES = 8 * 1024 * 1024


def required_hex(value: object, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or not all(c in "0123456789abcdef" for c in value)):
        raise ValueError("invalid pseudonymous digest: " + field)
    return value


def partition(episode_hmac_sha256: str) -> str:
    """Episode-disjoint split; only keyed episode IDs are permitted."""
    episode = required_hex(episode_hmac_sha256, "episode_hmac_sha256")
    value = int(hashlib.sha256(
        ("dust-predictive-split-v1:" + episode).encode()).hexdigest()[:8], 16)
    bucket = value % 10
    if bucket < 6:
        return "train"
    if bucket < 8:
        return "validation"
    return "test"


def validate_record(record: object) -> dict:
    if not isinstance(record, dict) or set(record) != FIELDS:
        raise ValueError("unexpected/unsafe probe record fields")
    if record["schema"] != SCHEMA:
        raise ValueError("probe record schema mismatch")
    episode = required_hex(record["episode_hmac_sha256"], "episode")
    revision = required_hex(record["model_revision_sha256"], "model_revision")
    index = record["candidate_index"]
    if type(index) is not int or not 0 <= index < 4096:
        raise ValueError("invalid direction candidate index")
    if record["probe_time_order_attested"] is not True:
        raise ValueError("no independent pre-probe feature attestation")
    features = record["features_pre_probe"]
    if not isinstance(features, list) or len(features) != len(FEATURE_NAMES):
        raise ValueError("predictive feature shape mismatch")
    if any(type(v) not in (int, float) or not math.isfinite(v)
           or abs(v) > 1e6 for v in features):
        raise ValueError("invalid/nonfinite predictive feature")
    nums = {}
    for key in ("sigma", "loss_clean", "loss_plus", "loss_minus"):
        value = record[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("nonfinite loss/sigma")
        nums[key] = float(value)
    if not (0 < nums["sigma"] <= 0.5):
        raise ValueError("invalid sigma")
    if any(nums[name] < 0 or nums[name] > 100 for name in
           ("loss_clean", "loss_plus", "loss_minus")):
        raise ValueError("invalid forward loss")
    # Plus sign is the classifier's candidate direction. Minus is the
    # antithetic control; never leak either post-probe loss into features.
    gain = nums["loss_clean"] - nums["loss_plus"]
    return {
        "episode": episode, "model_revision_sha256": revision,
        "candidate": index, "split": partition(episode),
        "features": tuple(float(v) for v in features),
        "label": int(gain > 0), "gain": gain,
        "antithetic_slope": (
            (nums["loss_plus"] - nums["loss_minus"]) / (2 * nums["sigma"])
        ),
    }


def validate_jsonl(path: Path) -> dict:
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("probe evidence exceeds bounded size")
    groups = defaultdict(list)
    seen = set()
    count = 0
    revisions = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or len(line) > 4096:
                raise ValueError("empty/oversized evidence line")
            count += 1
            if count > MAX_ROWS:
                raise ValueError("probe rows exceed cap")
            record = json.loads(line)
            item = validate_record(record)
            key = (item["episode"], item["candidate"])
            if key in seen:
                raise ValueError("duplicate episode/direction")
            seen.add(key)
            groups[item["episode"]].append(item)
            revisions.add(item["model_revision_sha256"])
    if not groups:
        raise ValueError("empty probe evidence")
    if len(revisions) != 1:
        raise ValueError("mixed model revisions: reject silent domain shift")
    per_split = {"train": 0, "validation": 0, "test": 0}
    for episode, rows in groups.items():
        split = partition(episode)
        if any(item["split"] != split for item in rows):
            raise AssertionError("episode partition inconsistency")
        per_split[split] += len(rows)
    # Intentional: return only aggregate statistics, never prompt contents,
    # candidate features, or individual loss observations.
    return {
        "schema": SCHEMA,
        "mode": "VALIDATE_ONLY_NO_TRAINER_ADMISSION",
        "raw_user_data_exposed": False,
        "rows": count,
        "episodes": len(groups),
        "split_candidate_counts": per_split,
        "single_model_revision": True,
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pre_probe_attestation_independently_verified": False,
        "classification_training_authorized": False,
        "notes": "Schema/flags cannot prove actual pre-probe timing or "
                 "keyed-HMAC custody. Require independent witness before "
                 "any labeled K2 classifier fine-tuning.",
    }
