"""Fail-closed offline audit of legacy synthetic LFM3 optimizer smoke receipts.

This is NOT a model evaluator, training admission, or a weights promotion tool.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
from pathlib import Path

SCHEMA = "sensorgraft.lfm3.optimizer-receipt-audit.v1"
MAX_BYTES = 65536
DIGEST = re.compile(r"[a-f0-9]{64}\Z")
RECEIPT_FIELDS = frozenset({
    "adapter_sha256", "adapter_update_norm_sum", "automatically_promoted",
    "base_sha256", "elapsed_seconds", "holdout_disjoint", "holdout_loss_after",
    "holdout_loss_before", "model_backbone_weights_modified", "optimizer",
    "production_ready", "real_private_traces_used", "status", "steps",
    "train_loss_after", "train_loss_before", "train_steps_loss",
    "trainable_params",
})
ALLOWED = frozenset({"adamw", "muon"})
STATUS = "offline_synthetic_candidate_only"


class InvalidReceipt(ValueError):
    """Malformed or unsupported legacy receipt; never log source payload."""


def _pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise InvalidReceipt("duplicate_json_key")
        out[key] = value
    return out


def _nonfinite(_):
    raise InvalidReceipt("nonfinite_constant")


def _value(v):
    return type(v) in (float, int) and math.isfinite(v) and 0 <= v < 1e12


def validate(obj):
    if not isinstance(obj, dict) or set(obj) != RECEIPT_FIELDS:
        raise InvalidReceipt("unexpected_receipt_fields")
    for name in ("base_sha256", "adapter_sha256"):
        v = obj[name]
        if not isinstance(v, str) or DIGEST.fullmatch(v) is None:
            raise InvalidReceipt("invalid_digest")
    if obj["base_sha256"] == obj["adapter_sha256"]:
        raise InvalidReceipt("identical_model_and_adapter_digest")
    if obj["optimizer"] not in ALLOWED if isinstance(obj["optimizer"], str) else True:
        raise InvalidReceipt("unsupported_optimizer")
    if obj["status"] != STATUS:
        raise InvalidReceipt("unsupported_status")
    if type(obj["steps"]) is not int or not 1 <= obj["steps"] <= 10000:
        raise InvalidReceipt("invalid_step_count")
    if (type(obj["trainable_params"]) is not int or
            not 1 <= obj["trainable_params"] <= 10**10):
        raise InvalidReceipt("invalid_trainable_params")
    if any(type(obj[k]) is not bool for k in (
        "holdout_disjoint", "automatically_promoted",
        "production_ready", "model_backbone_weights_modified",
        "real_private_traces_used",
    )):
        raise InvalidReceipt("invalid_boolean")
    if obj["production_ready"] or obj["automatically_promoted"]:
        raise InvalidReceipt("unsupported_promotion_claim")
    if obj["model_backbone_weights_modified"] or obj["real_private_traces_used"]:
        raise InvalidReceipt("unsupported_training_claim")
    metrics = ("adapter_update_norm_sum", "elapsed_seconds",
               "train_loss_before", "train_loss_after",
               "holdout_loss_before", "holdout_loss_after")
    if not all(_value(obj[k]) for k in metrics) or obj["elapsed_seconds"] <= 0:
        raise InvalidReceipt("invalid_metric")
    steps = obj["train_steps_loss"]
    if (not isinstance(steps, list) or len(steps) != obj["steps"]
            or not all(_value(v) for v in steps)):
        raise InvalidReceipt("unverified_step_series")
    return obj


def audit(adamw, muon):
    """Classify observed historical receipts, refusing ALL promotion claims."""
    a, b = validate(adamw), validate(muon)
    if a["optimizer"] != "adamw" or b["optimizer"] != "muon":
        raise InvalidReceipt("optimizer_pair_mismatch")
    if a["base_sha256"] != b["base_sha256"]:
        raise InvalidReceipt("different_base_models")
    if a["adapter_sha256"] == b["adapter_sha256"]:
        raise InvalidReceipt("same_candidate_adapter_digest")
    if a["trainable_params"] != b["trainable_params"]:
        raise InvalidReceipt("different_trainable_parameter_groups")
    # A positive claimed holdout result in a smoke run is not a quality metric.
    return {
        "schema": SCHEMA,
        "status": "blocked",
        "reason": "historical_synthetic_optimizer_smoke_only",
        "optimizer_pair": ["adamw", "muon"],
        "step_count_by_optimizer": {
            "adamw": a["steps"], "muon": b["steps"],
        },
        "reported_trainable_parameter_count": a["trainable_params"],
        "holdout_disjoint_claimed": bool(a["holdout_disjoint"] and b["holdout_disjoint"]),
        "independent_training_seeds_verified": False,
        "at_least_three_seed_replicates_verified": False,
        "pretrained_model_source_verified": False,
        "independent_holdout_quality_verified": False,
        "per_node_pipette_evidence_verified": False,
        "optimizer_state_resume_verified": False,
        "source_identity_verified": False,
        "weight_promotion_permitted": False,
    }


def read(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW |
                 getattr(os, "O_NONBLOCK", 0) |
                 getattr(os, "O_CLOEXEC", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_BYTES:
            raise InvalidReceipt("unsafe_receipt_file")
        with os.fdopen(os.dup(fd), "rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise InvalidReceipt("oversized_receipt")
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    finally:
        os.close(fd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adamw", required=True, type=Path)
    p.add_argument("--muon", required=True, type=Path)
    args = p.parse_args()
    try:
        result = audit(read(args.adamw), read(args.muon))
    except (InvalidReceipt, OSError, ValueError, TypeError, OverflowError):
        result = {"schema": SCHEMA, "status": "blocked",
                  "reason": "invalid_or_untrusted_legacy_receipt",
                  "source_identity_verified": False,
                  "weight_promotion_permitted": False}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    # Smoke artifacts are categorically not promotion-eligible.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
