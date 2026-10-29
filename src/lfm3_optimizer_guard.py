"""Experimental optimizer-policy firewall for LFM3/SensorGraft research.

Untrusted LFM3 suggestions are JSON data, never executable instructions.
All candidates require a NEW isolated toy-proxy run; no live Trainer mutation,
no direct source-trace training, no use of heldout metrics for adaptation.
This module has no Torch import, file writes, network or production authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

SCHEMA = "sg-lfm3-optimizer-observation.v1"
PROPOSAL_SCHEMA = "sg-lfm3-optimizer-proposal.v1"
CHECKPOINT = "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
SHA = re.compile(r"[0-9a-f]{64}\Z")
RUN = re.compile(r"[a-z0-9][a-z0-9._-]{0,47}\Z")
OBS_FIELDS = {
    "schema", "source", "model_scope", "reference_lfm3_checkpoint_sha256",
    "run_id", "train_session_sha256", "dev_session_sha256", "seed", "steps",
    "optimizer", "learning_rate", "weight_decay", "momentum", "ns_steps",
    "train_loss_start", "train_loss_end", "dev_loss", "max_grad_norm",
    "step_mean_ms", "peak_memory_mb", "nonfinite_updates",
    "actual_lfm3_weights_trained", "model_weights_written", "production_ready",
}
PROPOSAL_FIELDS = {
    "schema", "observer", "parent_observation_sha256", "target_optimizer",
    "learning_rate", "weight_decay", "momentum", "ns_steps", "steps",
    "cycle_index", "reason_code", "execution_scope",
}
OPTIMIZERS = ("adamw", "muon_hybrid")
REASONS = ("dev_loss_plateau", "gradient_instability", "training_latency",
           "controlled_optimizer_comparison", "no_change")


class OptimizerResearchRefused(ValueError):
    pass


def _finite(value: object) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def _pairs(entries: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in entries:
        if key in out:
            raise OptimizerResearchRefused("duplicate optimizer JSON key")
        out[key] = value
    return out


def parse_strict(raw: bytes, max_bytes: int = 4096) -> object:
    if type(raw) is not bytes or len(raw) > max_bytes:
        raise OptimizerResearchRefused("optimizer packet over byte budget")
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(
                              OptimizerResearchRefused("nonfinite JSON constant")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise OptimizerResearchRefused("invalid optimizer JSON") from exc


def validate_observation(o: object) -> dict:
    if type(o) is not dict or set(o) != OBS_FIELDS:
        raise OptimizerResearchRefused("unexpected telemetry field set")
    if (o["schema"] != SCHEMA
        or o["source"] != "fabricated-synthetic-lab-only"
        or o["model_scope"] != "toy-proxy-not-lfm3-finetuning"
        or o["reference_lfm3_checkpoint_sha256"] != CHECKPOINT
        or type(o["run_id"]) is not str or not RUN.fullmatch(o["run_id"])
        or type(o["train_session_sha256"]) is not str
        or not SHA.fullmatch(o["train_session_sha256"])
        or type(o["dev_session_sha256"]) is not str
        or not SHA.fullmatch(o["dev_session_sha256"])
        or o["train_session_sha256"] == o["dev_session_sha256"]
        or type(o["seed"]) is not int or not 0 <= o["seed"] < 2**32
        or type(o["steps"]) is not int or not 1 <= o["steps"] <= 128
        or o["optimizer"] not in OPTIMIZERS
        or type(o["nonfinite_updates"]) is not int
        or not 0 <= o["nonfinite_updates"] <= o["steps"]
        or o["actual_lfm3_weights_trained"] is not False
        or o["model_weights_written"] is not False
        or o["production_ready"] is not False):
        raise OptimizerResearchRefused("untrusted or malformed optimizer observation")
    for name in ("learning_rate", "weight_decay", "momentum", "train_loss_start",
                 "train_loss_end", "dev_loss", "max_grad_norm", "step_mean_ms",
                 "peak_memory_mb"):
        if not _finite(o[name]):
            raise OptimizerResearchRefused("nonfinite telemetry metric")
    if (not 0 < o["learning_rate"] <= 0.01
        or not 0 <= o["weight_decay"] <= 0.1
        or not 0 <= o["momentum"] <= 0.99
        or type(o["ns_steps"]) is not int or not 0 <= o["ns_steps"] <= 5
        or o["optimizer"] == "adamw" and o["ns_steps"] != 0
        or o["optimizer"] == "muon_hybrid" and not 2 <= o["ns_steps"] <= 5
        or min(o["train_loss_start"], o["train_loss_end"], o["dev_loss"]) < 0
        or o["max_grad_norm"] < 0 or o["step_mean_ms"] <= 0
        or o["peak_memory_mb"] <= 0):
        raise OptimizerResearchRefused("telemetry metric outside training limits")
    return o


def fingerprint(o: dict) -> str:
    validate_observation(o)
    raw = json.dumps(o, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(b"sg-lfm3-toy-optimizer-observation-v1\0" + raw).hexdigest()


def observer_view(o: dict) -> dict:
    """Give an LLM only development telemetry; no heldout traces or answers."""
    validate_observation(o)
    return {
        "schema": "sg-lfm3-optimizer-observer-view.v1",
        "observation_sha256": fingerprint(o),
        "model_scope": o["model_scope"],
        "optimizer": o["optimizer"],
        "steps": o["steps"],
        "learning_rate": o["learning_rate"],
        "train_loss_start": o["train_loss_start"],
        "train_loss_end": o["train_loss_end"],
        "dev_loss": o["dev_loss"],
        "max_grad_norm": o["max_grad_norm"],
        "step_mean_ms": o["step_mean_ms"],
        "peak_memory_mb": o["peak_memory_mb"],
        "nonfinite_updates": o["nonfinite_updates"],
        "permitted_role": "propose_new_synthetic_experiment_only",
    }


def validate_proposal(proposal: object, parent: dict, *, max_cycle: int = 2) -> dict:
    validate_observation(parent)
    if type(proposal) is not dict or set(proposal) != PROPOSAL_FIELDS:
        raise OptimizerResearchRefused("unrecognized optimizer proposal fields")
    if (proposal["schema"] != PROPOSAL_SCHEMA
        or proposal["observer"] != "lfm3-untrusted-advisor"
        or proposal["execution_scope"] != "new-synthetic-experiment-only"
        or proposal["parent_observation_sha256"] != fingerprint(parent)
        or proposal["target_optimizer"] not in OPTIMIZERS
        or proposal["reason_code"] not in REASONS
        or type(proposal["cycle_index"]) is not int
        or not 0 <= proposal["cycle_index"] <= max_cycle
        or type(proposal["steps"]) is not int
        or proposal["steps"] != parent["steps"]
        or type(proposal["ns_steps"]) is not int):
        raise OptimizerResearchRefused("unapproved or unbound model suggestion")
    for key in ("learning_rate", "weight_decay", "momentum"):
        if not _finite(proposal[key]):
            raise OptimizerResearchRefused("nonfinite proposal parameter")
    optimizer = proposal["target_optimizer"]
    lr = proposal["learning_rate"]
    wd = proposal["weight_decay"]
    momentum = proposal["momentum"]
    ns = proposal["ns_steps"]
    if (not 0 <= wd <= 0.05
        or not 0.80 <= momentum <= 0.99
        or optimizer == "adamw" and (not 1e-5 <= lr <= 0.002 or ns != 0)
        or optimizer == "muon_hybrid" and (not 0.0001 <= lr <= 0.01
                                           or not 2 <= ns <= 5)
        or not 0.5 * parent["learning_rate"] <= lr <= 2.0 * parent["learning_rate"]):
        raise OptimizerResearchRefused("proposal exceeds conservative optimizer bounds")
    if parent["nonfinite_updates"]:
        raise OptimizerResearchRefused("unstable parent cannot seed automatic advice")
    return proposal


def compare_dev_only(baseline: dict, candidate: dict) -> dict:
    """Never consumes heldout outcomes. This *cannot* authorize deployment."""
    validate_observation(baseline)
    validate_observation(candidate)
    if (baseline["train_session_sha256"] != candidate["train_session_sha256"]
        or baseline["dev_session_sha256"] != candidate["dev_session_sha256"]
        or baseline["reference_lfm3_checkpoint_sha256"]
           != candidate["reference_lfm3_checkpoint_sha256"]
        or baseline["seed"] != candidate["seed"]
        or baseline["steps"] != candidate["steps"]):
        raise OptimizerResearchRefused("unpaired optimizer trials or session contamination")
    valid = (
        baseline["nonfinite_updates"] == 0
        and candidate["nonfinite_updates"] == 0
        and candidate["dev_loss"] <= baseline["dev_loss"] * 0.98
        and candidate["step_mean_ms"] <= baseline["step_mean_ms"] * 1.25
        and candidate["peak_memory_mb"] <= baseline["peak_memory_mb"] * 1.10
        and candidate["max_grad_norm"] <= max(1.0, baseline["max_grad_norm"] * 1.5)
    )
    return {
        "schema": "sg-lfm3-optimizer-dev-comparison.v1",
        "baseline_sha256": fingerprint(baseline),
        "candidate_sha256": fingerprint(candidate),
        "paired_dev_improvement_fraction":
            (baseline["dev_loss"] - candidate["dev_loss"]) / max(1e-12, baseline["dev_loss"]),
        "bounded_candidate_for_manual_review": valid,
        "heldout_evaluated": False,
        "may_train_on_source_traces": False,
        "may_modify_running_optimizer": False,
        "may_promote_to_fleet": False,
        "model_weights_modified": False,
        "production_ready": False,
    }
