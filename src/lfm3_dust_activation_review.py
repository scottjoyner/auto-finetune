"""Conservative review of activation-conditioned LFM3 Dust pilot receipts.

The source evidence is unsigned; a passing structural review is NOT
independent proof of execution, authenticity, causal timing or usefulness.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from src.lfm3_dust_forward_review import verify as verify_direction

_SHA = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "sensorgraft.lfm3.dust-activation-review.v1"


def blocked(reason):
    return {"schema": SCHEMA, "status": "blocked", "reason": reason,
            "trial_authorized": False, "weight_promotion_permitted": False}


def review_activation(report):
    if not isinstance(report, dict):
        return blocked("invalid_source")
    if (report.get("policy") != "activation_attended"
            or report.get("direction_head") != "causal_prompt_activation_two_head_training_probes_only"
            or report.get("extra_prompt_only_forwards") != report.get("steps")
            or report.get("raw_activations_persisted") is not False):
        return blocked("missing_activation_head_contract")
    transformed = deepcopy(report)
    transformed["policy"] = "attended"
    transformed["direction_head"] = "causal_online_kernel_attention_training_probes_only"
    transformed.pop("extra_prompt_only_forwards", None)
    transformed.pop("raw_activations_persisted", None)
    if not isinstance(transformed.get("records"), list):
        return blocked("invalid_activation_records")
    digests = []
    for row in transformed["records"]:
        if not isinstance(row, dict):
            return blocked("invalid_activation_event")
        digest = row.pop("prompt_context_sha256", None)
        differences = row.pop("attention_head_disagreement", None)
        extra = row.pop("prompt_only_extra_forwards", None)
        if (type(digest) is not str or not _SHA.fullmatch(digest)
                or not isinstance(differences, list) or len(differences) != 4
                or any(type(x) not in (float, int) or not math.isfinite(x)
                       or not 0 <= x <= 1e4 for x in differences)
                or extra != 1):
            return blocked("invalid_or_missing_context_digest")
        digests.append(digest)
    if len(digests) != transformed["steps"]:
        return blocked("incomplete_context_precommit")
    result = verify_direction(transformed)
    if result.get("status") != "manual_review_only":
        return blocked("underlying_forward_contract_failed")
    return {
        "schema": SCHEMA, "status": "research_review_required",
        "seed": result["seed"], "policy": "activation_attended",
        "steps": transformed["steps"],
        "attended_selection_accuracy": result["score_after"],
        "accepted_steps": result["accepted_steps"],
        "actual_antithetic_forwards": 4 * transformed["steps"],
        "extra_prompt_only_forwards": transformed["steps"],
        "context_digest_count": len(digests),
        "raw_activations_stored_in_receipt": False,
        "head_coefficient_training": "online_historical_attention_memory",
        "trained_transformer_attention_weights": False,
        "independent_execution_attestation": False,
        "model_or_optimizer_superiority_proven": False,
        "trial_authorized": False, "weight_promotion_permitted": False,
    }


def compare_activation_to_control(activation, control):
    a = review_activation(activation)
    b = verify_direction(control)
    if a["status"] != "research_review_required" or b["status"] != "manual_review_only":
        return blocked("invalid_paired_candidate_or_control")
    if (b["policy"] not in ("random", "attended")
            or a["seed"] != b["seed"] or a["steps"] != control.get("steps")
            or activation.get("base_sha256") != control.get("base_sha256")
            or activation.get("data_manifest") != control.get("data_manifest")
            or activation.get("initial_lora_b_sha256") != control.get("initial_lora_b_sha256")
            or activation.get("selected_adapter_names") != control.get("selected_adapter_names")
            or activation.get("selection_before") != control.get("selection_before")
            or activation.get("total_antithetic_forwards") != control.get("total_antithetic_forwards")):
        return blocked("nonmatched_frozen_base_or_probe_budget")
    return {"schema": SCHEMA, "status": "paired_manual_review_required",
            "seed": a["seed"], "activation": a,
            "control": {"policy": b["policy"], "score_after": b["score_after"],
                        "accepted_steps": b["accepted_steps"]},
            "additional_prompt_forward_overhead": a["extra_prompt_only_forwards"],
            "training_improvement_or_benefit_proven": False,
            "trial_authorized": False, "weight_promotion_permitted": False}
