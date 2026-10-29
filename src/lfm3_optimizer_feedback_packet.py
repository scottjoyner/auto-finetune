"""Prepare a privacy-minimal read-only LFM3 optimizer observation prompt.

No inference, optimizer mutations, serving changes or checkpoint writes.
The LLM's response is untrusted; lfm3_optimizer_observer validates it separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.lfm3_optimizer_observer import _read_json, blocked, observe

SCHEMA = "sensorgraft.lfm3.optimizer-feedback-packet.v1"
SYSTEM = (
    "You are LFM3 acting only as a training telemetry research observer. "
    "No tools, shell commands, checkpoint updates, model deployment, or training "
    "operations are authorized. Do not claim optimizer superiority from synthetic "
    "random-token loss. Return only one JSON object with exactly these keys: "
    "optimizer, learning_rate, base_sha256, report_sha256, reason. "
    "Use only adamw or muon, and only an LR from the provided grid. "
    "If data are insufficient to select a trial, return an empty response. "
    "A separate verifier decides whether any suggestion can be recorded."
)


def build_packet(report):
    result = observe(report)
    if result["status"] != "research_observation_only":
        return blocked("observer_rejected_source:" + result["reason"])
    # In v1 no private traces or natural-language user data may enter a prompt.
    if report.get("synthetic_only") is not True or report.get("real_private_traces_used") is not False:
        return blocked("private_training_data_not_authorized")
    metrics = [{
        "seed": x["seed"],
        "optimizer": x["optimizer"],
        "median_grad_norm": x["median_grad_norm"],
        "median_update_norm": x["median_update_norm"],
        "mean_step_s": x["mean_step_s"],
        "first_train_loss": x["start_train_loss"],
        "last_train_loss": x["end_train_loss"],
    } for x in result["observations"]]
    evidence = {
        "base_sha256": result["base_sha256"],
        "report_sha256": result["report_sha256"],
        "training_data": "synthetic_random_tokens_not_task_quality",
        "observations": metrics,
        "candidate_grid": result["proposed_next_experiment_grid"],
        "required_fields": [
            "optimizer", "learning_rate", "base_sha256", "report_sha256", "reason"
        ],
        "auto_execute": False,
    }
    return {
        "schema": SCHEMA,
        "status": "advisory_prompt_only",
        "source_report_sha256": result["report_sha256"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(evidence, sort_keys=True, allow_nan=False)},
        ],
        "model_inference_performed": False,
        "trial_authorized": False,
        "weight_promotion_permitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        packet = build_packet(_read_json(args.report))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        packet = blocked("invalid_input:" + type(exc).__name__)
    print(json.dumps(packet, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
