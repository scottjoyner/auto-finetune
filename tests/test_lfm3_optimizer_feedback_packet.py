"""Check that LFM3 sees only verified synthetic numerical telemetry."""
import copy
import json
from pathlib import Path

import pytest

from src.lfm3_optimizer_feedback_packet import SYSTEM, build_packet
from src.lfm3_optimizer_observer import observe

EVIDENCE = Path(__file__).resolve().parents[1] / (
    "research/evidence/lfm3-real-lora-20261029/"
    "lfm3-real-lora-3seed-20step-equal-lr-20261029.json"
)


@pytest.fixture(scope="module")
def report():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_lfm3_observes_only_verified_numeric_metrics(report):
    p = build_packet(report)
    assert p["status"] == "advisory_prompt_only"
    assert p["source_report_sha256"] == observe(report)["report_sha256"]
    assert p["model_inference_performed"] is False
    assert p["trial_authorized"] is False
    assert p["weight_promotion_permitted"] is False
    assert [m["role"] for m in p["messages"]] == ["system", "user"]
    body = json.loads(p["messages"][1]["content"])
    assert len(body["observations"]) == 6
    assert len(body["candidate_grid"]) == 6
    assert body["auto_execute"] is False
    assert body["training_data"] == "synthetic_random_tokens_not_task_quality"
    assert "events" not in body
    assert "train" not in body


def test_instruction_contract_has_no_executable_actions():
    assert "No tools" in SYSTEM
    assert "separate verifier" in SYSTEM


def test_source_tampering_never_generates_model_input(report):
    tampered = copy.deepcopy(report)
    tampered["arms"][0]["events"][0]["base_sha256"] = "0" * 64
    result = build_packet(tampered)
    assert result["status"] == "blocked"
    assert result["trial_authorized"] is False
    assert "messages" not in result


def test_private_traces_are_not_accepted(report):
    tampered = copy.deepcopy(report)
    tampered["real_private_traces_used"] = True
    result = build_packet(tampered)
    assert result["status"] == "blocked"
    assert "messages" not in result


def test_fresh_report_yields_new_prompt_digest(report):
    changed = copy.deepcopy(report)
    changed["arms"][0]["events"][0]["grad_norm"] *= 1.01
    assert build_packet(changed)["status"] == "advisory_prompt_only"
    assert build_packet(changed)["source_report_sha256"] != build_packet(report)["source_report_sha256"]
