"""Mocked localhost inference tests: no network or model instance required."""
import json
from pathlib import Path

import pytest

from src.lfm3_optimizer_feedback_client import (
    MODEL_CHAT, MODEL_INVENTORY, _loaded_lfm3, query_loaded_lfm3,
)
from src.lfm3_optimizer_observer import BASE, observe

EVIDENCE = Path(__file__).resolve().parents[1] / (
    "research/evidence/lfm3-real-lora-20261029/"
    "lfm3-real-lora-3seed-20step-equal-lr-20261029.json"
)


@pytest.fixture(scope="module")
def report():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


class Reply:
    status = 200
    def __init__(self, value):
        self.value = value if isinstance(value, bytes) else json.dumps(value).encode()
    def read(self, limit):
        return self.value[:limit]


class FakeOpener:
    def __init__(self, inventory, completion=None):
        self.inventory, self.completion = inventory, completion
        self.calls = []
    def open(self, request, timeout):
        self.calls.append((request.full_url, request.get_method(), timeout))
        if request.full_url == MODEL_INVENTORY:
            return Reply(self.inventory)
        assert request.full_url == MODEL_CHAT
        assert request.get_method() == "POST"
        body = json.loads(request.data)
        assert body["temperature"] == 0
        assert body["max_tokens"] == 160
        assert body["model"] == "liquidai/lfm3-1.7b"
        assert len(body["messages"]) == 2
        return Reply(self.completion)


def inventory(loaded=True):
    return {"models": [{"type": "llm", "key": "liquidai/lfm3-1.7b",
                        "loaded_instances": [{"id": "test"}] if loaded else []}]}


def test_denies_inference_without_explicit_optin(report):
    fake = FakeOpener(inventory())
    out = query_loaded_lfm3(report, allow_local_inference=False, opener=fake)
    assert out["status"] == "disabled_explicit_opt_in_required"
    assert fake.calls == []


def test_does_not_load_an_unloaded_model(report):
    fake = FakeOpener(inventory(loaded=False))
    out = query_loaded_lfm3(report, allow_local_inference=True, opener=fake)
    assert out["status"] == "skipped_no_loaded_lfm3"
    assert fake.calls == [(MODEL_INVENTORY, "GET", 5)]
    assert out["weight_promotion_permitted"] is False


def test_valid_response_only_records_advice(report):
    sha = observe(report)["report_sha256"]
    payload = {"optimizer": "adamw", "learning_rate": 9e-5,
               "base_sha256": BASE, "report_sha256": sha,
               "reason": "Compare isolated lower-LR research arm"}
    fake = FakeOpener(inventory(), {"choices": [{"message": {"content": json.dumps(payload)}}]})
    out = query_loaded_lfm3(report, allow_local_inference=True, opener=fake)
    assert out["status"] == "advisory_model_response_reviewed"
    assert out["suggestion_review"]["status"] == "bounded_research_suggestion_only"
    assert out["trial_authorized"] is False
    assert out["weight_promotion_permitted"] is False
    assert fake.calls[-1][0] == MODEL_CHAT


def test_model_cannot_submit_training_commands(report):
    sha = observe(report)["report_sha256"]
    payload = {"optimizer": "adamw", "learning_rate": 1e-4,
               "base_sha256": BASE, "report_sha256": sha, "reason": "Try control",
               "action": "launch_training"}
    fake = FakeOpener(inventory(), {"choices": [{"message": {"content": json.dumps(payload)}}]})
    out = query_loaded_lfm3(report, allow_local_inference=True, opener=fake)
    assert out["suggestion_review"]["status"] == "rejected"
    assert out["weight_promotion_permitted"] is False


def test_invalid_or_multiple_model_inventory_fails_closed(report):
    for inv in ({"models": "wrong"}, {"models": [None]},
                {"models": inventory()["models"] * 2}):
        fake = FakeOpener(inv)
        out = query_loaded_lfm3(report, allow_local_inference=True, opener=fake)
        assert out["status"] == "blocked_inference_or_response"
        assert len(fake.calls) == 1


def test_duplicate_response_json_rejected(report):
    fake = FakeOpener(inventory(), b'{"choices":[], "choices":[]}')
    out = query_loaded_lfm3(report, allow_local_inference=True, opener=fake)
    assert out["status"] == "blocked_inference_or_response"


def test_oversized_response_rejected(report):
    fake = FakeOpener(inventory(), b"x" * 40000)
    out = query_loaded_lfm3(report, allow_local_inference=True, opener=fake)
    assert out["status"] == "blocked_inference_or_response"


def test_no_unrelated_loaded_model_is_queried(report):
    fake = FakeOpener({"models": [{"type": "llm", "key": "k2-horizon-0.9b",
                                   "loaded_instances": [{"id": "test"}]}]})
    out = query_loaded_lfm3(report, allow_local_inference=True, opener=fake)
    assert out["status"] == "skipped_no_loaded_lfm3"
    assert len(fake.calls) == 1
