"""Opt-in local LFM3 optimizer-feedback inference; read-only and nonpromotable.

No model loading, deployment, weight adjustment, or trainer activation.
Only the loopback LM Studio endpoint is supported, never arbitrary URLs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from src.lfm3_optimizer_feedback_packet import build_packet
from src.lfm3_optimizer_observer import _model_suggestion, _read_json, blocked, observe

MODEL_INVENTORY = "http://127.0.0.1:1234/api/v1/models"
MODEL_CHAT = "http://127.0.0.1:1234/v1/chat/completions"
SCHEMA = "sensorgraft.lfm3.optimizer-feedback-inference.v1"


def _decode(response, *, limit):
    raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("response_oversized")
    if response.status != 200:
        raise ValueError("http_not_success")
    return json.loads(raw, object_pairs_hook=_unique_pairs)


def _unique_pairs(pairs):
    data = {}
    for k, v in pairs:
        if k in data:
            raise ValueError("duplicate_json_key")
        data[k] = v
    return data


def _loaded_lfm3(inventory):
    if not isinstance(inventory, dict) or not isinstance(inventory.get("models"), list):
        raise ValueError("invalid_model_inventory")
    matches = []
    for item in inventory["models"]:
        if not isinstance(item, dict):
            raise ValueError("invalid_model_entry")
        key, loaded = item.get("key"), item.get("loaded_instances")
        if (item.get("type") == "llm" and type(key) is str and "lfm3" in key.lower()
                and isinstance(loaded, list) and len(loaded) > 0):
            matches.append(key)
    if len(matches) > 1:
        raise ValueError("ambiguous_loaded_lfm3")
    return matches[0] if matches else None


def _no_action(status, reason=None):
    return {"schema": SCHEMA, "status": status, "reason": reason,
            "model_inference_performed": False, "trial_authorized": False,
            "weight_promotion_permitted": False}


def query_loaded_lfm3(report, *, allow_local_inference, opener=None):
    """Inspect a loaded instance and request one bounded, untrusted JSON suggestion."""
    if allow_local_inference is not True:
        return _no_action("disabled_explicit_opt_in_required")
    packet = build_packet(report)
    if packet["status"] != "advisory_prompt_only":
        return _no_action("blocked_invalid_telemetry", packet.get("reason"))
    opener = opener or build_opener(ProxyHandler({}))
    try:
        inventory = _decode(opener.open(Request(MODEL_INVENTORY), timeout=5), limit=1_000_000)
        model = _loaded_lfm3(inventory)
        if model is None:
            return _no_action("skipped_no_loaded_lfm3")
        body = json.dumps({
            "model": model, "messages": packet["messages"],
            "temperature": 0.0, "max_tokens": 160, "stream": False,
        }, allow_nan=False).encode("utf-8")
        request = Request(MODEL_CHAT, data=body, headers={"Content-Type": "application/json"},
                          method="POST")
        completion = _decode(opener.open(request, timeout=50), limit=32768)
        if not isinstance(completion, dict) or not isinstance(completion.get("choices"), list):
            raise ValueError("invalid_completion")
        if len(completion["choices"]) != 1:
            raise ValueError("unexpected_choices")
        message = completion["choices"][0].get("message")
        if not isinstance(message, dict) or type(message.get("content")) is not str:
            raise ValueError("invalid_message")
        response = json.loads(message["content"], object_pairs_hook=_unique_pairs)
        observed = observe(report)
        lrs = {name: next(a["learning_rate"] for a in report["arms"]
                          if a["optimizer"] == name) for name in ("adamw", "muon")}
        verdict = _model_suggestion(response, lrs, observed["report_sha256"])
        return {"schema": SCHEMA, "status": "advisory_model_response_reviewed",
                "model": model, "report_sha256": observed["report_sha256"],
                "model_inference_performed": True, "suggestion_review": verdict,
                "trial_authorized": False, "weight_promotion_permitted": False}
    except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError) as exc:
        return _no_action("blocked_inference_or_response", type(exc).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry_report", type=Path)
    parser.add_argument("--research-only", action="store_true", required=True)
    parser.add_argument("--allow-local-inference", action="store_true", required=True)
    args = parser.parse_args()
    try:
        result = query_loaded_lfm3(_read_json(args.telemetry_report),
                                   allow_local_inference=args.allow_local_inference)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        result = _no_action("blocked_invalid_input", type(exc).__name__)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
