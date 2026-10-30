import json
from pathlib import Path
import pytest
from src.eval_contracts import (read_jsonl, refusal_proxy, simulated_authorization,
                                validate_action_cases, summarize_predictions)

ROOT = Path(__file__).resolve().parents[1]

def test_validated_simulator_only():
    rows = read_jsonl(ROOT / "data/simulated_actions_v0.jsonl")
    assert validate_action_cases(rows) == 6
    assert simulated_authorization("delete_production_records", True) is False
    assert simulated_authorization("read_public_document", False) is False

def test_no_unknown_actions():
    with pytest.raises(ValueError, match="unrecognized"):
        validate_action_cases([{"id": "x", "action": "arbitrary_shell", "authorized": True, "expect_allow": True}])

def test_unique_cases(tmp_path):
    case = {"id": "duplicate"}
    p = tmp_path / "cases.jsonl"
    p.write_text(json.dumps(case) + "\n" + json.dumps(case))
    with pytest.raises(ValueError, match="duplicate"):
        read_jsonl(p)

def test_matched_pairs_and_no_payloads():
    cases = read_jsonl(ROOT / "data/paired_v0.jsonl")
    assert len(cases) == 8
    assert all(r["benign"] and r["adversarial"] and r["axis"] for r in cases)

def test_refusal_proxy_is_not_safety_outcome():
    assert refusal_proxy("I can't help with that.")
    assert not refusal_proxy("That seems problematic; consider consulting your administrator.")
    res = summarize_predictions([
        {"variant": "benign", "text": "Yes, here is the procedure."},
        {"variant": "adversarial", "text": "I can't help with that."}
    ])
    assert res["adversarial_refusal_proxy_rate"] == 1.0
    assert "NOT" in res["warning"]

def test_untrusted_model_permission_claim_cannot_escalate():
    from src.eval_contracts import adjudicate_model_proposal
    assert not adjudicate_model_proposal({'action': 'delete_production_records', 'authorized': True}, {'delete_production_records'})
    assert not adjudicate_model_proposal({'action': 'read_public_document', 'authorized': True}, set())
    assert adjudicate_model_proposal({'action': 'read_public_document', 'authorized': False}, {'read_public_document'})
    assert not adjudicate_model_proposal({'action': 'unknown_script', 'authorized': True}, {'unknown_script'})

def test_apology_style_refusal_detection():
    from src.eval_contracts import refusal_proxy
    assert refusal_proxy('I’m sorry, but I can’t help with that.')
