"""All outcomes here are synthetic mocks, NOT real official ARC results."""
from pathlib import Path

import pytest

from src.official_arc_result_gate import check_one, check_all
from src.official_arc_readiness import ARMS_EXPECTED
from src.three_checkpoint_eval import ARMS


def fake_arm(arm):
    return {
        "config":{"model":"hf","model_args":f"pretrained={ARMS[arm]}",
                  "limit":None},
        "n-shot":{"arc_challenge":0},
        "configs":{"arc_challenge":{
            "dataset_path":"allenai/ai2_arc","dataset_name":"ARC-Challenge",
            "output_type":"multiple_choice","test_split":"test",
        }},
        "n-samples":{"arc_challenge":{"original":1172,"effective":1172}},
        "results":{"arc_challenge":{"sample_len":1172,
            "acc,none":0.25,"acc_stderr,none":0.012,
            "acc_norm,none":0.26,"acc_norm_stderr,none":0.013,
        }},
    }


def fixture():
    hashes={name:f"synthetic-{i}" for i,name in enumerate(ARMS_EXPECTED)}
    static={"kind":"static readiness only: no model inference",
            "harness":{"version":"0.4.13"},
            "dataset":{"n_unique_questions":1172},
            "checkpoint_sha256":dict(hashes),"checkpoint_count":3}
    runs={arm:fake_arm(arm) for arm in ARMS_EXPECTED}
    return runs,static,hashes


def test_accepted_complete_synthetic_bundle_retains_release_block():
    runs,static,hashes=fixture()
    out=check_all(runs,static,hashes)
    assert out["checkpoint_count"]==3
    assert len(out["reported_metrics"])==3
    assert out["independent_run_provenance_verified"] is False
    assert out["zero_refusal_supported"] is False
    assert out["release_approved"] is False


@pytest.mark.parametrize("change",["model","limit","shot","count","score","stderr","task"])
def test_rejects_incomplete_or_mismatched_result(change):
    runs,static,hashes=fixture()
    d=runs[ARMS_EXPECTED[1]]
    if change=="model":
        d["config"]["model_args"]=f"pretrained={ARMS[ARMS_EXPECTED[0]]}"
    if change=="limit":
        d["config"]["limit"]=1
    if change=="shot":
        d["n-shot"]["arc_challenge"]=5
    if change=="count":
        d["n-samples"]["arc_challenge"]["effective"]=1
    if change=="score":
        d["results"]["arc_challenge"]["acc_norm,none"]=float("nan")
    if change=="stderr":
        d["results"]["arc_challenge"]["acc_stderr,none"]=None
    if change=="task":
        d["configs"]["arc_challenge"]["dataset_name"]="ARC-Easy"
    with pytest.raises(ValueError):
        check_all(runs,static,hashes)


def test_rejects_missing_arm_or_swapped_order():
    runs,static,hashes=fixture()
    runs.pop(ARMS_EXPECTED[1])
    with pytest.raises(ValueError,match="three original arms"):
        check_all(runs,static,hashes)


def test_rejects_stale_model_hashes():
    runs,static,hashes=fixture()
    hashes[ARMS_EXPECTED[0]]="different"
    with pytest.raises(ValueError,match="contract"):
        check_all(runs,static,hashes)


def test_rejects_unbound_or_nonexistent_model_path():
    d=fake_arm(ARMS_EXPECTED[0])
    d["config"]["model_args"]="pretrained=/definitely/not/a/model"
    with pytest.raises((FileNotFoundError,ValueError)):
        check_one(ARMS_EXPECTED[0],d,{})
