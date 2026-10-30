"""Synthetic-only tests for nonofficial first-40 report intake."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

import src.arc_local_smoke_report as smoke
from src.official_arc_readiness import ARMS_EXPECTED
from src.three_checkpoint_eval import ARMS


def inputs(tmp_path, monkeypatch):
    path=tmp_path/"tiny-dataset.parquet"
    path.write_bytes(b"synthetic data; not ARC questions")
    monkeypatch.setattr(smoke,"SOURCE_SHA256",smoke.digest(path))
    config={
        "include":"official arc_easy parent only",
        "task":smoke.TASK,
        "dataset_path":"parquet",
        "dataset_name":None,
        "dataset_kwargs":{"data_files":{"test":str(path)}},
        "training_split":None,"validation_split":None,"test_split":"test",
    }
    yml=tmp_path/"arc_challenge_local_pinned.yaml"
    yml.write_text(yaml.safe_dump(config))
    results={}
    for arm,score in zip(ARMS_EXPECTED,(8,7,16)):
        results[arm]={
            "lm_eval_version":smoke.HARNESS_VERSION,
            "config":{
                "model":"hf",
                "model_args":{"pretrained":str(ARMS[arm]),"dtype":"bfloat16"},
                "device":"cpu","batch_size":"1","limit":40.0,
                "random_seed":0,"numpy_seed":1234,
                "torch_seed":1234,"fewshot_seed":1234,
            },
            "n-shot":{smoke.TASK:0},
            "n-samples":{smoke.TASK:{"original":1172,"effective":40}},
            "configs":{smoke.TASK:{
                "task":smoke.TASK,
                "dataset_path":"parquet",
                "dataset_kwargs":{"data_files":{"test":str(path)}},
                "test_split":"test",
                "output_type":"multiple_choice",
                "doc_to_text":"Question: {{question}}\nAnswer:",
                "doc_to_target":"{{choices.label.index(answerKey)}}",
                "doc_to_choice":"{{choices.text}}",
                "target_delimiter":" ",
            }},
            "results":{smoke.TASK:{
                "sample_len":40,
                "acc,none":score/40,
                "acc_stderr,none":0.06,
                "acc_norm,none":(score+2)/40,
                "acc_norm_stderr,none":0.07,
            }},
        }
    hashes={arm:f"TEST_SHA256_{i}" for i,arm in enumerate(ARMS_EXPECTED)}
    return results,yml,hashes


def test_valid_fixture_is_explicitly_nonofficial(tmp_path,monkeypatch):
    rows,yml,hashes=inputs(tmp_path,monkeypatch)
    r=smoke.summarize(rows,yml,hashes)
    assert r["n_questions_scored_per_arm"]==40
    assert r["not_full_official_benchmark"] is True
    assert r["model_release_approved"] is False
    assert r["per_arm"][ARMS_EXPECTED[1]]["acc"]["observed_correct"]==7
    assert r["comparisons_descriptive_only"]["B_minus_A_acc_pp"]==pytest.approx(-2.5)


@pytest.mark.parametrize("tamper",[
    "wrong_model","short_run","wrong_task","wrong_seeds",
    "bad_metric","wrong_precision","full_run_claim","bad_prompt",
])
def test_bad_source_or_incomplete_results_rejected(tmp_path,monkeypatch,tamper):
    rows,yml,hashes=inputs(tmp_path,monkeypatch)
    r=rows[ARMS_EXPECTED[1]]
    if tamper=="wrong_model":
        r["config"]["model_args"]["pretrained"]=str(ARMS[ARMS_EXPECTED[0]])
    elif tamper=="short_run":
        r["n-samples"][smoke.TASK]["effective"]=39
    elif tamper=="wrong_task":
        r["configs"][smoke.TASK]["dataset_path"]="allenai/ai2_arc"
    elif tamper=="wrong_seeds":
        r["config"]["fewshot_seed"]=6
    elif tamper=="bad_metric":
        r["results"][smoke.TASK]["acc,none"]=float("nan")
    elif tamper=="wrong_precision":
        r["config"]["model_args"]["dtype"]="float16"
    elif tamper=="full_run_claim":
        r["config"]["limit"]=None
    elif tamper=="bad_prompt":
        r["configs"][smoke.TASK]["doc_to_text"]="a different prompt"
    with pytest.raises(ValueError):
        smoke.summarize(rows,yml,hashes)


def test_original_pinned_data_sha_required(tmp_path,monkeypatch):
    rows,yml,hashes=inputs(tmp_path,monkeypatch)
    monkeypatch.setattr(smoke,"SOURCE_SHA256","unknown")
    with pytest.raises(ValueError,match="hash"):
        smoke.summarize(rows,yml,hashes)


def test_aliasing_model_hashes_rejected(tmp_path,monkeypatch):
    rows,yml,hashes=inputs(tmp_path,monkeypatch)
    hashes[ARMS_EXPECTED[1]]=hashes[ARMS_EXPECTED[0]]
    with pytest.raises(ValueError,match="aliases"):
        smoke.summarize(rows,yml,hashes)
