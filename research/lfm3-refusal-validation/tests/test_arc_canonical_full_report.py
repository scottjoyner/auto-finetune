"""Entirely synthetic tests for real canonical-score evidence intake."""
import copy
import math

import pytest

import src.arc_canonical_full_report as canonical
from src.official_arc_readiness import ARMS_EXPECTED
from src.three_checkpoint_eval import ARMS


@pytest.fixture()
def fixtures(tmp_path,monkeypatch):
    monkeypatch.setattr(canonical,"N",3)
    docs=[{
        "id":f"mock-{i}",
        "question":f"synthetic placeholder question {i}",
        "choices":{"label":["A","B"],"text":["placeholder one","placeholder two"]},
        "answerKey":"A",
    } for i in range(3)]
    runs={}
    samples={}
    refs={}
    hashes={}
    for j,arm in enumerate(ARMS_EXPECTED):
        model=tmp_path/arm
        model.mkdir()
        monkeypatch.setitem(ARMS,arm,model)
        hashes[arm]=f"test-checkpoint-{j}"
        data=[]
        for i in range(3):
            data.append({
                "doc_id":i,"filter":"none","doc":copy.deepcopy(docs[i]),
                "acc":int(i==j),
                "acc_norm":int(i!=2),
            })
        samples[arm]=data
        refs[arm]=copy.deepcopy(data)
        metric={"sample_len":3}
        for metric_name in canonical.KEYS:
            average=sum(x[metric_name] for x in data)/3
            metric[metric_name+",none"]=average
            metric[metric_name+"_stderr,none"]=math.sqrt(average*(1-average)/2)
        runs[arm]={
            "lm_eval_version":"0.4.13",
            "config":{
                "model":"hf","model_args":{"pretrained":str(model),"dtype":"bfloat16"},
                "device":"cpu","batch_size":"8","limit":None,
                "random_seed":0,"numpy_seed":1234,"torch_seed":1234,"fewshot_seed":1234,
            },
            "n-shot":{canonical.TASK:0},
            "n-samples":{canonical.TASK:{"original":3,"effective":3}},
            "configs":{canonical.TASK:{
                "task":canonical.TASK,"dataset_path":"allenai/ai2_arc",
                "dataset_name":"ARC-Challenge","test_split":"test",
                "output_type":"multiple_choice",
                "doc_to_text":"Question: {{question}}\nAnswer:",
                "doc_to_target":"{{choices.label.index(answerKey)}}",
                "doc_to_choice":"{{choices.text}}",
                "target_delimiter":" ",
                "should_decontaminate":True,
                "metric_list":[
                    {"metric":x,"aggregation":"mean","higher_is_better":True}
                    for x in canonical.KEYS
                ],
            }},
            "results":{canonical.TASK:metric},
        }
    hashes=dict(hashes)
    return runs,samples,refs,docs,hashes


def execute(f):
    runs,samples,refs,docs,hashes=f
    task={"source_task_yaml_sha256":"mock-task",
          "source_parent_yaml_sha256":"mock-parent"}
    receipts={arm:{"result_sha256":"faux"} for arm in ARMS_EXPECTED}
    return canonical.build_report(runs,samples,docs,hashes,task,receipts,refs)


def test_complete_synthetic_three_arm_reporting_preserves_release_hold(fixtures):
    report=execute(fixtures)
    assert report["n_questions_per_arm"]==3
    assert len(report["matched_question_comparisons"])==6
    assert report["canonical_builtin_task_used"] is True
    assert report["human_refusal_review_complete"] is False
    assert report["model_release_approved"] is False
    assert all(x["exact_match"] for x in report["canonical_vs_local_adapter_per_case"].values())


@pytest.mark.parametrize("change",[
    "limit","count","incorrect_task","wrong_dataset","wrong_model","bad_seed",
    "bad_metric","bad_stderr","altered_prompt","reordered_case","missing_case",
    "tampered_answer","bad_binary","partial_local","local_misorder",
])
def test_invalid_or_partial_receipt_rejected(fixtures,change):
    runs,samples,refs,docs,hashes=fixtures
    arm=ARMS_EXPECTED[1]
    run=runs[arm]
    if change=="limit":run["config"]["limit"]=40
    if change=="count":run["n-samples"][canonical.TASK]["effective"]=2
    if change=="incorrect_task":run["configs"][canonical.TASK]["task"]="arc_easy"
    if change=="wrong_dataset":run["configs"][canonical.TASK]["dataset_name"]="ARC-Easy"
    if change=="wrong_model":run["config"]["model_args"]["pretrained"]=runs[ARMS_EXPECTED[0]]["config"]["model_args"]["pretrained"]
    if change=="bad_seed":run["config"]["random_seed"]=3
    if change=="bad_metric":run["results"][canonical.TASK]["acc,none"]=float("nan")
    if change=="bad_stderr":run["results"][canonical.TASK]["acc_stderr,none"]=.8
    if change=="altered_prompt":run["configs"][canonical.TASK]["doc_to_text"]="wrong"
    if change=="reordered_case":samples[arm][0],samples[arm][1]=samples[arm][1],samples[arm][0]
    if change=="missing_case":samples[arm].pop()
    if change=="tampered_answer":samples[arm][0]["doc"]["answerKey"]="B"
    if change=="bad_binary":samples[arm][0]["acc"]=.5
    if change=="partial_local":refs[arm].pop()
    if change=="local_misorder":refs[arm][0]["doc_id"]=1
    with pytest.raises(ValueError):
        execute(fixtures)


def test_detects_canonical_local_scoring_difference_without_fake_equivalence(fixtures):
    runs,samples,refs,docs,hashes=fixtures
    refs[ARMS_EXPECTED[0]][0]["acc"]=0
    report=execute(fixtures)
    assert report["canonical_vs_local_adapter_per_case"][ARMS_EXPECTED[0]]["exact_match"] is False
    assert report["canonical_vs_local_adapter_per_case"][ARMS_EXPECTED[0]]["different_binary_outcomes"]["acc"]==1
    assert report["model_release_approved"] is False


def test_same_checkpoint_hash_aliases_rejected(fixtures):
    runs,samples,refs,docs,hashes=fixtures
    hashes[ARMS_EXPECTED[1]]=hashes[ARMS_EXPECTED[0]]
    with pytest.raises(ValueError,match="three original"):
        execute(fixtures)


def test_source_contract_checks_content_sha_and_yaml(tmp_path,monkeypatch):
    import hashlib,json
    import pyarrow as pa
    import pyarrow.parquet as pq
    docs=[{"id":"mock-0","answerKey":"A",
           "choices":{"label":["A"],"text":["placeholder"]},
           "question":"fake test"}]
    parquet=tmp_path/"test.parquet"
    pq.write_table(pa.Table.from_pylist(docs),parquet)
    challenge=tmp_path/"arc_challenge.yaml"
    easy=tmp_path/"arc_easy.yaml"
    challenge.write_text("stub task")
    easy.write_text("stub parent")
    monkeypatch.setattr(canonical,"N",1)
    monkeypatch.setattr(canonical,"SOURCE_SHA256",canonical.sha(parquet))
    monkeypatch.setattr(canonical,"CANONICAL_ROWS_HASH",
        hashlib.sha256(json.dumps(docs,sort_keys=True,ensure_ascii=False,
                      separators=(",",":")).encode()).hexdigest())
    monkeypatch.setattr(canonical,"check_task",
        lambda _c,_e:{"source_task_yaml_sha256":"fake",
                       "source_parent_yaml_sha256":"fake"})
    rows,identity=canonical.source_contract(parquet,challenge,easy)
    assert rows==docs and identity["source_task_yaml_sha256"]=="fake"
    monkeypatch.setattr(canonical,"CANONICAL_ROWS_HASH","changed")
    with pytest.raises(ValueError,match="canonical pinned"):
        canonical.source_contract(parquet,challenge,easy)
