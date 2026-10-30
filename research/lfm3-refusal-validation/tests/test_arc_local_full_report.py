"""Full-run provenance tests with fabricated 3-row data, never fake scientific evidence."""
import math
from copy import deepcopy

import pytest

from src.official_arc_readiness import ARMS_EXPECTED
from src.three_checkpoint_eval import ARMS
import src.arc_local_full_report as full


@pytest.fixture()
def fixture_data(tmp_path,monkeypatch):
    monkeypatch.setattr(full,"N",3)
    for arm in ARMS_EXPECTED:
        place=tmp_path/arm
        place.mkdir()
        monkeypatch.setitem(ARMS,arm,place)
    docs=[{"id":f"test-{i}","question":f"fictitious {i}",
           "choices":{"label":["A","B"],"text":["mock1","mock2"]},
           "answerKey":"A"} for i in range(3)]
    config={"dataset_kwargs":{"data_files":{"test":"/private/test.parquet"}}}
    runs={}
    samples={}
    for j,arm in enumerate(ARMS_EXPECTED):
        entries=[]
        for i in range(3):
            entries.append({
                "doc_id":i,"filter":"none","doc":deepcopy(docs[i]),
                "acc":1 if (i+j)%3==0 else 0,
                "acc_norm":1 if i<2 else 0,
            })
        samples[arm]=entries
        scores={"sample_len":3}
        for name in full.KEYS:
            value=sum(x[name] for x in entries)/3
            scores[f"{name},none"]=value
            scores[f"{name}_stderr,none"]=math.sqrt(value*(1-value)/2)
        runs[arm]={
            "lm_eval_version":"0.4.13",
            "config":{
                "model":"hf",
                "model_args":{"pretrained":str(ARMS[arm]),"dtype":"bfloat16"},
                "device":"cpu",
                "batch_size":"8","limit":None,
                "random_seed":0,"numpy_seed":1234,
                "torch_seed":1234,"fewshot_seed":1234,
            },
            "n-shot":{full.TASK:0},
            "n-samples":{full.TASK:{"original":3,"effective":3}},
            "configs":{full.TASK:{
                "task":full.TASK,"dataset_path":"parquet",
                "dataset_kwargs":deepcopy(config["dataset_kwargs"]),
                "test_split":"test","output_type":"multiple_choice",
                "doc_to_text":"Question: {{question}}\nAnswer:",
                "doc_to_target":"{{choices.label.index(answerKey)}}",
                "doc_to_choice":"{{choices.text}}",
                "target_delimiter":" ",
                "should_decontaminate":True,
                "metric_list":[
                    {"metric":name,"aggregation":"mean","higher_is_better":True}
                    for name in full.KEYS
                ],
            }},
            "results":{full.TASK:scores},
        }
    hashes={arm:f"synthetic-hash-{i}" for i,arm in enumerate(ARMS_EXPECTED)}
    artifacts={arm:{"results_sha256":"synthetic","samples_sha256":"synthetic"} for arm in ARMS_EXPECTED}
    return runs,samples,config,docs,hashes,artifacts


def run(args):
    runs,samples,config,docs,hashes,artifacts=args
    return full.build_report(runs,samples,config,docs,hashes,artifacts)


def test_valid_fake_full_run_computes_matched_statistics(fixture_data):
    result=run(fixture_data)
    assert result["n_questions_per_arm"]==3
    assert result["checkpoint_count"]==3
    assert len(result["matched_question_comparisons"])==6
    assert result["canonical_unmodified_hub_task_used"] is False
    assert result["human_refusal_review_complete"] is False
    assert result["model_release_approved"] is False
    assert result["source_documents_verified_against_pinned_parquet"] is True


@pytest.mark.parametrize("tamper",[
    "model","limit","count","num_fewshot","precision","batch","seeds",
    "case_missing","case_changed","case_order","case_duplicate","score",
    "stderr","target","datafiles","metric_binary","metric_config",
])
def test_invalid_real_scoring_receipts_fail_closed(fixture_data,tamper):
    runs,samples,config,docs,hashes,artifacts=fixture_data
    arm=ARMS_EXPECTED[1]
    report=runs[arm]
    if tamper=="model":
        report["config"]["model_args"]["pretrained"]=str(ARMS[ARMS_EXPECTED[0]])
    elif tamper=="limit":
        report["config"]["limit"]=3
    elif tamper=="count":
        report["n-samples"][full.TASK]["effective"]=2
    elif tamper=="num_fewshot":
        report["n-shot"][full.TASK]=5
    elif tamper=="precision":
        report["config"]["model_args"]["dtype"]="float32"
    elif tamper=="batch":
        report["config"]["batch_size"]="1"
    elif tamper=="seeds":
        report["config"]["numpy_seed"]=9
    elif tamper=="case_missing":
        samples[arm].pop()
    elif tamper=="case_changed":
        samples[arm][0]["doc"]["question"]="tampered"
    elif tamper=="case_order":
        samples[arm][0],samples[arm][1]=samples[arm][1],samples[arm][0]
    elif tamper=="case_duplicate":
        samples[arm][1]["doc_id"]=0
    elif tamper=="score":
        report["results"][full.TASK]["acc,none"]=0.7
    elif tamper=="stderr":
        report["results"][full.TASK]["acc_stderr,none"]=float("nan")
    elif tamper=="target":
        report["configs"][full.TASK]["doc_to_target"]="wrong"
    elif tamper=="datafiles":
        report["configs"][full.TASK]["dataset_kwargs"]["data_files"]={"test":"/wrong"}
    elif tamper=="metric_binary":
        samples[arm][0]["acc"]=0.5
    elif tamper=="metric_config":
        report["configs"][full.TASK]["metric_list"][0]["aggregation"]="sum"
    with pytest.raises(ValueError):
        run(fixture_data)


def test_missing_or_extra_arm_rejected(fixture_data):
    runs,samples,config,docs,hashes,artifacts=fixture_data
    runs.pop(ARMS_EXPECTED[-1])
    with pytest.raises(ValueError,match="only original three arms"):
        run(fixture_data)


def test_model_hash_alias_rejected(fixture_data):
    runs,samples,config,docs,hashes,artifacts=fixture_data
    hashes[ARMS_EXPECTED[1]]=hashes[ARMS_EXPECTED[0]]
    with pytest.raises(ValueError,match="hash"):
        run(fixture_data)


def test_source_contract_verifies_real_content_and_parent_sha(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    parquet=tmp_path/"mock.parquet"
    docs=[{"id":f"id-{i}","question":"fake"} for i in range(3)]
    pq.write_table(pa.Table.from_pylist(docs),parquet)
    parent=tmp_path/"arc_easy.yaml"
    parent.write_text("source: immutable test config\n")
    source_yaml=tmp_path/"adapter.yaml"
    source_yaml.write_text(yaml.safe_dump({
        "task":full.TASK,
        "include":str(parent),
        "dataset_path":"parquet",
        "dataset_name":None,
        "dataset_kwargs":{"data_files":{"test":str(parquet)}},
        "test_split":"test",
        "training_split":None,
        "validation_split":None,
    }))
    monkeypatch.setattr(full,"N",3)
    monkeypatch.setattr(full,"SOURCE_SHA256",full.sha(parquet))
    monkeypatch.setattr(full,"PARENT_ARC_YAML_SHA256",full.sha(parent))
    y,rows,hashes=full.source_contract(source_yaml,parquet)
    assert rows==docs
    assert hashes["official_parent_yaml_sha256"]==full.sha(parent)
    parent.write_text("tampered: yes\n")
    with pytest.raises(ValueError,match="changed official"):
        full.source_contract(source_yaml,parquet)


def test_source_contract_rejects_changed_parquet_and_symlink(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    data=tmp_path/"data.parquet"
    pq.write_table(pa.Table.from_pylist([{"id":"one"}]),data)
    parent=tmp_path/"arc_easy.yaml"
    parent.write_text("actual: local mock\n")
    task=tmp_path/"adapter.yaml"
    task.write_text(yaml.safe_dump({
        "include":str(parent),"task":full.TASK,
        "dataset_path":"parquet","dataset_name":None,
        "dataset_kwargs":{"data_files":{"test":str(data)}},
        "training_split":None,"validation_split":None,"test_split":"test",
    }))
    monkeypatch.setattr(full,"N",1)
    monkeypatch.setattr(full,"SOURCE_SHA256",full.sha(data))
    monkeypatch.setattr(full,"PARENT_ARC_YAML_SHA256",full.sha(parent))
    data.write_bytes(b"tampered source")
    with pytest.raises(ValueError,match="SHA256 changed"):
        full.source_contract(task,data)
    link=tmp_path/"adapter-link.yaml"
    link.symlink_to(task)
    with pytest.raises(ValueError,match="symlink"):
        full.source_contract(link,data)
