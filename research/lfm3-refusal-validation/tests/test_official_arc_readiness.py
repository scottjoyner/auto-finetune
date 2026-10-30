import json
from pathlib import Path

import pytest

from src.official_arc_readiness import (
    check_task, check_dataset, readiness, TASK_SPEC, ARMS_EXPECTED,
)


def valid_task(tmp_path):
    ch=tmp_path/"arc_challenge.yaml"
    easy=tmp_path/"arc_easy.yaml"
    ch.write_text("include: arc_easy.yaml\ntask: arc_challenge\ndataset_name: ARC-Challenge\n")
    base={**TASK_SPEC,
          "dataset_name":"ARC-Easy",
          "task":"arc_easy",
          "metric_list":[
              {"metric":"acc","aggregation":"mean","higher_is_better":True},
              {"metric":"acc_norm","aggregation":"mean","higher_is_better":True},
          ]}
    import yaml
    easy.write_text(yaml.safe_dump(base))
    return ch,easy


def test_task_requires_both_standard_metrics(tmp_path):
    a,b=valid_task(tmp_path)
    report=check_task(a,b)
    assert report["metrics"]==["acc","acc_norm"]
    assert report["target_delimiter"]==" "
    assert len(report["source_task_yaml_sha256"])==64


def test_task_change_fails_closed(tmp_path):
    a,b=valid_task(tmp_path)
    s=b.read_text().replace("multiple_choice","generate_until")
    b.write_text(s)
    with pytest.raises(ValueError,match="task field changed"):
        check_task(a,b)


def test_bad_metric_fails_closed(tmp_path):
    a,b=valid_task(tmp_path)
    b.write_text(b.read_text().replace("acc_norm","exact_match"))
    with pytest.raises(ValueError,match="acc_norm"):
        check_task(a,b)


def test_dataset_hash_mismatch_blocks(tmp_path):
    p=tmp_path/"fake.parquet"
    p.write_bytes(b"not ARC")
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        check_dataset(p)


def test_three_arm_gate_and_no_inference_claim(tmp_path):
    a,b=valid_task(tmp_path)
    task=check_task(a,b)
    dataset={"n_unique_questions":1172}
    hashes={name:str(i)+"x" for i,name in enumerate(ARMS_EXPECTED)}
    report=readiness(task,dataset,hashes,"0.4.13")
    assert report["checkpoint_count"]==3
    assert report["official_arc_scores_computed"] is False
    assert report["human_refusal_review_complete"] is False
    assert report["model_release_approved"] is False


def test_wrong_version_or_missing_arm_fails(tmp_path):
    a,b=valid_task(tmp_path)
    task=check_task(a,b)
    dataset={"n_unique_questions":1172}
    hashes={name:str(i) for i,name in enumerate(ARMS_EXPECTED)}
    with pytest.raises(ValueError,match="version"):
        readiness(task,dataset,hashes,"0.4.12")
    hashes.pop(ARMS_EXPECTED[-1])
    with pytest.raises(ValueError,match="three checkpoint"):
        readiness(task,dataset,hashes,"0.4.13")
