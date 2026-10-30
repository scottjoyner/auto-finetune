"""Read-only reproducibility audit for EleutherAI ARC-Challenge.

Only validates the canonical task YAML, pinned local ARC dataset identity,
and the study's frozen checkpoints. It performs NO model execution and
cannot substitute for benchmark metrics or independent human grading.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from src.arc_capability_eval import SOURCE_SHA256, SOURCE_REVISION
from src.three_checkpoint_eval import ARMS, checked_arm_paths

TASK_SPEC = {
    "dataset_path": "allenai/ai2_arc",
    "dataset_name": "ARC-Challenge",
    "output_type": "multiple_choice",
    "test_split": "test",
    "doc_to_text": "Question: {{question}}\nAnswer:",
    "doc_to_target": "{{choices.label.index(answerKey)}}",
    "doc_to_choice": "{{choices.text}}",
    "should_decontaminate": True,
}
REQUIRED_METRICS = ("acc", "acc_norm")
ARMS_EXPECTED = (
    "A_original_posttrained",
    "B_M1_abliterated_smoke",
    "C_base_pretrained",
)
HARNESS_VERSION = "0.4.13"
EXPECTED_CASES = 1172


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as src:
        for buf in iter(lambda:src.read(1024*1024),b""):
            h.update(buf)
    return h.hexdigest()


def check_task(challenge, easy):
    challenge=Path(challenge)
    easy=Path(easy)
    ch=yaml.safe_load(challenge.read_text())
    base=yaml.safe_load(easy.read_text())
    if not isinstance(ch,dict) or set(ch)!={"include","task","dataset_name"}:
        raise ValueError("unexpected challenge task inheritance schema")
    if ch != {"include":"arc_easy.yaml","task":"arc_challenge","dataset_name":"ARC-Challenge"}:
        raise ValueError("not the pinned EleutherAI ARC-Challenge task definition")
    expected=dict(base)
    expected.update(ch)
    if any(expected.get(k)!=v for k,v in TASK_SPEC.items()):
        raise ValueError("official ARC task field changed")
    metrics=tuple(x.get("metric") for x in expected.get("metric_list",[]))
    if metrics!=REQUIRED_METRICS:
        raise ValueError("expected separate official acc and acc_norm")
    for m in expected["metric_list"]:
        if m.get("aggregation")!="mean" or m.get("higher_is_better") is not True:
            raise ValueError("unexpected metric semantics")
    if base.get("doc_to_text","").endswith(" "):
        raise ValueError("official doc_to_text includes trailing delimiter")
    if expected.get("target_delimiter"," ")!=" ":
        raise ValueError("unexpected target delimiter")
    return {
        "task":"arc_challenge",
        "metrics":list(metrics),
        "zero_shot_eval_required":True,
        "prompt":TASK_SPEC["doc_to_text"],
        "target_delimiter":" ",
        "prompt_source":"official EleutherAI task, not custom answer-text proxy",
        "source_task_yaml_sha256":sha256(challenge),
        "source_parent_yaml_sha256":sha256(easy),
    }


def check_dataset(path, expected_sha=SOURCE_SHA256):
    path=Path(path)
    if sha256(path)!=expected_sha:
        raise ValueError("pinned ARC-Challenge test parquet SHA256 mismatch")
    table=pq.read_table(path,columns=["id","answerKey","choices"])
    data=table.to_pylist()
    if len(data)!=EXPECTED_CASES or len({r["id"] for r in data})!=EXPECTED_CASES:
        raise ValueError("not the pinned 1172 unique ARC test examples")
    formats={}
    for row in data:
        labels=row["choices"]["label"]
        options=row["choices"]["text"]
        if not 2 <= len(labels) <= 5 or len(labels)!=len(options):
            raise ValueError("invalid answer options")
        if row["answerKey"] not in labels or not all(str(x).strip() for x in options):
            raise ValueError("missing ground truth or blank answer")
        formats[str(len(options))]=formats.get(str(len(options)),0)+1
    return {
        "dataset":"allenai/ai2_arc ARC-Challenge test",
        "dataset_revision":SOURCE_REVISION,
        "local_parquet_sha256":expected_sha,
        "n_unique_questions":EXPECTED_CASES,
        "counts_by_number_of_options":formats,
        "raw_test_questions_or_answers_exported":False,
    }


def readiness(task, dataset, hashes, version):
    if version!=HARNESS_VERSION:
        raise ValueError("lm-eval version mismatch: expected 0.4.13")
    if tuple(hashes)!=ARMS_EXPECTED or len(set(hashes.values()))!=3:
        raise ValueError("frozen three checkpoint identities not satisfied")
    if task["metrics"]!=list(REQUIRED_METRICS) or dataset["n_unique_questions"]!=EXPECTED_CASES:
        raise ValueError("official task/dataset contract is incomplete")
    return {
        "kind":"static readiness only: no model inference",
        "harness":{"name":"EleutherAI lm-evaluation-harness","version":HARNESS_VERSION},
        "arc_task":task,
        "dataset":dataset,
        "checkpoint_sha256":hashes,
        "checkpoint_count":3,
        "model_weights_modified":False,
        "official_arc_scores_computed":False,
        "human_refusal_review_complete":False,
        "model_release_approved":False,
        "publication_approved":False,
        "limitations":[
            "Static task integrity is not equivalent to executing the standard evaluation",
            "A benchmark run must use the exact three identity-pinned checkpoints",
            "Official acc/acc_norm are distinct from previously custom-calculated metrics",
            "No benchmark alone can establish safety or zero refusal",
            "Independent human review and adjudication remain incomplete",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--challenge-yaml",type=Path,required=True)
    p.add_argument("--easy-yaml",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("readiness audit may not overwrite prior evidence")
    version=importlib.metadata.version("lm-eval")
    result=readiness(check_task(a.challenge_yaml,a.easy_yaml),
                     check_dataset(a.dataset),
                     checked_arm_paths(ARMS),version)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({
        "static_task_readiness":True,
        "checkpoint_count":result["checkpoint_count"],
        "dataset_questions":result["dataset"]["n_unique_questions"],
        "official_arc_scores_computed":False,
        "report":str(a.out),
    },indent=2))


if __name__=="__main__":
    main()
