"""Strict private per-case audit of a full local-parquet ARC benchmark.

EleutherAI's ARC choice-scoring semantics are inherited unchanged, while
the data transport is explicitly a local pinned-parquet adapter. This is
not the unmodified Hub leaderboard task and is not refusal/safety evidence.
No model evaluation or human grading is done by this module.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from src.arc_paired_exact_inference import exact_two_sided_discordant_p, holm_bonferroni
from src.official_arc_readiness import ARMS_EXPECTED, HARNESS_VERSION, SOURCE_SHA256
from src.three_checkpoint_eval import ARMS, checked_arm_paths

TASK="arc_challenge_local_pinned"
N=1172
# Taken from pinned 20261030_official_arc_static_readiness.json.
PARENT_ARC_YAML_SHA256="96da1d9efe1df88659481cedc13985cb347dc068554a2eae561303fa39bfab3f"
KEYS=("acc","acc_norm")
PAIRS=((ARMS_EXPECTED[0],ARMS_EXPECTED[1]),
       (ARMS_EXPECTED[0],ARMS_EXPECTED[2]),
       (ARMS_EXPECTED[1],ARMS_EXPECTED[2]))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fd:
        for block in iter(lambda:fd.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def checked_file(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("result file missing or symlink")
    return path


def source_contract(task_yaml,source_parquet):
    task_yaml=checked_file(task_yaml)
    source_parquet=checked_file(source_parquet)
    definition=yaml.safe_load(task_yaml.read_text())
    required={"task":TASK,"dataset_path":"parquet","dataset_name":None,
              "training_split":None,"validation_split":None,"test_split":"test"}
    if any(definition.get(key,"MISSING")!=value for key,value in required.items()):
        raise ValueError("adapted task differs from pinned local contract")
    adapter_file=definition.get("dataset_kwargs",{}).get("data_files",{})
    if set(adapter_file)!={"test"} or Path(adapter_file["test"]).resolve()!=source_parquet.resolve():
        raise ValueError("task adapter must use only the pinned test parquet")
    parent=Path(definition.get("include",""))
    if (not parent.is_file() or parent.name!="arc_easy.yaml"
            or sha(parent)!=PARENT_ARC_YAML_SHA256):
        raise ValueError("missing or changed official inherited ARC scoring YAML")
    if sha(source_parquet)!=SOURCE_SHA256:
        raise ValueError("source test parquet SHA256 changed")
    reference=pq.read_table(source_parquet).to_pylist()
    if len(reference)!=N or len({r["id"] for r in reference})!=N:
        raise ValueError("source ARC questions not unique or complete")
    return definition,reference,{"dataset_sha256":sha(source_parquet),
                                "adapter_sha256":sha(task_yaml),
                                "official_parent_yaml_sha256":sha(parent)}


def _metric(value,name):
    if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError(f"missing/nonfinite {name}")
    return float(value)


def check_arm(arm,result,samples,definition,source):
    cfg=result.get("config",{})
    args=cfg.get("model_args",{})
    if (result.get("lm_eval_version")!=HARNESS_VERSION or
        cfg.get("model")!="hf" or not isinstance(args,dict) or
        Path(args.get("pretrained","")).resolve()!=ARMS[arm].resolve() or
        args.get("dtype")!="bfloat16" or
        cfg.get("device")!="cpu" or
        str(cfg.get("batch_size"))!="8" or
        cfg.get("limit") is not None or
        result.get("n-shot",{}).get(TASK)!=0):
        raise ValueError("wrong evaluator, model, precision, zero-shot, or full-run settings")
    seeds=tuple(cfg.get(k) for k in ("random_seed","numpy_seed","torch_seed","fewshot_seed"))
    if any(type(value) is not int for value in seeds):
        raise ValueError("incomplete evaluator seed")
    counts=result.get("n-samples",{}).get(TASK,{})
    if counts.get("original")!=N or counts.get("effective")!=N:
        raise ValueError("not exactly full ARC test split")
    task=result.get("configs",{}).get(TASK,{})
    for k,value in {
        "task":TASK,"dataset_path":"parquet","test_split":"test",
        "output_type":"multiple_choice",
        "doc_to_text":"Question: {{question}}\nAnswer:",
        "doc_to_target":"{{choices.label.index(answerKey)}}",
        "doc_to_choice":"{{choices.text}}",
        "target_delimiter":" ",
    }.items():
        if task.get(k)!=value:
            raise ValueError("model output scoring task changed")
    if task.get("dataset_kwargs",{}).get("data_files")!=definition["dataset_kwargs"]["data_files"]:
        raise ValueError("task data transport differs by checkpoint")
    if task.get("metric_list") != [
        {"metric":name,"aggregation":"mean","higher_is_better":True}
        for name in KEYS
    ] or task.get("should_decontaminate") is not True:
        raise ValueError("official inherited ARC scoring metrics changed")

    scores=result.get("results",{}).get(TASK,{})
    if scores.get("sample_len")!=N or len(samples)!=N:
        raise ValueError("incomplete reported or actual sample count")
    choices={}
    for index,row in enumerate(samples):
        if (type(row.get("doc_id")) is not int or row["doc_id"]!=index or
            row.get("filter")!="none" or row.get("doc")!=source[index]):
            raise ValueError("missing, reordered or changed private ARC case")
        case={}
        for name in KEYS:
            v=row.get(name)
            if type(v) not in (int,float,bool) or v not in (0,1,False,True):
                raise ValueError("per-case score not binary")
            case[name]=int(v)
        choices[index]=case
    metrics={}
    for name in KEYS:
        observed=sum(row[name] for row in choices.values())
        official=_metric(scores.get(f"{name},none"),name)
        stderr=_metric(scores.get(f"{name}_stderr,none"),name+"_stderr")
        if (abs(official-observed/N)>1e-10 or
            abs(stderr-math.sqrt(official*(1-official)/(N-1)))>1e-8):
            raise ValueError("reported scores or uncertainty do not match private samples")
        metrics[name]={"correct":observed,"n":N,"fraction":official,"stderr":stderr}
    return seeds,choices,metrics


def build_report(runs,samples,definition,reference,hashes,artifact_hashes):
    if list(runs)!=list(ARMS_EXPECTED) or list(samples)!=list(ARMS_EXPECTED):
        raise ValueError("only original three arms accepted, in frozen order")
    if list(hashes)!=list(ARMS_EXPECTED) or len(set(hashes.values()))!=3:
        raise ValueError("invalid checkpoint hash identities")
    seeds=None
    choices={}
    metrics={}
    for arm in ARMS_EXPECTED:
        seen,choices[arm],metrics[arm]=check_arm(
            arm,runs[arm],samples[arm],definition,reference)
        if seeds is None:
            seeds=seen
        if seeds!=seen:
            raise ValueError("cross-arm scoring seed mismatch")
    comparisons=[]
    for name in KEYS:
        for left,right in PAIRS:
            a=choices[left]
            b=choices[right]
            n01=sum(1 for i in range(N) if a[i][name]==0 and b[i][name]==1)
            n10=sum(1 for i in range(N) if a[i][name]==1 and b[i][name]==0)
            comparisons.append({
                "metric":name,"left":left,"right":right,"paired_questions":N,
                "right_only_correct":n01,"left_only_correct":n10,
                "right_minus_left_percentage_points":100*(n01-n10)/N,
                "exploratory_exact_mcnemar_p":exact_two_sided_discordant_p(n01,n10),
            })
    corrected=holm_bonferroni([r["exploratory_exact_mcnemar_p"] for r in comparisons])
    for row,adj in zip(comparisons,corrected):
        row["holm_p_six_metric_by_contrast_tests"]=adj
    return {
        "evidence_class":"actual full 1172-case locally pinned ARC-Challenge multiple-choice likelihood scoring",
        "local_parquet_adapter_used":True,
        "canonical_unmodified_hub_task_used":False,
        "benchmark_harness":"EleutherAI lm-eval",
        "harness_version":HARNESS_VERSION,
        "n_questions_per_arm":N,
        "checkpoint_count":3,
        "checkpoint_sha256":hashes,
        "private_raw_result_and_sample_sha256":artifact_hashes,
        "seeds":dict(zip(("random","numpy","torch","fewshot"),seeds)),
        "metrics":metrics,
        "matched_question_comparisons":comparisons,
        "source_documents_verified_against_pinned_parquet":True,
        "human_refusal_review_complete":False,
        "zero_refusal_supported":False,
        "canonical_official_benchmark_complete":False,
        "model_release_approved":False,
        "limitations":[
            "Dataset transport adapted from canonical Hub task; do not describe as official leaderboard score",
            "Same test split, prompt and choice-scoring rules inherited, but no independent hardware reproduction",
            "Multiple exploratory comparisons corrected; significant effects need independent replication",
            "Original pretrained C is not instruction tuned; differences are not causal editing effects",
            "Human refusal, helpfulness and harmful-actionability grading remain uncompleted",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task-yaml",required=True,type=Path)
    p.add_argument("--parquet",required=True,type=Path)
    for arm in ARMS_EXPECTED:
        p.add_argument("--"+arm.replace("_","-")+"-results",dest=arm+"_results",required=True,type=Path)
        p.add_argument("--"+arm.replace("_","-")+"-samples",dest=arm+"_samples",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("cannot overwrite full benchmark aggregate")
    definition,reference,source_hashes=source_contract(a.task_yaml,a.parquet)
    runs={}
    samples={}
    artifact_hashes={}
    for arm in ARMS_EXPECTED:
        result=checked_file(getattr(a,arm+"_results"))
        sample=checked_file(getattr(a,arm+"_samples"))
        runs[arm]=json.loads(result.read_text())
        with sample.open() as f:
            samples[arm]=[json.loads(line) for line in f if line.strip()]
        artifact_hashes[arm]={"results_sha256":sha(result),"samples_sha256":sha(sample)}
    report=build_report(runs,samples,definition,reference,checked_arm_paths(ARMS),artifact_hashes)
    report.update(source_hashes)
    os.umask(0o077)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f:
        f.write(json.dumps(report,indent=2)+"\n")
    a.output.chmod(0o600)
    print(json.dumps({"n_cases":N,
                      "arms":{name:v["acc"]["fraction"] for name,v in report["metrics"].items()},
                      "canonical_unmodified_hub_task":False,
                      "human_refusal_review_complete":False,
                      "output":str(a.output)},indent=2))


if __name__=="__main__":
    main()
