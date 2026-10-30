"""Conservatively summarize real but NONSTANDARD limited local ARC runs.

The official full 1,172-question three-checkpoint benchmark remains pending.
No model execution, generation, grading, or model file change occurs here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import yaml

from src.official_arc_readiness import ARMS_EXPECTED, HARNESS_VERSION, SOURCE_SHA256
from src.three_checkpoint_eval import ARMS, checked_arm_paths

TASK="arc_challenge_local_pinned"
N=40


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda:fh.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def summarize(runs,task_yaml,checkpoint_hashes):
    task_yaml=Path(task_yaml)
    y=yaml.safe_load(task_yaml.read_text())
    if (y.get("task")!=TASK or y.get("dataset_path")!="parquet"
            or y.get("dataset_name","UNSET") is not None
            or y.get("training_split","UNSET") is not None
            or y.get("validation_split","UNSET") is not None
            or y.get("test_split")!="test"):
        raise ValueError("unexpected locally pinned ARC task transport")
    datafiles=y.get("dataset_kwargs",{}).get("data_files",{})
    if set(datafiles)!={"test"}:
        raise ValueError("local adapter must have exactly pinned test parquet")
    data_path=Path(datafiles["test"])
    if digest(data_path)!=SOURCE_SHA256:
        raise ValueError("source dataset hash changed")
    if tuple(runs)!=tuple(ARMS_EXPECTED) or set(checkpoint_hashes)!=set(ARMS_EXPECTED):
        raise ValueError("exact original three checkpoints required")
    if len(set(checkpoint_hashes.values()))!=3:
        raise ValueError("model aliases or hash collisions")
    out={}
    seeds=None
    for arm in ARMS_EXPECTED:
        source=runs[arm]
        cfg=source.get("config",{})
        args=cfg.get("model_args",{})
        if (source.get("lm_eval_version")!=HARNESS_VERSION
                or cfg.get("model")!="hf"
                or not isinstance(args,dict)
                or Path(args.get("pretrained","")).resolve()!=ARMS[arm].resolve()
                or args.get("dtype")!="bfloat16"
                or cfg.get("device")!="cpu"
                or cfg.get("batch_size")!="1"
                or cfg.get("limit")!=float(N)
                or source.get("n-shot",{}).get(TASK)!=0):
            raise ValueError("model, version, precision or limited-run protocol mismatch")
        run_seeds=tuple(cfg.get(k) for k in (
            "random_seed","numpy_seed","torch_seed","fewshot_seed"
        ))
        if any(type(v) is not int for v in run_seeds):
            raise ValueError("missing deterministic benchmark seed")
        if seeds is None:
            seeds=run_seeds
        if seeds!=run_seeds:
            raise ValueError("inconsistent seeds across arms")
        sizes=source.get("n-samples",{}).get(TASK,{})
        if sizes.get("original")!=1172 or sizes.get("effective")!=N:
            raise ValueError("not exactly first 40 of 1172 questions")
        task=source.get("configs",{}).get(TASK,{})
        if (task.get("task")!=TASK
                or task.get("dataset_path")!="parquet"
                or task.get("dataset_kwargs",{}).get("data_files")!=datafiles
                or task.get("test_split")!="test"
                or task.get("output_type")!="multiple_choice"
                or task.get("doc_to_text")!="Question: {{question}}\nAnswer:"
                or task.get("doc_to_target")!="{{choices.label.index(answerKey)}}"
                or task.get("doc_to_choice")!="{{choices.text}}"
                or task.get("target_delimiter")!=" "):
            raise ValueError("local adapter scoring/prompt semantics mismatch")
        row=source.get("results",{}).get(TASK,{})
        if row.get("sample_len")!=N:
            raise ValueError("reported sample length does not match")
        metrics={}
        for label in ("acc","acc_norm"):
            value=row.get(f"{label},none")
            error=row.get(f"{label}_stderr,none")
            if (type(value) not in (float,int) or not math.isfinite(value)
                    or not 0<=value<=1 or type(error) not in (float,int)
                    or not math.isfinite(error) or not 0<=error<=1):
                raise ValueError("invalid score or standard error")
            if abs(round(value*N)-(value*N))>1e-8:
                raise ValueError("score is incompatible with count of 40")
            metrics[label]={"observed_correct":round(value*N),
                            "n":N,"fraction":value,"standard_error":error}
        out[arm]=metrics
    return {
        "evidence_class":"REAL local pinned ARC-Challenge 40-item deterministic CPU SMOKE only",
        "not_full_official_benchmark":True,
        "task_name":TASK,
        "harness":"EleutherAI lm-eval",
        "harness_version":HARNESS_VERSION,
        "task_transport":"official ARC scoring semantics inherited; local pinned parquet substituted for uncached Hub transport",
        "adapter_yaml_sha256":digest(task_yaml),
        "dataset_sha256":SOURCE_SHA256,
        "checkpoint_sha256":checkpoint_hashes,
        "n_questions_original_split":1172,
        "n_questions_scored_per_arm":N,
        "selection":"FIRST 40; not randomized and not representative",
        "device":"cpu","precision":"bfloat16",
        "seeds":dict(zip(("random","numpy","torch","fewshot"),seeds)),
        "per_arm":out,
        "comparisons_descriptive_only":{
            "B_minus_A_acc_pp":100*(out[ARMS_EXPECTED[1]]["acc"]["fraction"]-out[ARMS_EXPECTED[0]]["acc"]["fraction"]),
            "B_minus_A_acc_norm_pp":100*(out[ARMS_EXPECTED[1]]["acc_norm"]["fraction"]-out[ARMS_EXPECTED[0]]["acc_norm"]["fraction"]),
        },
        "no_matched_item_significance_test":True,
        "no_human_refusal_grading":True,
        "full_1172_question_benchmark_completed":False,
        "model_release_approved":False,
        "limitations":[
            "No full benchmark or representative statistical comparison is established by a first-40 smoke test",
            "Local dataset adapter is not the unmodified canonical Hub task config",
            "No per-item paired significance tests or independent external replication have been performed",
            "Model C is not instruction-tuned; direct causal interpretation inappropriate",
            "Refusal removal and safety claims remain blocked on independent blinded human review",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task-yaml",type=Path,required=True)
    for arm in ARMS_EXPECTED:
        p.add_argument("--"+arm.replace("_","-"),dest=arm,type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("cannot overwrite previous research evidence")
    files={}
    results={}
    for arm in ARMS_EXPECTED:
        path=getattr(a,arm)
        if path.is_symlink() or not path.is_file():
            raise ValueError("result input missing or symlink")
        files[arm]=digest(path)
        results[arm]=json.loads(path.read_text())
    report=summarize(results,a.task_yaml,checked_arm_paths(ARMS))
    report["private_result_json_sha256"]=files
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({
        "measured_cases_per_arm":N,
        "raw_accuracy":{k:v["acc"]["fraction"] for k,v in report["per_arm"].items()},
        "normalized_accuracy":{k:v["acc_norm"]["fraction"] for k,v in report["per_arm"].items()},
        "full_benchmark_complete":False,
        "report":str(a.output),
    },indent=2))


if __name__=="__main__":
    main()
