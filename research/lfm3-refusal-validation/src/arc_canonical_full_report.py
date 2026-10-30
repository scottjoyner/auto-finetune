"""Private full-score custody audit for the builtin EleutherAI ARC task.

This checks real 1172-case, three-arm canonical task reports against the
frozen source data. Only aggregates and private-file SHA256 receipts leave
the local runtime. It NEVER claims behavioral safety or human grading.
"""
import argparse
import json
import math
import os
from pathlib import Path

import pyarrow.parquet as pq

from src.arc_local_full_report import (
    sha, checked_file, N, KEYS, PAIRS,
)
from src.arc_paired_exact_inference import exact_two_sided_discordant_p, holm_bonferroni
from src.official_arc_readiness import (
    ARMS_EXPECTED, HARNESS_VERSION, SOURCE_SHA256, SOURCE_REVISION,
    check_task,
)
from src.three_checkpoint_eval import ARMS, checked_arm_paths

TASK="arc_challenge"
CANONICAL_ROWS_HASH="de86c368d3a59701a5114595cefc1869edac18bd558710f930bf6b6a7a753224"


def source_contract(parquet,task_yaml,parent_yaml):
    test=checked_file(parquet)
    if sha(test)!=SOURCE_SHA256:
        raise ValueError("frozen ARC source checksum differs")
    definition=check_task(task_yaml,parent_yaml)
    docs=pq.read_table(test).to_pylist()
    if len(docs)!=N or len({d["id"] for d in docs})!=N:
        raise ValueError("not exactly 1172 distinct ARC cases")
    # Stable JSON serialization, independent of parquet byte-level encoding.
    import hashlib
    encoded=json.dumps(docs,sort_keys=True,ensure_ascii=False,
                       separators=(",",":")).encode()
    if hashlib.sha256(encoded).hexdigest()!=CANONICAL_ROWS_HASH:
        raise ValueError("source rows not canonical pinned Hub revision")
    return docs,definition


def check_one(arm,result,samples,documents):
    config=result.get("config",{})
    model=config.get("model_args",{})
    if (result.get("lm_eval_version")!=HARNESS_VERSION or
        config.get("model")!="hf" or not isinstance(model,dict) or
        Path(model.get("pretrained","")).resolve()!=ARMS[arm].resolve() or
        model.get("dtype")!="bfloat16" or
        config.get("device")!="cpu" or
        str(config.get("batch_size"))!="8" or
        config.get("limit") is not None or
        result.get("n-shot",{}).get(TASK)!=0):
        raise ValueError("canonical task run is not frozen zero-shot CPU configuration")
    seeds=tuple(config.get(n) for n in (
        "random_seed","numpy_seed","torch_seed","fewshot_seed"))
    if any(type(v) is not int for v in seeds):
        raise ValueError("missing or invalid deterministic harness seeds")
    counts=result.get("n-samples",{}).get(TASK,{})
    if counts.get("original")!=N or counts.get("effective")!=N:
        raise ValueError("canonical scoring is partial or sample counts changed")
    task=result.get("configs",{}).get(TASK,{})
    for key,value in {
        "task":TASK,"dataset_path":"allenai/ai2_arc","dataset_name":"ARC-Challenge",
        "test_split":"test","output_type":"multiple_choice",
        "doc_to_text":"Question: {{question}}\nAnswer:",
        "doc_to_target":"{{choices.label.index(answerKey)}}",
        "doc_to_choice":"{{choices.text}}",
        "target_delimiter":" ","should_decontaminate":True,
    }.items():
        if task.get(key)!=value:
            raise ValueError("unmodified canonical ARC task definition mismatch")
    if task.get("metric_list") != [
        {"metric":k,"aggregation":"mean","higher_is_better":True} for k in KEYS
    ]:
        raise ValueError("unmodified ARC scoring metrics differ")
    row=result.get("results",{}).get(TASK,{})
    if row.get("sample_len")!=N or len(samples)!=N:
        raise ValueError("reported and recorded canonical samples incomplete")
    values={}
    for index,sample in enumerate(samples):
        if (type(sample.get("doc_id")) is not int or
            sample["doc_id"]!=index or
            sample.get("filter")!="none" or
            sample.get("doc")!=documents[index]):
            raise ValueError("canonical source case changed, dropped, or reordered")
        vals={}
        for m in KEYS:
            raw=sample.get(m)
            if type(raw) not in (bool,int,float) or raw not in (0,1,False,True):
                raise ValueError("private per-case result is not binary")
            vals[m]=int(raw)
        values[index]=vals
    metrics={}
    for m in KEYS:
        total=sum(v[m] for v in values.values())
        frac=row.get(f"{m},none")
        stderr=row.get(f"{m}_stderr,none")
        if (type(frac) not in (float,int) or not math.isfinite(frac) or
            type(stderr) not in (float,int) or not math.isfinite(stderr) or
            abs(frac-total/N)>1e-10 or
            abs(stderr-math.sqrt(frac*(1-frac)/(N-1)))>1e-8):
            raise ValueError("canonical metrics don't agree with individual case outcomes")
        metrics[m]={"correct":total,"n":N,"fraction":frac,"stderr":stderr}
    return seeds,values,metrics


def build_report(runs,samples,docs,hashes,task_identity,receipts,local=None):
    if (tuple(runs)!=ARMS_EXPECTED or tuple(samples)!=ARMS_EXPECTED
            or tuple(hashes)!=ARMS_EXPECTED or len(set(hashes.values()))!=3):
        raise ValueError("only the three original checkpoint identities accepted")
    per_case={}
    metrics={}
    seeds=None
    for arm in ARMS_EXPECTED:
        seen,per_case[arm],metrics[arm]=check_one(arm,runs[arm],samples[arm],docs)
        if seeds is None:
            seeds=seen
        elif seen!=seeds:
            raise ValueError("mismatched benchmark seeds")
    contrasts=[]
    for m in KEYS:
        for left,right in PAIRS:
            correct_new=sum(1 for i in range(N) if per_case[left][i][m]==0 and per_case[right][i][m]==1)
            correct_lost=sum(1 for i in range(N) if per_case[left][i][m]==1 and per_case[right][i][m]==0)
            contrasts.append({
                "metric":m,"left":left,"right":right,
                "right_only_correct":correct_new,
                "left_only_correct":correct_lost,
                "right_minus_left_percentage_points":100*(correct_new-correct_lost)/N,
                "exploratory_exact_mcnemar_p":exact_two_sided_discordant_p(correct_new,correct_lost),
            })
    adj=holm_bonferroni([x["exploratory_exact_mcnemar_p"] for x in contrasts])
    for c,p in zip(contrasts,adj):
        c["holm_adjusted_p_six_comparisons"]=p
    baseline_comparison={}
    if local is not None:
        if tuple(local)!=ARMS_EXPECTED:
            raise ValueError("local reference cases incomplete")
        for arm in ARMS_EXPECTED:
            reference=local[arm]
            if len(reference)!=N:
                raise ValueError("local reference set missing questions")
            checked=0
            disagreements={m:0 for m in KEYS}
            for i,row in enumerate(reference):
                if row.get("doc_id")!=i or row.get("doc")!=docs[i]:
                    raise ValueError("local adapter reference questions differ")
                checked+=1
                for m in KEYS:
                    if row.get(m) != per_case[arm][i][m]:
                        disagreements[m]+=1
            baseline_comparison[arm]={
                "identical_question_count":checked,
                "different_binary_outcomes":disagreements,
                "exact_match":all(v==0 for v in disagreements.values()),
            }
    return {
        "evidence_class":"real full canonical EleutherAI arc_challenge 3-arm zero-shot evaluation",
        "harness_version":HARNESS_VERSION,
        "canonical_builtin_task_used":True,
        "hub_dataset_revision_content_matches_pinned_source":True,
        "dataset_content_sha256":CANONICAL_ROWS_HASH,
        "dataset_source_revision":SOURCE_REVISION,
        "task_yaml_checksums":{
            "challenge":task_identity["source_task_yaml_sha256"],
            "parent":task_identity["source_parent_yaml_sha256"],
        },
        "checkpoint_sha256":hashes,
        "n_questions_per_arm":N,
        "seeds":dict(zip(("random","numpy","torch","fewshot"),seeds)),
        "metrics":metrics,
        "matched_question_comparisons":contrasts,
        "canonical_vs_local_adapter_per_case":baseline_comparison,
        "private_result_and_case_sha256":receipts,
        "case_level_results_private":True,
        "human_refusal_review_complete":False,
        "harmful_actionability_review_complete":False,
        "zero_refusal_proven":False,
        "model_release_approved":False,
        "publication_approved":False,
        "limitations":[
            "On-line canonical data source current contents were independently checked against the pinned dataset revision before scoring",
            "Independent hardware replication and blinded human-review judgment remain separate incomplete requirements",
            "Nonsignificance cannot establish ability equivalence or zero refusal",
            "Pretrained C is not instruction-tuned and is not a causal control for edit B",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-parquet",type=Path,required=True)
    p.add_argument("--official-task-yaml",type=Path,required=True)
    p.add_argument("--official-parent-yaml",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    for arm in ARMS_EXPECTED:
        key="--"+arm.replace("_","-")
        p.add_argument(key+"-results",dest=arm+"_results",type=Path,required=True)
        p.add_argument(key+"-samples",dest=arm+"_samples",type=Path,required=True)
        p.add_argument(key+"-local-samples",dest=arm+"_local",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to replace previous full benchmark evidence")
    docs,task_identity=source_contract(args.source_parquet,args.official_task_yaml,args.official_parent_yaml)
    runs={}
    samples={}
    baseline={}
    receipts={}
    for arm in ARMS_EXPECTED:
        r=checked_file(getattr(args,arm+"_results"))
        c=checked_file(getattr(args,arm+"_samples"))
        b=checked_file(getattr(args,arm+"_local"))
        runs[arm]=json.loads(r.read_text())
        with c.open() as f:samples[arm]=[json.loads(x) for x in f if x.strip()]
        with b.open() as f:baseline[arm]=[json.loads(x) for x in f if x.strip()]
        receipts[arm]={"result_sha256":sha(r),"case_sha256":sha(c),"local_case_sha256":sha(b)}
    report=build_report(runs,samples,docs,checked_arm_paths(ARMS),task_identity,receipts,baseline)
    os.umask(0o077)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:f.write(json.dumps(report,indent=2)+"\n")
    args.output.chmod(0o600)
    print(json.dumps({"valid_canonical_arms":len(report["metrics"]),
                      "n_questions":N,"all_local_binary_outcomes_identical":all(x["exact_match"] for x in report["canonical_vs_local_adapter_per_case"].values()),
                      "human_review_completed":False,"release_approved":False,"aggregate":str(args.output)},indent=2))

if __name__=="__main__":
    main()
