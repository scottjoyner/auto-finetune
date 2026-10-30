"""Private full ARC-Challenge answer-text loglikelihood evaluation for A/B/C.

This is a reproducible custom scoring protocol, NOT an official leaderboard score.
Results stay private, resumable, and contain no question text.
"""
import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from src.arc_capability_eval import read_arc, SOURCE_SHA256, SOURCE_REVISION
from src.refusal_probe import load_local_model
from src.three_checkpoint_eval import ARMS, checked_arm_paths

ROOT=Path(__file__).resolve().parents[1]
ORDER=tuple(ARMS)


def prepare(row):
    options=[str(x).strip() for x in row["choices"]["text"]]
    labels=list(row["choices"]["label"])
    if len(options)!=len(labels) or not 2<=len(options)<=5 or row["answerKey"] not in labels:
        raise ValueError("invalid question options")
    if not all(options):
        raise ValueError("blank option")
    return {
        "id":row["id"],
        "prompt":"Question: "+row["question"].strip()+"\nAnswer:",
        "correct_index":labels.index(row["answerKey"]),
        "options":options,
    }


def score_case(model, tokenizer, case):
    """Scores P(' '+option_text | prompt) by sum and mean token log-prob."""
    prefix=tokenizer.encode(case["prompt"],add_special_tokens=False)
    if not prefix:
        raise ValueError("empty prompt tokenization")
    suffixes=[tokenizer.encode(" "+opt,add_special_tokens=False) for opt in case["options"]]
    if any(not ids for ids in suffixes):
        raise ValueError("empty option tokenization")
    n=len(suffixes)
    max_length=len(prefix)+max(len(s) for s in suffixes)
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("tokenizer lacks pad/eos")
    batch=torch.full((n,max_length),int(pad),dtype=torch.long)
    mask=torch.zeros((n,max_length),dtype=torch.long)
    for i,suffix in enumerate(suffixes):
        seq=prefix+suffix
        batch[i,:len(seq)]=torch.tensor(seq,dtype=torch.long)
        mask[i,:len(seq)]=1
    with torch.inference_mode():
        logits=model(input_ids=batch,attention_mask=mask,use_cache=False,logits_to_keep=0).logits
    if logits.shape[:2] != batch.shape:
        raise ValueError("incomplete logits from model forward")
    raw=[]
    norm=[]
    for i,suffix in enumerate(suffixes):
        start=len(prefix)-1
        scorelogits=logits[i,start:start+len(suffix)].float()
        target=torch.tensor(suffix,dtype=torch.long)
        if scorelogits.shape[0]!=len(target):
            raise ValueError("choice logits length mismatch")
        selected=scorelogits.gather(-1,target.unsqueeze(-1)).squeeze(-1)
        logprob=float((selected-torch.logsumexp(scorelogits,dim=-1)).sum().item())
        raw.append(logprob)
        norm.append(logprob/len(suffix))
    raw_best=max(range(n),key=lambda i:raw[i])
    norm_best=max(range(n),key=lambda i:norm[i])
    return {
        "id":case["id"],"correct_index":case["correct_index"],
        "predicted_raw_index":raw_best,"predicted_length_normalized_index":norm_best,
        "raw_correct":raw_best==case["correct_index"],
        "length_normalized_correct":norm_best==case["correct_index"],
        "candidate_token_lengths":[len(x) for x in suffixes],
        "raw_logprobs":[round(v,6) for v in raw],
        "mean_per_token_logprobs":[round(v,6) for v in norm],
    }


def manifest_for(selected,sha,size):
    return {
        "experiment":"ARC-Challenge answer-text conditional loglikelihood three-arm",
        "dataset":"allenai/ai2_arc",
        "revision":SOURCE_REVISION,
        "dataset_sha256":SOURCE_SHA256,
        "sample_size":size,
        "case_ids":[c["id"] for c in selected],
        "checkpoint_order":list(ORDER),
        "checkpoint_sha256":sha,
        "precision":"bf16",
        "device":"cpu",
        "prompt_format":"Question: <question> newline Answer: <space>choice_text",
        "scoring":"raw sum token loglikelihood and mean-per-token loglikelihood",
        "model_inference":"no tool execution; no mutation",
        "seed":17,
        "human_release_approved":False,
    }


def resume(existing,selected):
    selected_ids=[x["id"] for x in selected]
    positions={q:i for i,q in enumerate(selected_ids)}
    seen=set()
    previous=-1
    for r in existing:
        k=(r.get("arm"),r.get("id"))
        if k in seen:
            raise ValueError("duplicate ARC generation")
        seen.add(k)
        if k[0] not in ORDER or k[1] not in positions:
            raise ValueError("unknown checkpoint or ARC case")
        index=ORDER.index(k[0])*len(selected)+positions[k[1]]
        if index<=previous:
            raise ValueError("journal ordering mismatch")
        previous=index
    return seen


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sample-size",type=int,default=0,
                   help="0 selects all 1,172 test questions; otherwise seeded smoke")
    p.add_argument("--results-dir",type=Path,default=ROOT/"results/arc_full_likelihood_private")
    a=p.parse_args()
    if not 0<=a.sample_size<=1172:
        p.error("sample-size must be between 0 and 1172")
    os.umask(0o077)
    source=read_arc()
    selected=source if a.sample_size==0 else random.Random(17).sample(source,a.sample_size)
    cases=[prepare(row) for row in selected]
    hashes=checked_arm_paths(ARMS)
    manifest=manifest_for(cases,hashes,len(cases))
    a.results_dir.mkdir(parents=True,exist_ok=True)
    os.chmod(a.results_dir,0o700)
    meta=a.results_dir/"private_manifest.json"
    journal=a.results_dir/"private_cases.jsonl"
    if meta.exists():
        if json.loads(meta.read_text())!=manifest:
            raise ValueError("existing ARC run manifest mismatch; never mix runs")
    elif journal.exists():
        raise ValueError("ARC journal without manifest")
    else:
        meta.write_text(json.dumps(manifest,indent=2)+"\n")
        os.chmod(meta,0o600)
    records=[json.loads(x) for x in journal.read_text().splitlines() if x] if journal.exists() else []
    seen=resume(records,cases)
    for arm,path in ARMS.items():
        pending=[c for c in cases if (arm,c["id"]) not in seen]
        if not pending:
            continue
        print(f"Loading {arm}: {len(pending)} cases remaining",flush=True)
        model,tokenizer=load_local_model(path)
        for j,case in enumerate(pending,1):
            started=time.monotonic()
            r=score_case(model,tokenizer,case)
            r["arm"]=arm
            r["elapsed_seconds"]=round(time.monotonic()-started,3)
            with journal.open("a") as f:
                f.write(json.dumps(r)+"\n")
                f.flush()
                os.fsync(f.fileno())
            os.chmod(journal,0o600)
            records.append(r)
            if j==1 or j%25==0 or j==len(pending):
                print(f"{arm}: {j}/{len(pending)} completed",flush=True)
        del model,tokenizer
        gc.collect()
    counts={}
    for arm in ORDER:
        found=[r for r in records if r["arm"]==arm]
        counts[arm]={
            "evaluated":len(found),
            "length_normalized_correct":sum(r["length_normalized_correct"] for r in found),
            "raw_loglikelihood_correct":sum(r["raw_correct"] for r in found),
        }
    aggregate={
        "experiment":manifest["experiment"],
        "full_test_complete":len(records)==len(cases)*len(ORDER),
        "total_cases":len(records),
        "cases_per_arm":len(cases),
        "dataset_sha256":SOURCE_SHA256,
        "checkpoint_sha256":hashes,
        "per_arm":counts,
        "limitations":[
            "Custom full ARC choice-text likelihood protocol, not official lm-evaluation-harness benchmark",
            "Single prompt wording, no answer choice order permutation",
            "Length-normalized loglikelihood is not a validated safety metric",
            "Post-trained and pretrained comparisons have different learned response distributions",
            "No independent safety / human evaluation is inferred",
        ],
    }
    summary=a.results_dir/"safe_aggregate.json"
    summary.write_text(json.dumps(aggregate,indent=2)+"\n")
    os.chmod(summary,0o600)
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=="__main__":
    main()
