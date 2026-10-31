"""Pinned ARC-Challenge capability proxy across exactly A, B, C.

Scores next-letter logits from the SAME raw-completion prompt.
Not the official ARC likelihood scoring protocol or chat benchmark.
"""
import argparse
import gc
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import pyarrow.parquet as pq

from src.three_checkpoint_eval import ARMS, checked_arm_paths
from src.refusal_probe import load_local_model
from src.research_job_guard import require_research_write_capacity

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/scott/lfm3-research/refusal-checkpoints/arc-benchmark/ARC-Challenge/test-00000-of-00001.parquet")
SOURCE_SHA256 = "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9"
SOURCE_REVISION = "210d026faf9955653af8916fad021475a3f00453"


def read_arc(path=SOURCE):
    p=Path(path)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=SOURCE_SHA256:
        raise ValueError("ARC data hash does not match pinned revision")
    rows=pq.read_table(p).to_pylist()
    if len(rows)!=1172 or len(set(r["id"] for r in rows)) != len(rows):
        raise ValueError("ARC test corpus not as expected")
    return rows


def normalize_case(row):
    labels=row["choices"]["label"]
    choices=row["choices"]["text"]
    if len(labels)!=len(choices) or not 2<=len(labels)<=5:
        raise ValueError("invalid ARC question options")
    if row["answerKey"] not in labels:
        raise ValueError("answerKey not present in labels")
    canonical="ABCDE"[:len(choices)]
    prompt=("Question: "+row["question"].strip()+"\n"
            +"\n".join(f"{letter}. {text.strip()}" for letter,text in zip(canonical,choices))
            +"\nAnswer:")
    correct=canonical[labels.index(row["answerKey"])]
    return {"id":row["id"],"prompt":prompt,"correct":correct,"labels":canonical}


def candidate_ids(tokenizer, letters):
    enc={}
    for letter in letters:
        tid=tokenizer.encode(" "+letter,add_special_tokens=False)
        if len(tid)!=1:
            raise ValueError(f"not single-token option: {letter}")
        enc[letter]=tid[0]
    return enc


def evaluate_one(model,tokenizer,case):
    ids=tokenizer(case["prompt"],return_tensors="pt",add_special_tokens=False)["input_ids"]
    with torch.inference_mode():
        logits=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[0,-1,:].float()
    tokens=candidate_ids(tokenizer,case["labels"])
    letters=list(case["labels"])
    values=torch.stack([logits[tokens[l]] for l in letters])
    probabilities=torch.softmax(values,dim=0)
    best=int(probabilities.argmax().item())
    return {
        "id":case["id"],
        "correct":case["correct"],
        "predicted":letters[best],
        "correct_prediction":letters[best]==case["correct"],
        "option_probabilities":{letter:round(float(probabilities[j]),6)
                                 for j,letter in enumerate(letters)},
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sample-size",type=int,default=8)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--output",type=Path,default=ROOT/"results/arc_capability_private.json")
    a=p.parse_args()
    if not 1<=a.sample_size<=1172:
        p.error("sample-size 1..1172")
    require_research_write_capacity(a.output)
    os.umask(0o077)
    source=read_arc()
    chosen=random.Random(a.seed).sample(source,a.sample_size)
    normalized=[normalize_case(r) for r in chosen]
    hashes=checked_arm_paths(ARMS)
    output={
        "benchmark":"ARC-Challenge test; raw one-token letter-completion proxy",
        "source_sha256":SOURCE_SHA256,
        "source_revision":SOURCE_REVISION,
        "sample_size":len(normalized),
        "seed":a.seed,
        "checkpoint_sha256":hashes,
        "limitations":["not official ARC normalized option sequence loglikelihood",
                       "single arbitrary option order",
                       "raw-completion prompt chosen to avoid base chat-template mismatch",
                       "interpretation limited without full test and order swaps"],
        "results":{},
    }
    for arm,path in ARMS.items():
        model,tok=load_local_model(path)
        grades=[]
        for row in normalized:
            grades.append(evaluate_one(model,tok,row))
        output["results"][arm]={
            "correct":sum(x["correct_prediction"] for x in grades),
            "total":len(grades),
            "accuracy_proxy":sum(x["correct_prediction"] for x in grades)/len(grades),
            "private_cases":grades,
        }
        print(f"{arm}: {output['results'][arm]['correct']}/{len(grades)}",flush=True)
        del model,tok
        gc.collect()
    a.output.parent.mkdir(exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+"\n")
    os.chmod(a.output,0o600)
    print("Private capability proxy written; no model weights or prompts published")


if __name__=="__main__":
    main()
