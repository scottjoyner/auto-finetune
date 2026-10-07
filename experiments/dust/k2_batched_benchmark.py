"""Benchmark serial vs microbatched orthogonal-antithetic K2 forward-only updates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import time

from experiments.dust.k2_data import select_disjoint
from experiments.dust.k2_matched_compare import digest, load_base, read_mem_available
from experiments.dust.k2_structured_train_compare import cosine_error, run_backend


def parse_batches(v:str)->list[int]:
    out=[int(x) for x in v.split(",") if x.strip()]
    if not out or any(x not in (1,2,4,8,16) for x in out):
        raise argparse.ArgumentTypeError("batches must be comma list of 1/2/4/8/16")
    return out


def run(*,model_dir:Path,train_file:Path,heldout_file:Path,expected_sha:str,
        device:str,seed:int,population:int,sigma:float,lr:float,
        direction_batches:list[int],max_seconds:int)->dict:
    import torch
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(model_dir),trust_remote_code=True,local_files_only=True)
    train,held,dataset=select_disjoint(train_file,heldout_file,tok,
        train_count=4,eval_count=2,max_tokens=128,eval_max_tokens=512)
    torch.cuda.reset_peak_memory_stats() if device=="cuda" else None
    model,sha=load_base(model_dir,device=device,expected_sha=expected_sha)
    try:
        serial=run_backend(model,backend="orthogonal_antithetic",
            train=train,heldout=held,device=device,seed=seed,steps=1,
            population=population,sigma=sigma,lr=lr,max_seconds=max_seconds)
        rows=[]
        for batch in direction_batches:
            if read_mem_available()<1536*1024**2:
                raise MemoryError("Host memory below batch benchmark floor")
            batched=run_backend(model,backend="orthogonal_antithetic_batched",
                train=train,heldout=held,device=device,seed=seed,steps=1,
                population=population,sigma=sigma,lr=lr,max_seconds=max_seconds,
                direction_batch=batch)
            a=cosine_error(batched["final_a"]-batched["initial_a"],
                           serial["final_a"]-serial["initial_a"])
            b=cosine_error(batched["final_b"]-batched["initial_b"],
                           serial["final_b"]-serial["initial_b"])
            rows.append({
                "direction_batch":batch,
                "elapsed_seconds":batched["metrics"]["elapsed_seconds"],
                "forward_calls":batched["metrics"]["history"][0]["forward_calls"],
                "a_vs_serial":a,"b_vs_serial":b,
                "train_ce_delta":batched["metrics"]["train_ce_delta"],
                "heldout_ce_delta":batched["metrics"]["heldout_ce_delta"],
            })
        return {
            "schema":"auto-finetune.dust-k2-batched-benchmark.v1",
            "research_only":True,"production_promotion_authorized":False,
            "model_weights_sha256":sha,
            "model_config_sha256":digest(model_dir/"config.json"),
            "seed":seed,"population":population,"sigma":sigma,"learning_rate":lr,
            "dataset":dataset,
            "serial":{
                "elapsed_seconds":serial["metrics"]["elapsed_seconds"],
                "forward_calls":serial["metrics"]["history"][0]["forward_calls"],
                "train_ce_delta":serial["metrics"]["train_ce_delta"],
                "heldout_ce_delta":serial["metrics"]["heldout_ce_delta"],
            },
            "batched":rows,
            "max_device_memory_allocated":(
                torch.cuda.max_memory_allocated() if device=="cuda" else None),
            "remaining_host_mem_available_bytes":read_mem_available(),
            "checkpoint_written":False,"raw_user_data_in_report":False,
        }
    finally:
        del model


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir",type=Path,required=True)
    ap.add_argument("--train-jsonl",type=Path,required=True)
    ap.add_argument("--heldout-jsonl",type=Path,required=True)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--expected-sha256",default="6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--population",type=int,choices=(64,128,256,512,1024),default=256)
    ap.add_argument("--sigma",type=float,default=.25)
    ap.add_argument("--lr",type=float,default=.1)
    ap.add_argument("--direction-batches",type=parse_batches,default=[1,2,4,8])
    ap.add_argument("--max-seconds",type=int,default=360)
    ap.add_argument("--output",type=Path)
    a=ap.parse_args(argv)
    if a.output is not None and a.output.exists(): ap.error("Refusing overwrite")
    report=run(model_dir=a.model_dir,train_file=a.train_jsonl,heldout_file=a.heldout_jsonl,
        expected_sha=a.expected_sha256,device=a.device,seed=a.seed,population=a.population,
        sigma=a.sigma,lr=a.lr,direction_batches=a.direction_batches,max_seconds=a.max_seconds)
    enc=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if a.output is None: print(enc,end="")
    else:
        with a.output.open("x",encoding="utf-8") as f:f.write(enc)
    return 0
if __name__=="__main__": raise SystemExit(main())
