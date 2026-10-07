"""Low-overhead K2 serial-vs-batched orthogonal perturbation benchmark."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

from experiments.dust.k2_forward_only import FinalProjectionLoRA
from experiments.dust.k2_gradient_calibration import select_train_example
from experiments.dust.k2_matched_compare import digest, load_base, read_mem_available
from experiments.dust.k2_structured_train_compare import (
    cosine_error, structured_step, structured_step_batched,
)

def run_one(model, samples, *, device, seed, population, sigma, lr, direction_batch):
    import torch
    adapter=FinalProjectionLoRA(model,rank=4,seed=seed+17)
    frozen=adapter.module.weight.detach().clone()
    a0=adapter.a.detach().clone(); b0=adapter.b.detach().clone()
    history=[]; started=time.monotonic()
    try:
        for step,sample in enumerate(samples):
            if direction_batch is None:
                result=structured_step(
                    model,adapter,sample,device,seed=seed*100000+step,
                    population=population,sigma=sigma,lr=lr)
            else:
                result=structured_step_batched(
                    model,adapter,sample,device,seed=seed*100000+step,
                    population=population,sigma=sigma,lr=lr,
                    direction_batch=direction_batch)
            if device=="cuda": torch.cuda.synchronize()
            history.append({"step":step+1,**result})
        if not torch.equal(frozen,adapter.module.weight):
            raise AssertionError("Frozen base projection changed")
        if any(x.grad is not None for x in model.parameters()):
            raise AssertionError("Frozen base accumulated gradients")
        return {
            "elapsed_seconds":time.monotonic()-started,
            "history":history,
            "forward_calls":sum(x["forward_calls"] for x in history),
            "min_host_mem_available_bytes":min(
                x["min_host_mem_available_bytes"] for x in history),
            "a_update":adapter.a.detach().clone()-a0,
            "b_update":adapter.b.detach().clone()-b0,
        }
    finally:
        adapter.close()

def run(*,model_dir,train_file,expected_sha,device,seed,population,sigma,lr,batches,max_tokens,steps):
    import torch
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(
        str(model_dir),trust_remote_code=True,local_files_only=True)
    samples=[]; selections=[]
    for index in range(steps):
        row,meta=select_train_example(train_file,tok,index=index,max_tokens=max_tokens)
        samples.append(row); selections.append(meta)
    if device=="cuda": torch.cuda.reset_peak_memory_stats()
    model,model_sha=load_base(model_dir,device=device,expected_sha=expected_sha)
    try:
        serial=run_one(model,samples,device=device,seed=seed,population=population,
                       sigma=sigma,lr=lr,direction_batch=None)
        rows=[]
        for batch in batches:
            candidate=run_one(
                model,samples,device=device,seed=seed,population=population,
                sigma=sigma,lr=lr,direction_batch=batch)
            rows.append({
                "direction_batch":batch,
                "elapsed_seconds":candidate["elapsed_seconds"],
                "forward_calls":candidate["forward_calls"],
                "min_host_mem_available_bytes":candidate["min_host_mem_available_bytes"],
                "a_vs_serial":cosine_error(candidate["a_update"],serial["a_update"]),
                "b_vs_serial":cosine_error(candidate["b_update"],serial["b_update"]),
            })
        return {
            "schema":"auto-finetune.dust-k2-batched-kernel.v2",
            "research_only":True,"production_promotion_authorized":False,
            "model_weights_sha256":model_sha,
            "model_config_sha256":digest(model_dir/"config.json"),
            "selections":selections,"steps":steps,"seed":seed,
            "population":population,"sigma":sigma,"learning_rate":lr,
            "serial":{"elapsed_seconds":serial["elapsed_seconds"],
                      "forward_calls":serial["forward_calls"],
                      "min_host_mem_available_bytes":serial["min_host_mem_available_bytes"]},
            "batched":rows,
            "max_device_memory_allocated":(
                torch.cuda.max_memory_allocated() if device=="cuda" else None),
            "remaining_host_mem_available_bytes":read_mem_available(),
            "raw_user_data_in_report":False,"checkpoint_written":False,
        }
    finally:
        del model

def parse_batches(v):
    out=[int(x) for x in v.split(",") if x.strip()]
    if not out or any(x not in (1,2,4,8,16) for x in out):
        raise argparse.ArgumentTypeError("batches must be 1/2/4/8/16")
    return out

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir",type=Path,required=True)
    ap.add_argument("--train-jsonl",type=Path,required=True)
    ap.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    ap.add_argument("--expected-sha256",default="6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--population",type=int,choices=(256,512,1024,1536),default=1024)
    ap.add_argument("--sigma",type=float,default=.25)
    ap.add_argument("--lr",type=float,default=.1)
    ap.add_argument("--direction-batches",type=parse_batches,default=[4,8])
    ap.add_argument("--max-tokens",type=int,default=128)
    ap.add_argument("--steps",type=int,choices=(1,2),default=1)
    ap.add_argument("--output",type=Path)
    a=ap.parse_args(argv)
    if a.output is not None and a.output.exists(): ap.error("Refusing overwrite")
    report=run(model_dir=a.model_dir,train_file=a.train_jsonl,
        expected_sha=a.expected_sha256,device=a.device,seed=a.seed,
        population=a.population,sigma=a.sigma,lr=a.lr,batches=a.direction_batches,
        max_tokens=a.max_tokens,steps=a.steps)
    rendered=json.dumps(report,indent=2,sort_keys=True)+"\n"
    if a.output is None: print(rendered,end="")
    else:
        with a.output.open("x",encoding="utf-8") as f:f.write(rendered)
    return 0

if __name__=="__main__": raise SystemExit(main())
