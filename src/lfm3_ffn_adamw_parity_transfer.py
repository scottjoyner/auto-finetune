"""Once-only fictional parity transfer audit for dev-selected AdamW LR 0.004.

The LR 0.004 arm won on a reused fictional development set. This separate,
frozen, previously unevaluated question FAMILY uses four fictional parity
prompts to test out-of-family transfer against AdamW LR 0.002 and original
frozen LFM3. It does not tune any hyperparameters from holdout results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_frozen_readout_lab import TRAIN,DEV,target_tokens,assert_fixed_lab_contract
from src.lfm3_ffn_optimizer_lr_gate import gate_calibration
from src.lfm3_last_layer_lora_lab import TARGET,RANK,RESEARCH_SEEDS,install_adapter
from src.lfm3_real_checkpoint_smoke import ResearchRefused,verify_restricted_environment

PARITY=(
    ("Fictional parity task: 8 is even. Answer A for even, B for odd. Letter:", "A"),
    ("Fictional parity task: 13 is odd. Answer A for even, B for odd. Letter:", "B"),
    ("Fictional parity task: 42 is even. Answer A for even, B for odd. Letter:", "A"),
    ("Fictional parity task: 37 is odd. Answer A for even, B for odd. Letter:", "B"),
)
LRS=(0.002,0.004)
STEPS=3

def frozen_parity_contract(rows=PARITY, lrs=LRS, steps=STEPS):
    assert_fixed_lab_contract()
    if (type(rows) is not tuple or rows != PARITY or len(rows)!=4
            or type(lrs) is not tuple or lrs!=LRS or type(steps) is not int or steps!=STEPS
            or len({x[0] for x in TRAIN+DEV+rows})!=8
            or tuple(x[1] for x in rows)!=("A","B","A","B")
            or any(not p.startswith("Fictional parity task:") or len(p)>120 for p,_ in rows)):
        raise ResearchRefused("parity source/task family or study budget modified")
    return True

def run(snapshot:Path, calibration:Path, original_dev:Path):
    frozen_parity_contract()
    measured=json.loads(calibration.read_text())
    previous=json.loads(original_dev.read_text())
    external=gate_calibration(measured,previous)
    if (external["best_option_index"]!=2
            or not external["development_only_research_review_eligible"]
            or external["heldout_evaluated"] is not False):
        raise ResearchRefused("heldout admission must be fixed by development metrics only")
    verify_restricted_environment()
    proof=verify_snapshot(snapshot)
    import torch
    import transformers
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if transformers.__version__!="5.21.0":
        raise ResearchRefused("unreviewed pretrained runtime")
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(
        str(snapshot),local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16)
    if model.__class__.__name__!="Lfm3ForCausalLM":
        raise ResearchRefused("unknown LFM3 backbone")
    model.eval();model.requires_grad_(False)
    base=tuple(model.parameters())
    versions=tuple(p._version for p in base)
    layer=model.get_submodule(TARGET)
    if not isinstance(layer,torch.nn.Linear) or tuple(layer.weight.shape)!=(2048,7680):
        raise ResearchRefused("unknown last FFN tensor")
    ids=target_tokens(tok)

    def encode(rows):
        out=[]
        for p,l in rows:
            t=tok.apply_chat_template([{"role":"user","content":p}],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,
                return_dict=True,return_tensors="pt")
            if t["input_ids"].shape[1]>80:
                raise ResearchRefused("fictional parity prompt too long")
            out.append((t,ids[l]))
        return out
    train=encode(TRAIN);dev=encode(DEV);holdout=encode(PARITY)
    def loss(items):
        values=[]
        for inputs,answer in items:
            logits=model(**inputs,use_cache=False).logits[0,-1].float()
            values.append(F.cross_entropy(logits.unsqueeze(0),torch.tensor([answer])))
        return torch.stack(values).mean()
    with torch.no_grad():
        frozen_holdout=float(loss(holdout))
    results=[]
    for i,seed in enumerate(RESEARCH_SEEDS):
        for idx,lr in enumerate(LRS):
            A,B,handle=install_adapter(layer,seed=seed)
            tensors=list(A.parameters())+list(B.parameters())
            opt=torch.optim.AdamW(tensors,lr=lr,weight_decay=0.0)
            try:
                with torch.no_grad():
                    first=float(loss(dev))
                if abs(first - measured["trials"][i*5+1]["dev_loss_before"])>0.0001:
                    raise ResearchRefused("inconsistent source LFM3 baseline")
                for _ in range(STEPS):
                    opt.zero_grad(set_to_none=True)
                    objective=loss(train)
                    if not bool(torch.isfinite(objective)):
                        raise ResearchRefused("nonfinite actual model FFN loss")
                    objective.backward()
                    if (B.weight.grad is None or not bool(torch.isfinite(B.weight.grad).all())
                            or float(B.weight.grad.abs().sum())<=0):
                        raise ResearchRefused("no genuine final FFN model gradient")
                    g=torch.nn.utils.clip_grad_norm_(tensors,1.0)
                    if not bool(torch.isfinite(g)):
                        raise ResearchRefused("bad gradient norm")
                    opt.step()
                with torch.no_grad():
                    after_dev=float(loss(dev))
                expected=measured["trials"][i*5+(1 if idx==0 else 2)]["dev_loss_after"]
                if abs(after_dev-expected)>0.01:
                    raise ResearchRefused("development training result did not reproduce")
                # Only now do one holdout evaluation, with zero subsequent updates.
                with torch.no_grad():
                    after_holdout=float(loss(holdout))
                if not math.isfinite(after_holdout):
                    raise ResearchRefused("nonfinite parity holdout")
                results.append({
                    "seed":seed,"optimizer":"adamw","learning_rate":lr,
                    "three_training_steps":STEPS,
                    "dev_loss_reproduced":after_dev,"precommitted_dev_loss":expected,
                    "parity_holdout_loss":after_holdout,
                    "weight_update_after_holdout":False,
                    "adapter_saved":False,
                })
            finally:handle.remove()
    if (tuple(p._version for p in base)!=versions
            or any(p.requires_grad or p.grad is not None for p in base)):
        raise ResearchRefused("pinned pretrained model was modified")
    return {
        "schema":"sg-lfm3-real-ffn-adamw-parity-transfer-once.v1",
        "checkpoint_sha256":proof["checkpoint_sha256"],
        "model_revision":proof["revision"],
        "target_module":TARGET,
        "fixed_calibration_winner_index":2,
        "seed_count":3,"arm_count":6,
        "training_example_count":2,"dev_example_count":2,
        "parity_task_holdout_example_count":4,
        "synthetic_train_sha256":hashlib.sha256(repr(TRAIN).encode()).hexdigest(),
        "synthetic_dev_sha256":hashlib.sha256(repr(DEV).encode()).hexdigest(),
        "fictional_parity_holdout_sha256":hashlib.sha256(repr(PARITY).encode()).hexdigest(),
        "original_frozen_parity_holdout_loss":frozen_holdout,
        "trials":results,
        "only_fictional_data":True,
        "different_prompt_task_family":True,
        "independently_curated_benchmark":False,
        "heldout_used_for_training_or_lr_selection":False,
        "extra_training_after_holdout":False,
        "pretrained_weights_modified":False,
        "model_observer_did_not_control_optimizer":True,
        "source_authenticated":False,
        "may_promote":False,
        "production_ready":False,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only",action="store_true",required=True)
    p.add_argument("--ack-sandbox-research",action="store_true",required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--original-dev",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.snapshot,a.calibration,a.original_dev),sort_keys=True))
