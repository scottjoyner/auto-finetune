"""Actual frozen LFM3 final-FFN LoRA optimizer LR calibration, fictional dev only.

Three fixed seeds × five frozen optimizer/LR arms × three gradient steps.
No holdout reuse, observer-induced training, source traces, checkpoints or
production changes. A future external evaluator may compare the evidence;
model suggestions cannot select learning rates for this run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_frozen_readout_lab import TRAIN, DEV, target_tokens, assert_fixed_lab_contract
from src.lfm3_last_layer_lora_lab import TARGET, RANK, RESEARCH_SEEDS, install_adapter
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

ARMS = (
    ("adamw", 0.001),
    ("adamw", 0.002),
    ("adamw", 0.004),
    ("muon_match_rms_adamw", 0.002),
    ("muon_match_rms_adamw", 0.004),
)
STEPS=3

def check_calibration_contract(arms=ARMS, seeds=RESEARCH_SEEDS, steps=STEPS):
    if (type(arms) is not tuple or arms!=ARMS
            or type(seeds) is not tuple or seeds!=RESEARCH_SEEDS
            or type(steps) is not int or steps!=STEPS):
        raise ResearchRefused("optimizer calibration is limited to 3x5x3 trial matrix")
    assert_fixed_lab_contract()
    return True

def run(snapshot: Path) -> dict:
    check_calibration_contract()
    verify_restricted_environment()
    proof=verify_snapshot(snapshot)
    import torch
    import transformers
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__!="5.21.0" or not hasattr(torch.optim,"Muon"):
        raise ResearchRefused("unreviewed native Torch Muon or Transformers environment")
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=True)
    model=AutoModelForCausalLM.from_pretrained(
        str(snapshot),local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16)
    if model.__class__.__name__!="Lfm3ForCausalLM":
        raise ResearchRefused("wrong frozen LFM3 model architecture")
    model.eval()
    model.requires_grad_(False)
    base=tuple(model.parameters())
    versions=tuple(p._version for p in base)
    target=model.get_submodule(TARGET)
    if not isinstance(target,torch.nn.Linear) or tuple(target.weight.shape)!=(2048,7680):
        raise ResearchRefused("unexpected frozen FFN target shape")
    ids=target_tokens(tok)

    def encode(rows):
        encoded=[]
        for prompt,letter in rows:
            data=tok.apply_chat_template(
                [{"role":"user","content":prompt}],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,
                return_dict=True,return_tensors="pt")
            if data["input_ids"].shape[1]>80:
                raise ResearchRefused("fictional prompt exceeds approved context budget")
            encoded.append((data,ids[letter]))
        return encoded

    train=encode(TRAIN)
    dev=encode(DEV)
    def loss(rows):
        results=[]
        for tokens,answer in rows:
            logits=model(**tokens,use_cache=False).logits[0,-1].float()
            results.append(F.cross_entropy(logits.unsqueeze(0),torch.tensor([answer])))
        return torch.stack(results).mean()

    trials=[]
    for seed in RESEARCH_SEEDS:
        for name,lr in ARMS:
            A,B,handle=install_adapter(target,seed=seed)
            params=list(A.parameters())+list(B.parameters())
            opt=(torch.optim.AdamW(params,lr=lr,weight_decay=0.0)
                 if name=="adamw" else torch.optim.Muon(
                     params,lr=lr,momentum=0.95,ns_steps=5,
                     adjust_lr_fn="match_rms_adamw",weight_decay=0.0))
            try:
                with torch.no_grad():
                    initial_train=float(loss(train))
                    initial_dev=float(loss(dev))
                max_norm=0.0
                for step in range(STEPS):
                    opt.zero_grad(set_to_none=True)
                    objective=loss(train)
                    if not bool(torch.isfinite(objective)):
                        raise ResearchRefused("nonfinite calibration loss")
                    objective.backward()
                    if (B.weight.grad is None or not bool(torch.isfinite(B.weight.grad).all())
                            or float(B.weight.grad.detach().abs().sum())<=0):
                        raise ResearchRefused("missing genuine output-adapter gradient")
                    norm=torch.nn.utils.clip_grad_norm_(params,1.0)
                    if not bool(torch.isfinite(norm)):
                        raise ResearchRefused("invalid full-model optimizer gradients")
                    max_norm=max(max_norm,float(norm))
                    opt.step()
                    if not all(bool(torch.isfinite(v).all()) for v in params):
                        raise ResearchRefused("nonfinite native optimizer adapter update")
                with torch.no_grad():
                    after_train=float(loss(train))
                    after_dev=float(loss(dev))
                if not all(math.isfinite(v) and v>=0 for v in
                           (initial_train,initial_dev,after_train,after_dev,max_norm)):
                    raise ResearchRefused("nonfinite measured calibration evidence")
                trials.append({
                    "seed":seed,"optimizer":name,"learning_rate":lr,
                    "steps":STEPS,"target":TARGET,"rank":RANK,
                    "trainable_adapter_parameters":sum(x.numel() for x in params),
                    "train_loss_before":initial_train,"dev_loss_before":initial_dev,
                    "train_loss_after":after_train,"dev_loss_after":after_dev,
                    "max_gradient_norm_before_clipping":max_norm,
                    "full_model_gradient_verified":True,
                    "pretrained_backbone_modified":False,
                    "adapter_checkpoint_written":False,
                })
            finally:handle.remove()
    if (tuple(p._version for p in base)!=versions
            or any(p.requires_grad or p.grad is not None for p in base)):
        raise ResearchRefused("frozen original LFM3 was mutated")
    return {
        "schema":"sg-lfm3-real-ffn-optimizer-lr-calibration-3seed.v1",
        "checkpoint_sha256":proof["checkpoint_sha256"],
        "model_revision":proof["revision"],
        "target_module":TARGET,
        "pretrained_parameters_frozen":sum(p.numel() for p in base),
        "trainable_adapter_parameters":38_912,
        "allowed_arms":[{"optimizer":name,"learning_rate":lr} for name,lr in ARMS],
        "seeds":list(RESEARCH_SEEDS),"steps_per_arm":STEPS,
        "train_sha256":hashlib.sha256(repr(TRAIN).encode()).hexdigest(),
        "dev_sha256":hashlib.sha256(repr(DEV).encode()).hexdigest(),
        "only_two_fabricated_train_and_two_dev":True,
        "holdout_evaluated":False,
        "source_authenticated":False,
        "model_advice_used_for_training":False,
        "trials":trials,
        "base_model_weights_modified":False,
        "adapter_saved":False,
        "optimizer_reconfigured_live":False,
        "may_auto_execute":False,
        "may_promote_fleet":False,
        "production_ready":False,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only",action="store_true",required=True)
    p.add_argument("--ack-sandbox-research",action="store_true",required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(run(args.snapshot),sort_keys=True))
