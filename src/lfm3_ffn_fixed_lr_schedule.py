"""Precommitted actual-LFM3 FFN optimizer learning-rate schedule comparison.

Fixed vs step-1 LR switch (no model-generated choice): 3 frozen seeds,
native AdamW and RMS-matched Muon, 3 real LoRA gradient steps per arm.
Fictional training/development only; no heldout, promotion or checkpoints.
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

SCHEDULES=(
    ("adamw", (0.004,0.004,0.004)),
    ("adamw", (0.002,0.004,0.004)),
    ("muon_match_rms_adamw", (0.004,0.004,0.004)),
    ("muon_match_rms_adamw", (0.002,0.004,0.004)),
)
MODEL_PIN="135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"

def fixed_schedule_contract(schedules=SCHEDULES,seeds=RESEARCH_SEEDS):
    assert_fixed_lab_contract()
    if (type(schedules) is not tuple or schedules!=SCHEDULES
            or type(seeds) is not tuple or seeds!=RESEARCH_SEEDS
            or len(schedules)!=4 or any(len(schedule)!=3 for _,schedule in schedules)):
        raise ResearchRefused("optimizer learning-rate mutation outside precommitted research matrix")
    return True

def run(snapshot:Path):
    fixed_schedule_contract()
    verify_restricted_environment()
    proof=verify_snapshot(snapshot)
    if proof["checkpoint_sha256"]!=MODEL_PIN:
        raise ResearchRefused("wrong pretrained model for scheduler trial")
    import torch
    import transformers
    from torch.nn import functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if transformers.__version__!="5.21.0" or not hasattr(torch.optim,"Muon"):
        raise ResearchRefused("unsupported research optimizer runtime")
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(str(snapshot),trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(
        str(snapshot),trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16)
    if model.__class__.__name__!="Lfm3ForCausalLM":
        raise ResearchRefused("wrong model class")
    model.eval();model.requires_grad_(False)
    backbone=tuple(model.parameters())
    before=tuple(p._version for p in backbone)
    target=model.get_submodule(TARGET)
    if not isinstance(target,torch.nn.Linear) or tuple(target.weight.shape)!=(2048,7680):
        raise ResearchRefused("unexpected FFN adapter target")
    targets=target_tokens(tok)
    def prep(rows):
        out=[]
        for prompt,letter in rows:
            tokens=tok.apply_chat_template(
                [{"role":"user","content":prompt}],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,
                return_dict=True,return_tensors="pt")
            if tokens["input_ids"].shape[1]>80:
                raise ResearchRefused("unreviewed fictional prompt context")
            out.append((tokens,targets[letter]))
        return out
    train,dev=prep(TRAIN),prep(DEV)
    def objective(items):
        losses=[]
        for x,y in items:
            logits=model(**x,use_cache=False).logits[0,-1].float()
            losses.append(F.cross_entropy(logits.unsqueeze(0),torch.tensor([y])))
        return torch.stack(losses).mean()
    records=[]
    with torch.no_grad():
        frozen_dev=float(objective(dev))
    for seed in RESEARCH_SEEDS:
        for kind,schedule in SCHEDULES:
            A,B,handle=install_adapter(target,seed=seed)
            params=list(A.parameters())+list(B.parameters())
            if kind=="adamw":
                optimizer=torch.optim.AdamW(params,lr=schedule[0],weight_decay=0.)
            else:
                optimizer=torch.optim.Muon(
                    params,lr=schedule[0],momentum=0.95,ns_steps=5,
                    adjust_lr_fn="match_rms_adamw",weight_decay=0.)
            observations=[]
            try:
                with torch.no_grad():
                    initial_dev=float(objective(dev))
                if abs(initial_dev-frozen_dev)>0.0001:
                    raise ResearchRefused("optimizer arms did not share frozen base")
                for i,rate in enumerate(schedule):
                    # This is a precommitted RESEARCH LR schedule, not a model
                    # recommendation or online response to a heldout metric.
                    if len(optimizer.param_groups)!=1:
                        raise ResearchRefused("unexpected optimizer parameter groups")
                    optimizer.param_groups[0]["lr"]=rate
                    optimizer.zero_grad(set_to_none=True)
                    current=objective(train)
                    if not bool(torch.isfinite(current)):
                        raise ResearchRefused("nonfinite real-model objective")
                    current.backward()
                    if (B.weight.grad is None or not bool(torch.isfinite(B.weight.grad).all())
                            or float(B.weight.grad.abs().sum())<=0):
                        raise ResearchRefused("missing real FFN gradient")
                    norm=torch.nn.utils.clip_grad_norm_(params,1.)
                    if not bool(torch.isfinite(norm)):
                        raise ResearchRefused("nonfinite native gradient")
                    optimizer.step()
                    with torch.no_grad():
                        dev_loss=float(objective(dev))
                    if not math.isfinite(dev_loss):
                        raise ResearchRefused("nonfinite development telemetry")
                    observations.append({
                        "step":i+1,"learning_rate":rate,"dev_loss_after":dev_loss,
                        "training_objective_before_step":float(current.detach()),
                        "gradient_norm_before_clipping":float(norm),
                        "optimizer_changed_by_model":False,
                    })
                records.append({
                    "seed":seed,"optimizer":kind,"precommitted_learning_rates":list(schedule),
                    "trainable_parameters":sum(v.numel() for v in params),
                    "initial_dev_loss":initial_dev,"observations":observations,
                    "final_dev_loss":observations[-1]["dev_loss_after"],
                    "backbone_modified":False,"adapter_saved":False,
                })
            finally:
                handle.remove()
    if (tuple(p._version for p in backbone)!=before
            or any(p.requires_grad or p.grad is not None for p in backbone)):
        raise ResearchRefused("pretrained LFM3 changed during schedule study")
    return {
        "schema":"sg-lfm3-real-final-ffn-native-lr-schedule-v1",
        "checkpoint_sha256":proof["checkpoint_sha256"],
        "model_revision":proof["revision"],
        "frozen_backbone_parameters":sum(p.numel() for p in backbone),
        "target":TARGET,"rank":RANK,"trainable_parameters":38_912,
        "seeds":list(RESEARCH_SEEDS),
        "allowed_schedules":[{"optimizer":kind,"learning_rates":list(s)}
                             for kind,s in SCHEDULES],
        "real_model_training_trials":len(records),
        "frozen_dev_loss":frozen_dev,
        "train_sha256":hashlib.sha256(repr(TRAIN).encode()).hexdigest(),
        "dev_sha256":hashlib.sha256(repr(DEV).encode()).hexdigest(),
        "trials":records,
        "heldout_evaluated":False,"optimizer_advice_used":False,
        "learning_rate_schedule_precommitted":True,
        "model_weights_saved":False,
        "new_training_authorized":False,
        "automatic_optimizer_switching_authorized":False,
        "production_ready":False,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only",action="store_true",required=True)
    p.add_argument("--ack-sandbox-research",action="store_true",required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.snapshot),sort_keys=True))
