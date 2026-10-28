"""Real frozen LFM3 FFN-LoRA optimizer-state replay, CPU and memory only.

For each native AdamW/Muon arm, compare uninterrupted 3-step training against
1 step + authenticated-in-memory checkpoint + reinstantiated LoRA/optimizer
+ restored 2 steps. No checkpoint saved to disk; not a crash-recovery claim.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_frozen_readout_lab import TRAIN, DEV, assert_fixed_lab_contract, target_tokens
from src.lfm3_last_layer_lora_lab import (
    TARGET, RESEARCH_SEEDS, RANK, LEARNING_RATE, OPTIMIZERS, install_adapter)
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

SEED = RESEARCH_SEEDS[0]
TOTAL_STEPS = 3
RESUME_AFTER_STEP = 1


def strict_budget(seed=SEED, total=TOTAL_STEPS, resume=RESUME_AFTER_STEP):
    if (type(seed) is not int or seed != SEED
            or type(total) is not int or total != TOTAL_STEPS
            or type(resume) is not int or resume != RESUME_AFTER_STEP):
        raise ResearchRefused("unreviewed frozen LFM3 optimizer replay budget")
    return True


def run(snapshot: Path) -> dict:
    strict_budget()
    verify_restricted_environment()
    assert_fixed_lab_contract()
    proof = verify_snapshot(snapshot)
    import torch
    import transformers
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("unreviewed transformers CPU runtime")
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16)
    if model.__class__.__name__ != "Lfm3ForCausalLM":
        raise ResearchRefused("not the official LFM3 model")
    model.eval()
    model.requires_grad_(False)
    original = tuple(model.parameters())
    base_versions = tuple(p._version for p in original)
    layer = model.get_submodule(TARGET)
    if (not isinstance(layer, torch.nn.Linear)
            or tuple(layer.weight.shape) != (2048, 7680)
            or not hasattr(torch.optim, "Muon")):
        raise ResearchRefused("unreviewed native model or optimizer")
    ids = target_tokens(tok)

    def encode(rows):
        out = []
        for prompt, answer in rows:
            payload = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt")
            if payload["input_ids"].shape[1] > 80:
                raise ResearchRefused("unreviewed training context")
            out.append((payload, ids[answer]))
        return out

    train = encode(TRAIN)
    dev = encode(DEV)
    def loss(items):
        values = []
        for payload, answer in items:
            logits = model(**payload, use_cache=False).logits[0,-1].float()
            values.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([answer])))
        return torch.stack(values).mean()

    def optimizer_for(kind, tensors):
        if kind == "adamw":
            return torch.optim.AdamW(tensors, lr=LEARNING_RATE, weight_decay=0.0)
        if kind == "muon_match_rms_adamw":
            return torch.optim.Muon(
                tensors, lr=LEARNING_RATE, momentum=0.95, ns_steps=5,
                adjust_lr_fn="match_rms_adamw", weight_decay=0.0)
        raise ResearchRefused("optimizer outside fixed comparison")

    def execute(kind, interrupted):
        A,B,handle=install_adapter(layer,seed=SEED)
        tensors=list(A.parameters())+list(B.parameters())
        opt=optimizer_for(kind,tensors)
        step=0
        memory_checkpoint_sha=None
        memory_checkpoint_bytes=0

        def one_step():
            opt.zero_grad(set_to_none=True)
            value=loss(train)
            if not bool(torch.isfinite(value)):
                raise ResearchRefused("nonfinite LFM3 optimizer replay loss")
            value.backward()
            if B.weight.grad is None or not bool(torch.isfinite(B.weight.grad).all()):
                raise ResearchRefused("no valid real model FFN-LoRA output gradient")
            norm=torch.nn.utils.clip_grad_norm_(tensors,1.0)
            if not bool(torch.isfinite(norm)):
                raise ResearchRefused("nonfinite real FFN optimizer replay gradient")
            opt.step()
            if not all(bool(torch.isfinite(x).all()) for x in tensors):
                raise ResearchRefused("nonfinite trained adapter")

        try:
            one_step()
            step=1
            if interrupted:
                state={
                    "A":A.state_dict(),
                    "B":B.state_dict(),
                    "optimizer":opt.state_dict(),
                    "step":step,
                }
                memory=io.BytesIO()
                torch.save(state,memory)
                packet=memory.getvalue()
                if not 0 < len(packet) <= 64*1024*1024:
                    raise ResearchRefused("in-memory optimizer checkpoint out of bounds")
                memory_checkpoint_sha=hashlib.sha256(packet).hexdigest()
                memory_checkpoint_bytes=len(packet)
                # No filesystem checkpoint. Validate bytes before a strictly
                # weights-only deserialization, and recreate BOTH trainable
                # matrices and the native optimizer state.
                if hashlib.sha256(packet).hexdigest() != memory_checkpoint_sha:
                    raise ResearchRefused("in-memory optimizer checkpoint altered")
                state=torch.load(io.BytesIO(packet),map_location="cpu",weights_only=True)
                if type(state["step"]) is not int or state["step"] != 1:
                    raise ResearchRefused("checkpoint step identity mismatch")
                handle.remove()
                A,B,handle=install_adapter(layer,seed=SEED)
                with torch.no_grad():
                    A.load_state_dict(state["A"])
                    B.load_state_dict(state["B"])
                tensors=list(A.parameters())+list(B.parameters())
                opt=optimizer_for(kind,tensors)
                opt.load_state_dict(state["optimizer"])
            while step < TOTAL_STEPS:
                one_step()
                step += 1
            with torch.no_grad():
                dev_loss=float(loss(dev))
            weights=(A.weight.detach().clone(),B.weight.detach().clone())
            optimizer_state=opt.state_dict()
            if not math.isfinite(dev_loss):
                raise ResearchRefused("nonfinite restored real LFM3 development loss")
            return weights, dev_loss, optimizer_state, {
                "checkpoint_sha256":memory_checkpoint_sha,
                "checkpoint_in_memory_bytes":memory_checkpoint_bytes,
                "actual_gradient_steps":step,
            }
        finally:
            handle.remove()

    def state_exact(a,b):
        if type(a) is not type(b):
            return False
        if isinstance(a,torch.Tensor):
            return bool(torch.equal(a,b))
        if isinstance(a,dict):
            return a.keys()==b.keys() and all(state_exact(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):
            return len(a)==len(b) and all(state_exact(x,y) for x,y in zip(a,b))
        return a==b

    arms=[]
    for kind in OPTIMIZERS:
        uninterrupted=execute(kind,False)
        restored=execute(kind,True)
        same_adapter=all(torch.equal(x,y) for x,y in zip(
            uninterrupted[0],restored[0]))
        same_optimizer_state=state_exact(uninterrupted[2],restored[2])
        same_dev_loss=uninterrupted[1]==restored[1]
        if not all((same_adapter,same_optimizer_state,same_dev_loss)):
            raise ResearchRefused("real FFN optimizer replay diverged; no candidate allowed")
        arms.append({
            "optimizer":kind,
            "seed":SEED,
            "three_step_dev_loss":uninterrupted[1],
            "resumed_three_step_dev_loss":restored[1],
            "recreated_adapter_bit_equal":same_adapter,
            "recreated_optimizer_state_bit_equal":same_optimizer_state,
            "recreated_dev_loss_bit_equal":same_dev_loss,
            "memory_checkpoint_sha256":restored[3]["checkpoint_sha256"],
            "memory_checkpoint_bytes":restored[3]["checkpoint_in_memory_bytes"],
            "resume_after_step":RESUME_AFTER_STEP,
            "total_steps":TOTAL_STEPS,
            "durable_checkpoint_written":False,
        })
    if (tuple(p._version for p in original) != base_versions
            or any(p.grad is not None or p.requires_grad for p in original)):
        raise ResearchRefused("original pretrained LFM3 was modified")
    return {
        "schema":"sg-lfm3-real-ffn-native-optimizer-inmemory-replay.v1",
        "checkpoint_sha256":proof["checkpoint_sha256"],
        "model_revision":proof["revision"],
        "target_module":TARGET,
        "base_parameters_frozen":1_728_801_536,
        "rank":RANK,
        "trainable_parameters":38_912,
        "experiment_seed":SEED,
        "training_steps_each":TOTAL_STEPS,
        "inmemory_checkpoint_after_step":RESUME_AFTER_STEP,
        "arms":arms,
        "exact_replay_both_optimizers":True,
        "only_fabricated_synthetic_data":True,
        "disk_checkpoint_written":False,
        "power_loss_recovery_proven":False,
        "original_model_weights_modified":False,
        "may_change_model":False,
        "may_promote_fleet":False,
        "production_ready":False,
    }


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-only",action="store_true",required=True)
    parser.add_argument("--ack-sandbox-research",action="store_true",required=True)
    parser.add_argument("--snapshot",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.snapshot),sort_keys=True))
