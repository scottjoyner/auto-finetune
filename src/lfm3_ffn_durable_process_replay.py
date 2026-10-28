"""Scratch-only cross-PROCESS frozen-LFM3 FFN LoRA optimizer restart.

Three mutually exclusive processes: baseline(3 steps), save(1 step, fsync
adapter + native optimizer state), resume(verify, instantiate new 1.7B model,
load weights-only checkpoint, train remaining 2 steps). Only fictional
TRAIN/DEV examples; no NAS, Neo4j, model deployment or production data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path

from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_ffn_optimizer_replay import admit_checkpoint_bytes
from src.lfm3_frozen_readout_lab import TRAIN, DEV, assert_fixed_lab_contract, target_tokens
from src.lfm3_last_layer_lora_lab import TARGET, RANK, RESEARCH_SEEDS, OPTIMIZERS, LEARNING_RATE, install_adapter
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

SCHEMA = "sg-lfm3-ffn-native-optimizer-process-restart.v1"
SNAPSHOT_REV = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
MODEL_SHA = "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
SEED = RESEARCH_SEEDS[0]
SAMPLE_BUDGET = 3
RESUME_STEP = 1
STATE_NAME = "native-optimizer.pt"
MANIFEST_NAME = "manifest.json"


def scratch_directory(path: Path, *, must_exist=True) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ResearchRefused("research checkpoint needs an absolute pathlib.Path")
    if path.parent != Path("/tmp") or not path.name.startswith("lfm3-durable-"):
        raise ResearchRefused("checkpoint directory must be /tmp/lfm3-durable-*")
    if path.is_symlink():
        raise ResearchRefused("checkpoint directory is a symlink")
    if must_exist:
        st=path.lstat()
        if not path.is_dir() or (st.st_mode & 0o077):
            raise ResearchRefused("scratch directory must be private ordinary directory")
    return path


def strict_run_args(mode: str, optimizer: str, directory: Path):
    if mode not in ("baseline", "save", "resume") or type(mode) is not str:
        raise ResearchRefused("unsupported cross-process optimizer replay mode")
    if optimizer not in OPTIMIZERS or type(optimizer) is not str:
        raise ResearchRefused("unreviewed optimizer for cross-process replay")
    scratch_directory(directory)
    return True


def _read_regular(path: Path, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise ResearchRefused("checkpoint input symlink refused")
    st=path.lstat()
    import stat
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_size <= 0 or st.st_size > max_bytes:
        raise ResearchRefused("checkpoint input not one bounded regular file")
    flags=os.O_RDONLY | getattr(os,"O_NOFOLLOW",0)
    fd=os.open(path,flags)
    try:
        real=os.fstat(fd)
        if not stat.S_ISREG(real.st_mode) or real.st_nlink!=1 or real.st_ino!=st.st_ino or real.st_dev!=st.st_dev:
            raise ResearchRefused("checkpoint input identity changed")
        data=os.read(fd,max_bytes+1)
        if len(data)!=st.st_size:
            raise ResearchRefused("checkpoint changed during bounded read")
        return data
    finally:
        os.close(fd)


def _write_exclusive(directory: Path, name: str, data: bytes) -> None:
    if name not in (STATE_NAME, MANIFEST_NAME) or type(data) is not bytes:
        raise ResearchRefused("unreviewed checkpoint output")
    target=directory/name
    flags=os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os,"O_NOFOLLOW",0)
    fd=os.open(target,flags,0o600)
    try:
        with os.fdopen(fd,"wb",closefd=False) as output:
            output.write(data)
            output.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def save_durable(directory: Path, state: dict, *, optimizer: str) -> dict:
    """Crash-consistent *lab* write: exclusive payload, fsync, exclusive manifest, fsync dir.

    Digest is an accidental-tamper detector, not a signed external producer.
    If interrupted before manifest, resume rejects incomplete checkpoint.
    """
    import torch
    strict_run_args("save",optimizer,directory)
    if (type(state) is not dict or set(state)!={"schema","optimizer","seed",
            "step","model_sha256","model_revision","target","rank","A","B","optimizer_state"}
            or state["schema"]!=SCHEMA or state["optimizer"]!=optimizer
            or state["seed"]!=SEED or state["step"]!=RESUME_STEP
            or state["model_sha256"]!=MODEL_SHA
            or state["model_revision"]!=SNAPSHOT_REV
            or state["target"]!=TARGET or state["rank"]!=RANK):
        raise ResearchRefused("unapproved native optimizer checkpoint identity")
    packet=io.BytesIO()
    torch.save(state,packet)
    contents=packet.getvalue()
    if not 0<len(contents)<=16*1024*1024:
        raise ResearchRefused("native optimizer checkpoint exceeds 16MiB")
    digest=hashlib.sha256(contents).hexdigest()
    manifest={
        "schema":SCHEMA,"optimizer":optimizer,"model_sha256":MODEL_SHA,
        "model_revision":SNAPSHOT_REV,"target":TARGET,"rank":RANK,
        "seed":SEED,"checkpoint_step":RESUME_STEP,"bytes":len(contents),
        "sha256":digest,"contains_model_backbone":False,
        "review_only":True,"production_ready":False,
    }
    if (directory/STATE_NAME).exists() or (directory/MANIFEST_NAME).exists():
        raise ResearchRefused("scratch checkpoint must be written only once")
    _write_exclusive(directory,STATE_NAME,contents)
    _write_exclusive(directory,MANIFEST_NAME,
                     (json.dumps(manifest,sort_keys=True)+"\n").encode())
    fd=os.open(directory,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
    try:os.fsync(fd)
    finally:os.close(fd)
    return manifest


def load_durable(directory: Path, optimizer: str):
    import torch
    strict_run_args("resume",optimizer,directory)
    manifest_bytes=_read_regular(directory/MANIFEST_NAME,4_096)
    try: manifest=json.loads(manifest_bytes)
    except (ValueError, UnicodeDecodeError) as e:
        raise ResearchRefused("invalid bounded checkpoint manifest") from e
    if (type(manifest) is not dict or set(manifest)!={
        "schema","optimizer","model_sha256","model_revision","target","rank","seed",
        "checkpoint_step","bytes","sha256","contains_model_backbone",
        "review_only","production_ready"}
        or manifest["schema"]!=SCHEMA or manifest["optimizer"]!=optimizer
        or manifest["model_sha256"]!=MODEL_SHA or manifest["model_revision"]!=SNAPSHOT_REV
        or manifest["target"]!=TARGET or manifest["rank"]!=RANK or manifest["seed"]!=SEED
        or type(manifest["checkpoint_step"]) is not int
        or manifest["checkpoint_step"]!=RESUME_STEP
        or type(manifest["bytes"]) is not int
        or not 0<manifest["bytes"]<=16*1024*1024
        or manifest["contains_model_backbone"] is not False
        or manifest["review_only"] is not True or manifest["production_ready"] is not False):
        raise ResearchRefused("untrusted or incompatible optimizer restart manifest")
    packet=_read_regular(directory/STATE_NAME,16*1024*1024)
    if len(packet)!=manifest["bytes"]:
        raise ResearchRefused("optimizer payload/manifest size mismatch")
    admit_checkpoint_bytes(packet,manifest["sha256"])
    try:
        state=torch.load(io.BytesIO(packet),map_location="cpu",weights_only=True)
    except Exception as e:
        raise ResearchRefused("weights-only optimizer state parse failed") from e
    if (type(state) is not dict or set(state)!={"schema","optimizer","seed","step",
            "model_sha256","model_revision","target","rank","A","B","optimizer_state"}
            or state["schema"]!=SCHEMA or state["optimizer"]!=optimizer
            or state["model_sha256"]!=MODEL_SHA or state["model_revision"]!=SNAPSHOT_REV
            or state["seed"]!=SEED or state["step"]!=RESUME_STEP
            or state["target"]!=TARGET or state["rank"]!=RANK
            or not isinstance(state["A"], dict) or not isinstance(state["B"], dict)
            or not isinstance(state["optimizer_state"], dict)):
        raise ResearchRefused("optimizer checkpoint model/shape/step mismatch")
    A=state["A"].get("weight")
    B=state["B"].get("weight")
    if (not isinstance(A,torch.Tensor) or A.shape!=(4,7680)
            or not isinstance(B,torch.Tensor) or B.shape!=(2048,4)
            or A.dtype!=torch.float32 or B.dtype!=torch.float32
            or not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(B).all())):
        raise ResearchRefused("adapter weight matrix incompatible with frozen LFM3")
    return state,manifest


def canonical_state_sha(value) -> str:
    """Deterministic hash of nested native optimizer tensors and scalars."""
    import torch
    h=hashlib.sha256()
    def walk(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous()
            h.update(b"T"+str(t.dtype).encode()+repr(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b"D")
            for k in sorted(v,key=lambda x:(str(type(x)),str(x))):
                walk(k);walk(v[k])
        elif type(v) in (tuple,list):
            h.update(b"L"+str(len(v)).encode())
            for x in v:walk(x)
        elif type(v) in (str,int,float,bool,type(None)):
            h.update(b"S"+repr((type(v).__name__,v)).encode())
        else:
            raise ResearchRefused("unexpected native optimizer state value")
    walk(value)
    return h.hexdigest()


def real_process(snapshot: Path, directory: Path, *, mode: str, optimizer: str) -> dict:
    strict_run_args(mode,optimizer,directory)
    verify_restricted_environment()
    assert_fixed_lab_contract()
    proof=verify_snapshot(snapshot)
    if proof["checkpoint_sha256"]!=MODEL_SHA or proof["revision"]!=SNAPSHOT_REV:
        raise ResearchRefused("pinned pretrained backbone changed")
    import torch
    import transformers
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if transformers.__version__!="5.21.0" or not hasattr(torch.optim,"Muon"):
        raise ResearchRefused("unreviewed native optimizer runtime")
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(str(snapshot),trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(
        str(snapshot),trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16)
    if model.__class__.__name__!="Lfm3ForCausalLM":
        raise ResearchRefused("wrong pretrained LFM3 architecture")
    model.eval()
    model.requires_grad_(False)
    backbone=tuple(model.parameters())
    before_versions=tuple(p._version for p in backbone)
    target=model.get_submodule(TARGET)
    if not isinstance(target,torch.nn.Linear) or target.weight.shape!=(2048,7680):
        raise ResearchRefused("unsupported final FFN layer")
    labels=target_tokens(tok)
    def samples(rows):
        out=[]
        for prompt,answer in rows:
            payload=tok.apply_chat_template(
                [{"role":"user","content":prompt}],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,
                return_dict=True,return_tensors="pt")
            if payload["input_ids"].shape[1]>80:
                raise ResearchRefused("fictional example above input budget")
            out.append((payload,labels[answer]))
        return out
    train,dev=samples(TRAIN),samples(DEV)
    def loss(items):
        values=[]
        for payload,answer in items:
            logits=model(**payload,use_cache=False).logits[0,-1].float()
            values.append(F.cross_entropy(logits.unsqueeze(0),torch.tensor([answer])))
        return torch.stack(values).mean()
    A,B,handle=install_adapter(target,seed=SEED)
    params=list(A.parameters())+list(B.parameters())
    if optimizer=="adamw":
        opt=torch.optim.AdamW(params,lr=LEARNING_RATE,weight_decay=0)
    else:
        opt=torch.optim.Muon(params,lr=LEARNING_RATE,momentum=0.95,
                             ns_steps=5,adjust_lr_fn="match_rms_adamw",
                             weight_decay=0)
    try:
        step=0
        manifest=None
        if mode=="resume":
            checkpoint,manifest=load_durable(directory,optimizer)
            with torch.no_grad():
                A.load_state_dict(checkpoint["A"])
                B.load_state_dict(checkpoint["B"])
            opt.load_state_dict(checkpoint["optimizer_state"])
            step=RESUME_STEP
        goal=RESUME_STEP if mode=="save" else SAMPLE_BUDGET
        while step<goal:
            opt.zero_grad(set_to_none=True)
            l=loss(train)
            if not bool(torch.isfinite(l)):
                raise ResearchRefused("nonfinite frozen LFM3 train objective")
            l.backward()
            norm=torch.nn.utils.clip_grad_norm_(params,1.0)
            if not bool(torch.isfinite(norm)) or B.weight.grad is None:
                raise ResearchRefused("invalid full-model gradient")
            opt.step()
            if not all(bool(torch.isfinite(p).all()) for p in params):
                raise ResearchRefused("optimizer caused nonfinite FFN adapter")
            step+=1
        if mode=="save":
            manifest=save_durable(directory,{
                "schema":SCHEMA,"optimizer":optimizer,"seed":SEED,"step":step,
                "model_sha256":MODEL_SHA,"model_revision":SNAPSHOT_REV,
                "target":TARGET,"rank":RANK,
                "A":A.state_dict(),"B":B.state_dict(),
                "optimizer_state":opt.state_dict(),
            },optimizer=optimizer)
        with torch.no_grad():
            dev_loss=float(loss(dev))
        digest=canonical_state_sha({
            "A":A.state_dict(),"B":B.state_dict(),
            "optimizer":opt.state_dict()})
        if (not math.isfinite(dev_loss)
                or tuple(p._version for p in backbone)!=before_versions
                or any(p.grad is not None or p.requires_grad for p in backbone)):
            raise ResearchRefused("base model unexpectedly mutated")
        return {
            "schema":SCHEMA,"mode":mode,"optimizer":optimizer,
            "snapshot_sha256":MODEL_SHA,"seed":SEED,
            "steps_completed":step,
            "dev_loss":dev_loss,
            "adapter_optimizer_canonical_sha256":digest,
            "manifest_payload_sha256":manifest["sha256"] if manifest else None,
            "cross_process_replay_verified":False,
            "source_authenticated":False,
            "fictional_only":True,
            "original_backbone_modified":False,
            "network_disabled_required":True,
            "production_ready":False,
        }
    finally:
        handle.remove()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-only",action="store_true",required=True)
    parser.add_argument("--ack-sandbox-research",action="store_true",required=True)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--scratch",type=Path,required=True)
    parser.add_argument("--mode",choices=["baseline","save","resume"],required=True)
    parser.add_argument("--optimizer",choices=OPTIMIZERS,required=True)
    args=parser.parse_args()
    print(json.dumps(real_process(args.snapshot,args.scratch,mode=args.mode,
                                  optimizer=args.optimizer),sort_keys=True))
