"""Experimental LFM3-1.7B Transformer gate-logit hook.

Official Lfm3Attention.forward uses sigmoid(self.gate_proj(hidden_states)).
The hook adds small bounded per-head logits, not weights.
Reversible context manager. No production use; no autonomous memory.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import math

MODEL_ID="LiquidAI/LFM3-1.7B"
CONFIG_SHA256="8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d"
MODEL_CODE_SHA256="e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab"
ATTENTION_LAYERS=(
    (2,"full_attention"),(4,"sliding_attention"),(6,"full_attention"),
    (8,"sliding_attention"),(10,"full_attention"),(12,"sliding_attention"),
    (14,"full_attention"),(16,"sliding_attention"),(18,"full_attention"),
    (20,"sliding_attention"),(22,"full_attention")
)
MAX_ABS_LOGIT_DELTA=0.25
TOP={"schema","model_id","model_config_sha256","source_kind","source_vector_sha256",
     "research_only","model_weights_modified","attention_backend_modified","layers"}
LAYER_FIELDS={"layer_index","layer_type","head_logit_deltas"}

class GateHookRefused(ValueError):
    pass

def strict_json(raw: bytes):
    if len(raw)>16_384:raise GateHookRefused("plan exceeds size cap")
    def pairs(entries):
        out={}
        for key,value in entries:
            if key in out:raise GateHookRefused("duplicate gate plan field")
            out[key]=value
        return out
    try:return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(GateHookRefused("nonfinite JSON")))
    except (ValueError,UnicodeError) as exc:
        raise GateHookRefused("invalid gate plan JSON") from exc

def validate(plan):
    if type(plan) is not dict or set(plan)!=TOP:
        raise GateHookRefused("plan schema keys mismatch")
    if (plan["schema"]!="sg-lfm3-gate-plan.v1" or plan["model_id"]!=MODEL_ID
        or plan["model_config_sha256"]!=CONFIG_SHA256
        or plan["source_kind"]!="synthetic-lab-v1"
        or plan["research_only"] is not True
        or plan["model_weights_modified"] is not False
        or plan["attention_backend_modified"] is not False
        or type(plan["source_vector_sha256"]) is not str
        or len(plan["source_vector_sha256"])!=64
        or any(c not in "0123456789abcdef" for c in plan["source_vector_sha256"])):
        raise GateHookRefused("plan requires fixed model and synthetic-only provenance")
    layers=plan["layers"]
    if type(layers) is not list or len(layers)!=len(ATTENTION_LAYERS):
        raise GateHookRefused("gate plan must cover exactly 11 attention layers")
    for row,(index,kind) in zip(layers,ATTENTION_LAYERS):
        if (type(row) is not dict or set(row)!=LAYER_FIELDS
            or type(row["layer_index"]) is not int or row["layer_index"]!=index
            or row["layer_type"]!=kind
            or type(row["head_logit_deltas"]) is not list
            or len(row["head_logit_deltas"])!=16):
            raise GateHookRefused("attention layer mapping mismatch")
        for value in row["head_logit_deltas"]:
            if (type(value) not in (float,int) or not math.isfinite(value)
                or abs(value)>MAX_ABS_LOGIT_DELTA+1e-7):
                raise GateHookRefused("out-of-bounds gate logit delta")
    return plan

def validate_config(model,config_bytes):
    if type(config_bytes) is not bytes or hashlib.sha256(config_bytes).hexdigest()!=CONFIG_SHA256:
        raise GateHookRefused("LFM3 model config SHA256 not pinned")
    cfg=model.config
    expected=(("model_type","lfm3"),("num_hidden_layers",24),
              ("num_attention_heads",16),("num_key_value_heads",4),
              ("head_dim",128),("hidden_size",2048),("sliding_window",256))
    if any(getattr(cfg,key,None)!=value for key,value in expected):
        raise GateHookRefused("model architecture is not pinned LFM3-1.7B")
    parsed=strict_json(config_bytes)
    layers=parsed.get("layer_types")
    if (type(layers) is not list or len(layers)!=24
        or list(getattr(cfg,"layer_types",[]))!=layers
        or [(i,layers[i]) for i in range(24) if layers[i]!="conv"]!=list(ATTENTION_LAYERS)):
        raise GateHookRefused("attention blocks do not match LFM3 checkpoint config")
    return True

@contextmanager
def synthetic_head_gate_hooks(model,plan,config_bytes,modeling_bytes,*,enabled=False):
    if enabled is not True:
        raise GateHookRefused("real inference hook requires explicit synthetic experiment enable")
    validate(plan)
    validate_config(model,config_bytes)
    if type(modeling_bytes) is not bytes or hashlib.sha256(modeling_bytes).hexdigest()!=MODEL_CODE_SHA256:
        raise GateHookRefused("LFM3 modeling implementation source is not pinned")
    try:import torch
    except ImportError as exc:raise GateHookRefused("PyTorch required for research hook") from exc
    layer_modules=model.model.layers
    if len(layer_modules)!=24:
        raise GateHookRefused("decoder layer count mismatched")
    hooks=[]
    try:
        for row in plan["layers"]:
            index=row["layer_index"]
            block=layer_modules[index]
            if not getattr(block,"is_attention_layer",False):
                raise GateHookRefused("expected LFM3 attention block is not attention")
            linear=block.self_attn.gate_proj
            if (not isinstance(linear,torch.nn.Linear)
                or linear.in_features!=2048 or linear.out_features!=16
                or bool(linear._forward_hooks)):
                raise GateHookRefused("gate projection unsupported or already hooked")
            deltas=row["head_logit_deltas"]
            def hook(_module,_input,output,values=deltas):
                if output.shape[-1]!=16 or output.ndim!=3:
                    raise GateHookRefused("unexpected gate_proj output shape")
                bias=torch.tensor(values,device=output.device,dtype=output.dtype)
                return output + bias.view(1,1,16)
            hooks.append(linear.register_forward_hook(hook))
        yield {"hooked_attention_layers":len(hooks),
               "research_only":True,"weights_modified":False}
    finally:
        for handle in reversed(hooks):handle.remove()
