"""No-training inventory of candidate Muon vs AdamW LFM3 parameter groups.

Never constructs optimizers or updates weights. Matrix shape alone is NOT
sufficient: 2-D embeddings, output heads, convolution projections, head-gate
projections, norms, and LoRA tensors stay on AdamW pending independent review.
"""
from __future__ import annotations
import hashlib
import json
import re
from src.lfm3_optimizer_guard import CHECKPOINT, OptimizerResearchRefused

MATRIX = re.compile(
    r"^model\.layers\.(\d+)\."
    r"(?:feed_forward\.w[123]|self_attn\.[qkvo]_proj)\.weight\Z"
)


def plan_full_model_metadata(named_parameters, *,
                             model_class: str,
                             checkpoint_sha256: str,
                             explicit_research_review: bool = False) -> dict:
    if (model_class != "Lfm3ForCausalLM"
            or checkpoint_sha256 != CHECKPOINT
            or explicit_research_review is not True):
        raise OptimizerResearchRefused("full-model Muon group plan not admitted")
    observed = {}
    aliases = set()
    eligible_names = []
    class_counts = {}
    class_tensors = {}
    total = 0
    for name, parameter in named_parameters:
        if (type(name) is not str or not 1 <= len(name) <= 180
                or name in observed or id(parameter) in aliases):
            raise OptimizerResearchRefused("invalid/repeated model tensor identity")
        aliases.add(id(parameter))
        shape = tuple(parameter.shape)
        if not 1 <= len(shape) <= 4 or any(type(s) is not int or s < 1 for s in shape):
            raise OptimizerResearchRefused("invalid tensor dimensions")
        elements = 1
        for size in shape:
            elements *= size
        if total + elements > 3_000_000_000:
            raise OptimizerResearchRefused("unbounded checkpoint parameter inventory")
        observed[name] = shape
        total += elements
        match = MATRIX.fullmatch(name)
        if match and len(shape) == 2 and min(shape) > 1:
            layer = int(match.group(1))
            if not 0 <= layer < 24:
                raise OptimizerResearchRefused("candidate layer outside LFM3")
            if ".feed_forward." in name:
                group = "research_muon_ffn_candidate"
            else:
                group = "research_muon_attention_candidate"
            eligible_names.append(name)
        elif "lora" in name.lower():
            group = "adamw_lora_adapter"
        elif ".conv." in name:
            group = "adamw_conv_and_conv_projection"
        elif ".gate_proj." in name:
            group = "adamw_head_gate_projection"
        elif "embed" in name or "lm_head" in name or "output" in name:
            group = "adamw_embedding_output"
        elif "norm" in name or name.endswith(".bias"):
            group = "adamw_norm_and_bias"
        else:
            group = "adamw_unreviewed"
        class_counts[group] = class_counts.get(group, 0) + elements
        class_tensors[group] = class_tensors.get(group, 0) + 1
    if not observed or total < 1_000_000:
        raise OptimizerResearchRefused("not the full official parameter inventory")
    if sum(class_counts.values()) != total:
        raise OptimizerResearchRefused("unaccounted model parameter group")
    if not eligible_names:
        raise OptimizerResearchRefused("no approved candidate matrix classes discovered")
    ordered = json.dumps(
        [[n, list(observed[n])] for n in sorted(observed)], separators=(",", ":")
    ).encode()
    return {
        "schema": "sg-lfm3-full-optimizer-partition-inventory.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "model_class": model_class,
        "named_tensor_shape_sha256": hashlib.sha256(ordered).hexdigest(),
        "total_parameters": total,
        "total_tensors": len(observed),
        "candidate_muon_parameters":
            class_counts.get("research_muon_ffn_candidate", 0)
            + class_counts.get("research_muon_attention_candidate", 0),
        "candidate_muon_tensor_count": len(eligible_names),
        "adamw_or_unreviewed_parameters": total
            - class_counts.get("research_muon_ffn_candidate", 0)
            - class_counts.get("research_muon_attention_candidate", 0),
        "category_parameters": dict(sorted(class_counts.items())),
        "category_tensors": dict(sorted(class_tensors.items())),
        "sample_candidate_names": sorted(eligible_names)[:8],
        "actual_optimizer_allocated": False,
        "model_parameters_modified": False,
        "adapter_trainable_inventory_verified": False,
        "muon_eligibility_reviewed_for_training": False,
        "safe_to_change_running_optimizer": False,
        "production_ready": False,
    }
