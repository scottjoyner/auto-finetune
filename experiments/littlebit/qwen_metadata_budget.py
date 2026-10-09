"""Metadata-only Qwen3-0.6B tensor-shape and LittleBit storage feasibility.

Reads exactly one small pinned config.json. Does not load model weights,
download anything, instantiate a neural network, import torch or train.
Estimates *hypothetical* upstream-format state_dict tensor bytes; never
presents the result as observed saved-checkpoint size.

Usage:
  python -m experiments.littlebit.qwen_metadata_budget \
    --config /path/to/config.json --inspect-pinned-metadata
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import ceil
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "167b8104f88905a951069f5f95f9776908da5f68"
CONFIG_SHA256 = "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
TARGETS = (0.55, 0.30)


def read_pinned_config(path: Path) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise ValueError("config digest changed: refuse unpinned model metadata")
    obj = json.loads(raw)
    if obj.get("architectures") != ["Qwen3ForCausalLM"] or obj.get("model_type") != "qwen3":
        raise ValueError("unexpected model architecture")
    if obj.get("tie_word_embeddings") is not True:
        raise ValueError("input/output embeddings not tied: accounting invalid")
    for field in ("hidden_size", "intermediate_size", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "num_hidden_layers",
                  "vocab_size"):
        value = obj.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError("invalid config dimension: " + field)
    return obj


def qwen3_linear_shapes(config: dict) -> dict[str, tuple[int, int]]:
    h = config["hidden_size"]
    m = config["intermediate_size"]
    q = config["num_attention_heads"] * config["head_dim"]
    kv = config["num_key_value_heads"] * config["head_dim"]
    return {
        "self_attn.q_proj": (q, h),
        "self_attn.k_proj": (kv, h),
        "self_attn.v_proj": (kv, h),
        "self_attn.o_proj": (h, q),
        "mlp.gate_proj": (m, h),
        "mlp.up_proj": (m, h),
        "mlp.down_proj": (h, m),
    }


def projected_upstream_tensor_bytes(out_features: int, in_features: int,
                                    rank: int, *, branches: int = 1,
                                    scale_bytes: int = 4) -> int:
    """Calculated tensor payload: int32 padded signs, four scales and metadata.

    Both factor sign matrices are separately 32-bit row-padded. The two
    factor-shape tensors cost 16 bytes each per branch. Three upstream
    scalar metadata buffers cost 16 bytes per layer, shared by branches.
    This is NOT torch.save ZIP size and assumes bias-free linear layers.
    """
    if min(out_features, in_features, rank, branches, scale_bytes) < 1:
        raise ValueError("positive dimensions, branches and dtype required")
    if rank > min(out_features, in_features):
        raise ValueError("rank exceeds a matrix dimension")
    if branches not in (1, 2) or scale_bytes not in (2, 4):
        raise ValueError("unsupported branch or scale format")
    # U: [out, rank]; V: [rank, in].
    U = out_features * ceil(rank / 32) * 4
    V = rank * ceil(in_features / 32) * 4
    scales = (out_features + in_features + 2 * rank) * scale_bytes
    shapes = 2 * 2 * 8  # U_shape and V_shape: two int64 each
    buffers = 4 + 8 + 4  # _eff_bit_actual, _split_dim_final, _eff_bit_target
    return branches * (U + V + scales + shapes) + buffers


def pick_rank(out_features: int, in_features: int, *,
              target_bpw: float, scale_bytes: int = 4) -> dict:
    if not 0 < target_bpw <= 1:
        raise ValueError("sub-1-bit target out of range")
    candidates = []
    for rank in range(8, min(out_features, in_features) + 1, 8):
        num_bytes = projected_upstream_tensor_bytes(
            out_features, in_features, rank, scale_bytes=scale_bytes)
        bpw = (8 * num_bytes) / (out_features * in_features)
        if bpw <= target_bpw:
            candidates.append((rank, num_bytes, bpw))
    if not candidates:
        return {"status": "INFEASIBLE", "rank": None,
                "projected_tensor_bytes": None, "projected_bpw": None}
    rank, num_bytes, bpw = candidates[-1]
    return {"status": "FEASIBLE", "rank": rank,
            "projected_tensor_bytes": num_bytes, "projected_bpw": bpw}


def inspect_pinned_metadata(config: dict) -> dict:
    shapes = qwen3_linear_shapes(config)
    layers = config["num_hidden_layers"]
    original_linear_parameters = layers * sum(
        out_dim * in_dim for out_dim, in_dim in shapes.values())
    embedding_parameters = config["vocab_size"] * config["hidden_size"]
    total_selected_parameters = original_linear_parameters + embedding_parameters
    scenarios = []
    for target in TARGETS:
        rows = []
        all_valid = True
        one_layer_payload = 0
        for module, (out_dim, in_dim) in shapes.items():
            rank = pick_rank(out_dim, in_dim, target_bpw=target)
            all_valid &= rank["status"] == "FEASIBLE"
            if rank["projected_tensor_bytes"] is not None:
                one_layer_payload += rank["projected_tensor_bytes"]
            rows.append({"module": module, "weight_shape": [out_dim, in_dim],
                         "original_parameters": out_dim * in_dim, **rank})
        total_linear_bytes = one_layer_payload * layers if all_valid else None
        # Qwen3 config says tie_word_embeddings true: one stored embedding
        # matrix, not two. BF16 assumption is provisional and must be tested.
        embedding_bytes = 2 * embedding_parameters
        aggregate_bytes = (total_linear_bytes + embedding_bytes
                           if total_linear_bytes is not None else None)
        scenarios.append({
            "target_block_bpw": target, "modules": rows,
            "status": "FEASIBLE" if all_valid else "INFEASIBLE",
            "projected_linear_tensor_bytes": total_linear_bytes,
            "projected_block_bpw":
                8 * total_linear_bytes / original_linear_parameters
                if total_linear_bytes is not None else None,
            "bf16_tied_embedding_bytes": embedding_bytes,
            "projected_total_tensor_bytes_excluding_other_layers":
                aggregate_bytes,
            "projected_model_bpw_excluding_other_layers":
                8 * aggregate_bytes / total_selected_parameters
                if aggregate_bytes is not None else None,
        })
    return {
        "status": "METADATA_ONLY_ESTIMATE",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "config_sha256": CONFIG_SHA256, "layer_count": layers,
        "original_linear_parameters": original_linear_parameters,
        "tied_embedding_parameters": embedding_parameters,
        "selected_parameter_denominator": total_selected_parameters,
        "scale_bytes_assumed": 4, "embedding_bytes_assumed": 2,
        "residual_branches": 0,
        "basis": "upstream int32 row-padded sign factors, FP32 four-scale "
                 "branch, two int64 shapes per packed tensor, three buffers",
        "excluded": ["layer norms", "QK norms", "biases", "checkpoint ZIP header",
                     "tokenizer", "config and other package files",
                     "implementation-specific model module selection"],
        "no_pretrained_weights_loaded": True, "training": False,
        "provider_calls": 0, "gpu_seconds": 0, "nas_writes": False,
        "scenarios": scenarios,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--inspect-pinned-metadata", action="store_true")
    args = p.parse_args(argv)
    if not args.inspect_pinned_metadata:
        p.error("metadata inspection requires explicit opt in")
    print(json.dumps(inspect_pinned_metadata(read_pinned_config(args.config)),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
