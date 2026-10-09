"""Offline Qwen3 meta-module mapping + DENY-only byte-accurate rank admission.

No pretrained weights, checkpoint writes, downloads, training, model serving,
production actions, or authorization to *change* rank selection.

Requires a separate read-only checkout of the exact pinned LittleBit upstream
source, CPU-only torch 2.6.0 and transformers 4.51.3, plus exact 726-byte
pinned public Qwen3 config.json. All model parameters remain on meta device.

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES='' \
  python -m experiments.littlebit.qwen_meta_admission \
    --config /path/config.json --upstream-root /path/LittleBit \
    --inspect-meta --target-bpw 0.55
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .qwen_metadata_budget import (
    read_pinned_config, pick_rank, projected_upstream_tensor_bytes,
)

UPSTREAM_COMMIT = "42d658b0c79f76450b34b6a3547462c7cdc3e1a0"
PINNED_BLOBS = {
    "quantization/modules/littlebit.py": "43ce9d2f9383b676c3898346ece2d3994269a908",
    "quantization/utils/binary_packer.py": "78f0e20f8a525d3cdb0f94908fd767077834cf4b",
    "quantization/utils/quant_util.py": "c40a35168ab8f02411dfcd484f0fb4ae11b026b9",
}
SUFFIXES = frozenset((
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
))
EXPECTED_LAYERS = 28
EXPECTED_CONVERTED = EXPECTED_LAYERS * len(SUFFIXES)


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data
    ).hexdigest()


def verify_source(root: Path) -> Path:
    root = root.resolve(strict=True)
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True, timeout=6,
    ).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError("official source commit mismatch")
    for relative, expected in PINNED_BLOBS.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("source escaped pinned directory")
        if blob_sha(path.read_bytes()) != expected:
            raise ValueError("source blob mismatch: " + relative)
    return root


def rank_admission(out_features: int, in_features: int, selected_rank: int,
                   target_bpw: float, *, branches: int = 1,
                   scale_bytes: int = 4) -> dict:
    """Read-only admission. Never alters upstream ranks or model modules.

    Unlike upstream _compute_eff_bits, includes physical 32-bit row-padded
    factor tensors, four FP32 scales, int64 shape tensors and three buffers.
    """
    if branches not in (1, 2) or scale_bytes not in (2, 4):
        raise ValueError("unsupported storage branch or scale dtype")
    if type(selected_rank) is not int or selected_rank < 8 or selected_rank % 8:
        raise ValueError("invalid selected rank")
    if selected_rank > min(out_features, in_features):
        raise ValueError("selected rank exceeds matrix dimension")
    if not 0 < target_bpw <= 1:
        raise ValueError("invalid sub-1-bit target")
    actual_bytes = projected_upstream_tensor_bytes(
        out_features, in_features, selected_rank,
        branches=branches, scale_bytes=scale_bytes)
    denominator = out_features * in_features
    physical_bpw = 8 * actual_bytes / denominator
    allowed = []
    for rank in range(8, min(out_features, in_features) + 1, 8):
        candidate_bytes = projected_upstream_tensor_bytes(
            out_features, in_features, rank,
            branches=branches, scale_bytes=scale_bytes)
        if 8 * candidate_bytes <= target_bpw * denominator:
            allowed.append((rank, candidate_bytes))
    proposed = allowed[-1] if allowed else None
    return {
        "status": "ADMIT" if physical_bpw <= target_bpw else "DENY",
        "target_bpw": target_bpw, "selected_rank": selected_rank,
        "physical_bpw": physical_bpw,
        "projected_bytes": actual_bytes,
        "maximum_eligible_rank": proposed[0] if proposed else None,
        "maximum_eligible_bytes": proposed[1] if proposed else None,
        "policy": "byte-accurate deny-only; no rank mutation",
    }


def inspect(config_path: Path, upstream_root: Path, target_bpw: float,
            *, residual: bool = False) -> dict:
    # Validate source and config before importing executable dependencies.
    config_dict = read_pinned_config(config_path)
    root = verify_source(upstream_root)
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES", None) != ""
    ):
        raise RuntimeError("requires strict HF offline and CUDA-disabled environment")
    if any(k.startswith("quantization") for k in sys.modules):
        raise RuntimeError("quantization package already imported: refuse path ambiguity")
    import torch
    import transformers
    from transformers import Qwen3Config, Qwen3ForCausalLM

    if torch.__version__ != "2.6.0+cpu" or transformers.__version__ != "4.51.3":
        raise RuntimeError("unsupported torch/transformers versions")
    if torch.cuda.is_available() or torch.version.cuda or torch.version.hip:
        raise RuntimeError("CPU-only torch build required")
    # The source path is pinned. Do not import the upstream training CLI.
    sys.path.insert(0, str(root))
    try:
        util = importlib.import_module("quantization.utils.quant_util")
        qwen_config = Qwen3Config(**config_dict)
        with torch.device("meta"):
            model = Qwen3ForCausalLM(qwen_config)
            before = {
                name: tuple(module.weight.shape)
                for name, module in model.named_modules()
                if type(module) is torch.nn.Linear
            }
            if len(before) != EXPECTED_CONVERTED + 1 or "lm_head" not in before:
                raise AssertionError("unexpected Qwen3 module census before conversion")
            if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
                raise AssertionError("Qwen tied embeddings are not actually shared")
            args = argparse.Namespace(
                quant_func="STEBinary", quant_mod="LittleBitLinear",
                residual=residual, split_dim=1024, eff_bit=target_bpw,
                min_split_dim=8, kv_factor=1.0, use_itq=True,
                itq_n_iter=50, model_id="Qwen/Qwen3-0.6B",
            )
            util.apply_littlebit_patch(model, args, do_train=False)
            converted = {
                name: module
                for name, module in model.named_modules()
                if module.__class__.__name__ == "LittleBitLinear"
            }
            if len(converted) != EXPECTED_CONVERTED:
                raise AssertionError("unexpected upstream converted module count")
            if set(before) - set(converted) != {"lm_head"}:
                raise AssertionError("wrong quantizer exclusions")
            if type(model.lm_head) is not torch.nn.Linear:
                raise AssertionError("lm_head was unexpectedly quantized")
            if not all(parameter.is_meta for parameter in model.parameters()):
                raise AssertionError("materialized model parameter detected")
            if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
                raise AssertionError("tied embeddings changed after patch")
            rows = []
            original_total = 0
            projected_total = 0
            violated = 0
            for name, mod in sorted(converted.items()):
                parts = name.split(".")
                if len(parts) != 5 or parts[0:2] != ["model", "layers"]:
                    raise AssertionError("unexpected converted module path: " + name)
                layer_index = int(parts[2])
                suffix = ".".join(parts[3:])
                if not 0 <= layer_index < EXPECTED_LAYERS or suffix not in SUFFIXES:
                    raise AssertionError("unexpected converted Qwen module " + name)
                selected_rank = int(mod.split_dim)
                if mod.U.shape != (mod.out_features, selected_rank):
                    raise AssertionError("bad U shape " + name)
                if mod.V.shape != (selected_rank, mod.in_features):
                    raise AssertionError("bad V shape " + name)
                decision = rank_admission(
                    mod.out_features, mod.in_features, selected_rank,
                    target_bpw, branches=2 if residual else 1,
                )
                if decision["status"] == "DENY":
                    violated += 1
                original_total += mod.out_features * mod.in_features
                projected_total += decision["projected_bytes"]
                rows.append({
                    "path": name, "layer_index": layer_index, "type": suffix,
                    "shape": [mod.out_features, mod.in_features], **decision,
                })
            if len({row["path"] for row in rows}) != EXPECTED_CONVERTED:
                raise AssertionError("duplicate module paths")
            return {
                "status": "HOLD" if violated else "ADMIT",
                "method": "official_pinned_meta_patch__no_model_weights",
                "upstream_git_sha": UPSTREAM_COMMIT,
                "qwen_revision": "167b8104f88905a951069f5f95f9776908da5f68",
                "torch": torch.__version__, "transformers": transformers.__version__,
                "target_bpw": target_bpw, "residual": residual,
                "candidate_modules": len(before), "converted_modules": len(converted),
                "excluded_modules": ["lm_head"], "violations": violated,
                "estimated_linear_tensor_bytes": projected_total,
                "estimated_converted_linear_bpw":
                    8 * projected_total / original_total,
                "all_model_parameters_meta": True, "model_downloads": 0,
                "training": False, "gpu_seconds": 0, "nas_writes": False,
                "provider_calls": 0, "no_automatic_rank_mutation": True,
                "modules": rows,
            }
    finally:
        # The pinned package may only be used within this one bounded call.
        # Remove modules imported by the audit so a second independent case
        # re-verifies exact source bytes instead of reusing cached objects.
        for module_name in tuple(sys.modules):
            if module_name == "quantization" or module_name.startswith("quantization."):
                del sys.modules[module_name]
        sys.path.remove(str(root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--inspect-meta", action="store_true")
    parser.add_argument("--target-bpw", type=float, choices=(0.55, 0.30),
                        default=0.55)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--require-admissible", action="store_true")
    args = parser.parse_args(argv)
    if not args.inspect_meta:
        parser.error("read-only meta inspection requires --inspect-meta")
    summary = inspect(args.config, args.upstream_root, args.target_bpw,
                      residual=args.residual)
    print(json.dumps(summary, sort_keys=True, indent=2))
    if args.require_admissible and summary["status"] != "ADMIT":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
