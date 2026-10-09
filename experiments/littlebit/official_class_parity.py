"""Reproducible *official upstream class* CPU-only LittleBit initializer parity.

NO model downloads, NAS writes, training, schedulers, or CUDA. Runs the exact
source pinned by Git object ID in an isolated import namespace. Licensed
upstream files are read from a separate checkout and are not vendored here.

Usage:
  python -m experiments.littlebit.official_class_parity \
      --upstream-root /path/to/pinned/LittleBit --execute-cpu

The default invocation is deliberately disarmed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

UPSTREAM_COMMIT = "42d658b0c79f76450b34b6a3547462c7cdc3e1a0"
BLOBS = {
    "quantization/modules/littlebit.py": "43ce9d2f9383b676c3898346ece2d3994269a908",
    "quantization/utils/binary_packer.py": "78f0e20f8a525d3cdb0f94908fd767077834cf4b",
}
SEEDS = (7, 42, 1337)
SHAPES = ((64, 64, 8), (128, 256, 16), (256, 128, 16))
REQUIRED_STATE = ("U_packed", "V_packed", "U_shape", "V_shape",
                  "u1", "u2", "v1", "v2", "_eff_bit_actual",
                  "_eff_bit_target", "_split_dim_final")


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def verify_source(root: Path) -> dict[str, bytes]:
    root = root.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError("upstream HEAD is not the exact pinned revision")
    found = {}
    for rel, expected in BLOBS.items():
        file = (root / rel).resolve(strict=True)
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError("unsafe source path")
        raw = file.read_bytes()
        if git_blob_sha(raw) != expected:
            raise ValueError("upstream source blob mismatch: " + rel)
        found[rel] = raw
    return found


def load_official_class(sources: dict[str, bytes]):
    """Execute pinned class + packer, not package init, hub or training stack."""
    q = types.ModuleType("quantization")
    q.__path__ = []
    utils = types.ModuleType("quantization.utils")
    utils.__path__ = []
    packer = types.ModuleType("quantization.utils.binary_packer")
    exec(compile(sources["quantization/utils/binary_packer.py"],
                 "<pinned binary_packer.py>", "exec"), packer.__dict__)
    namespace = {"__name__": "pinned_littlebit_class"}
    with patch.dict(sys.modules, {
        "quantization": q,
        "quantization.utils": utils,
        "quantization.utils.binary_packer": packer,
    }):
        exec(compile(sources["quantization/modules/littlebit.py"],
                     "<pinned littlebit.py>", "exec"), namespace)
    return namespace["LittleBitLinear"]


def make_weights(torch, rows: int, cols: int, seed: int):
    # Fixed synthetic inputs, never loaded from a pretrained checkpoint.
    gen = torch.Generator(device="cpu").manual_seed(seed + 1000)
    return torch.randn((rows, cols), generator=gen, dtype=torch.float32)


def execute_case(torch, base_cls, *, rows: int, cols: int, rank: int,
                 seed: int, use_itq: bool, residual: bool, itq_iters: int):
    class UpstreamLinear(base_cls, torch.nn.Linear):
        def __init__(self):
            torch.nn.Linear.__init__(self, cols, rows, bias=False)

    weight = make_weights(torch, rows, cols, seed)
    # Use an identical initial RNG state for corresponding SVD and ITQ arms.
    # ITQ's internal random rotation also changes later randomized svd_lowrank
    # draws: the observed difference is the *official recipe*, not a
    # mathematically isolated ITQ-only causal effect.
    torch.manual_seed(seed + 11)
    layer = UpstreamLinear()
    with torch.no_grad():
        layer.weight.copy_(weight)
        layer.__quant_convert__(
            do_train=True, quant_func=torch.sign, split_dim=rank,
            min_split_dim=rank, use_itq=use_itq, itq_n_iter=itq_iters,
            residual=residual,
        )
        state = layer.state_dict()

        required = REQUIRED_STATE + (
            ("U_R_packed", "V_R_packed", "U_R_shape", "V_R_shape",
             "u1_R", "u2_R", "v1_R", "v2_R") if residual else ()
        )
        missing = sorted(set(required) - set(state))
        if missing:
            raise AssertionError("official state missing keys: " + str(missing))
        scale_names = ("u1", "u2", "v1", "v2") + (
            ("u1_R", "u2_R", "v1_R", "v2_R") if residual else ()
        )
        scale_types = {key: str(state[key].dtype) for key in scale_names}
        def reconstruct(suffix=""):
            U = torch.sign(getattr(layer, "U" + suffix).float())
            V = torch.sign(getattr(layer, "V" + suffix).float())
            u1 = getattr(layer, "u1" + suffix).float()
            u2 = getattr(layer, "u2" + suffix).float()
            v1 = getattr(layer, "v1" + suffix).float()
            v2 = getattr(layer, "v2" + suffix).float()
            return (U * (u1.T @ u2)) @ (V * (v1.T @ v2))
        approx = reconstruct()
        if residual:
            approx = approx + reconstruct("_R")
        inputs = torch.randn((2, cols),
                             generator=torch.Generator().manual_seed(seed + 99))
        forward_delta = float((layer(inputs) - inputs @ approx.T).abs().max())
        reconstruction_error = float((weight - approx).norm() / weight.norm())
        memory = io.BytesIO()
        torch.save(state, memory)

    if forward_delta > 1e-3 or not 0 <= reconstruction_error < 10:
        raise AssertionError("official forward parity or norm failed")
    tensor_bytes = sum(v.numel() * v.element_size() for v in state.values())
    names = ("U_packed", "V_packed") + (
        ("U_R_packed", "V_R_packed") if residual else ()
    )
    scale_bytes = sum(state[n].numel() * state[n].element_size()
                      for n in scale_names)
    denominator = weight.numel()
    return {
        "shape": [rows, cols], "rank": rank, "seed": seed,
        "residual": residual, "initializer": "joint_itq" if use_itq else "svd_only",
        "relative_frobenius_error": reconstruction_error,
        "forward_dense_max_abs_error": forward_delta,
        "reported_upstream_bpw": float(layer.eff_bit_actual),
        "measured_tensor_bpw": 8 * tensor_bytes / denominator,
        "measured_single_layer_zip_bpw": 8 * memory.tell() / denominator,
        "tensor_bytes": tensor_bytes, "zip_bytes": memory.tell(),
        "scale_bytes": scale_bytes, "scale_dtypes": scale_types,
        "packed_tensor_shapes": {n: list(state[n].shape) for n in names},
        "original_weight_sha256": hashlib.sha256(
            bytes(weight.contiguous().view(torch.uint8).flatten().tolist())
        ).hexdigest(),
    }


def run(root: Path, *, seeds=SEEDS, shapes=SHAPES,
        residual=False, itq_iters=5):
    import torch
    if torch.cuda.is_available() or torch.version.cuda or torch.version.hip:
        raise RuntimeError("CPU-only build required, no GPU/CUDA/ROCm")
    if not str(torch.__version__).startswith("2.6.0+cpu"):
        raise RuntimeError("pinned experiment requires torch 2.6.0+cpu")
    if not 1 <= itq_iters <= 50:
        raise ValueError("ITQ iterations out of bounded range")
    if len(seeds) > 3 or not set(seeds).issubset(SEEDS):
        raise ValueError("seeds exceed preregistered envelope")
    if len(shapes) > 3 or not set(shapes).issubset(SHAPES):
        raise ValueError("shapes exceed preregistered envelope")
    torch.set_num_threads(1)
    official = load_official_class(verify_source(root))
    cases = []
    for seed in seeds:
        for rows, cols, rank in shapes:
            for flag in (False, True):
                cases.append(execute_case(
                    torch, official, rows=rows, cols=cols, rank=rank,
                    seed=seed, use_itq=flag, residual=residual,
                    itq_iters=itq_iters))
    pairs = []
    for index in range(0, len(cases), 2):
        no, yes = cases[index:index + 2]
        pairs.append({
            "shape": no["shape"], "seed": no["seed"], "residual": residual,
            "itq_minus_svd_relative_error":
                yes["relative_frobenius_error"] - no["relative_frobenius_error"],
            "itq_better": yes["relative_frobenius_error"]
                          < no["relative_frobenius_error"],
        })
    mean = sum(p["itq_minus_svd_relative_error"] for p in pairs) / len(pairs)
    return {
        "status": "PASS", "scope": "official_class_CPU_synthetic",
        "upstream_commit": UPSTREAM_COMMIT, "torch_version": str(torch.__version__),
        "seed_list": list(seeds), "shapes": [list(s) for s in shapes],
        "itq_iterations": itq_iters, "residual": residual,
        "model_downloads": 0, "training": False, "gpu_seconds": 0,
        "provider_calls": 0, "production_dispatch": False,
        "cases": cases, "pairs": pairs, "paired_count": len(pairs),
        "itq_better_count": sum(p["itq_better"] for p in pairs),
        "mean_itq_minus_svd_error": mean,
        "caveat": "Upstream randomized svd_lowrank and random rank-one extraction; "
                  "results are recipe-level synthetic parity, not causal ITQ-only "
                  "ablation, heldout language-model performance or QAT.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--execute-cpu", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--itq-iterations", type=int, default=5)
    args = parser.parse_args(argv)
    if not args.execute_cpu:
        parser.error("research execution disarmed: --execute-cpu required")
    result = run(args.upstream_root, residual=args.residual,
                 itq_iters=args.itq_iterations)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
