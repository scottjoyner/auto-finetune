"""Local M1-style experimental weight edit for LFM3. Never overwrite the source.

This is a *smoke artifact* when calibrated on fewer than 16 pairs, not a
validated refusal modification. It must not be auto-served or published.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


DEFAULT_SOURCE = Path("/home/scott/lfm3-research/model-metadata")
DEFAULT_VECTOR = Path(__file__).resolve().parents[1] / "results/directions.safetensors"
DEFAULT_OUTPUT = Path("/home/scott/lfm3-research/refusal-checkpoints/abliterated-m1-smoke")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def residual_writer_names(cfg):
    """M1 candidate: output heads writing into the shared 2048-wide residual stream."""
    for i, layer_type in enumerate(cfg["layer_types"]):
        op = "conv.out_proj" if layer_type == "conv" else "self_attn.o_proj"
        yield f"model.layers.{i}.{op}.weight"
        yield f"model.layers.{i}.feed_forward.w2.weight"


def orthogonalize(W, direction, alpha):
    """(I - alpha * rr^T) W, with normalized r; compute in float32."""
    import torch
    if W.ndim != 2 or direction.ndim != 1 or W.shape[0] != direction.numel():
        raise ValueError("direction does not match a residual-writing matrix")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha out of range")
    r = direction.float()
    r = r / r.norm()
    xf = W.float()
    before = torch.linalg.vector_norm(r @ xf).item()
    result = (xf - alpha * r[:, None] * (r @ xf)[None, :]).to(W.dtype).contiguous()
    after = torch.linalg.vector_norm(r @ result.float()).item()
    return result, before, after


def build(source, vector_path, dest, layer, alpha, allow_smoke):
    import torch
    if not allow_smoke:
        raise ValueError("preliminary two-pair vector: --allow-smoke required")
    source = source.resolve(strict=True)
    vector_path = vector_path.resolve(strict=True)
    dest = dest.resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError("source and destination must be separate directory trees")
    original = source / "model.safetensors"
    config = json.loads((source / "config.json").read_text())
    if config["hidden_size"] != 2048 or config["num_hidden_layers"] != 24:
        raise ValueError("Unexpected architecture: must review residual writers")
    if not config.get("use_sandwich_norm", False):
        raise ValueError("Architecture change: review sandwich normalization")
    vectors = load_file(str(vector_path), device="cpu")
    r = vectors[f"layer_{layer}"]
    expected = tuple(residual_writer_names(config))
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(dest, 0o700)
    output = dest / "model.safetensors"
    if output.exists():
        raise FileExistsError(f"already exists: {output}")
    tmp = dest / "model.safetensors.incomplete"
    if tmp.exists():
        raise FileExistsError(f"stale incomplete artifact: {tmp}")
    state = load_file(str(original), device="cpu")
    if len(expected) != 48 or any(k not in state for k in expected):
        raise ValueError("expected exactly 48 validated residual-writing matrices")
    diagnostics = {}
    for name in expected:
        tensor = state[name]
        updated, before, after = orthogonalize(tensor, r, alpha)
        state[name] = updated
        diagnostics[name] = {"before": round(before, 5), "after": round(after, 5)}
    try:
        save_file(state, str(tmp), metadata={
            "research": "LFM3 refusal study M1-style; unvalidated smoke artifact",
            "base_sha256": digest(original),
            "direction_layer": str(layer),
            "alpha": str(alpha),
        })
        os.chmod(tmp, 0o600)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    for item in source.iterdir():
        if item.is_file() and item.name != "model.safetensors":
            shutil.copy2(item, dest / item.name)
    manifest = {
        "checkpoint_role": "abliterated-experimental",
        "release_authorized": False,
        "publication_approved": False,
        "method": "M1-style row-space weight projection",
        "source_path": str(source),
        "source_sha256": digest(original),
        "checkpoint_sha256": digest(output),
        "direction_path": str(vector_path),
        "direction_sha256": digest(vector_path),
        "direction_layer": layer,
        "alpha": alpha,
        "edited_matrices": len(expected),
        "post_normalization_warning": (
            "LFM3 sandwich RMS norms follow residual writers, so "
            "orthogonality before these norms does not prove full "
            "residual-stream refusal suppression."
        ),
        "calibration_warning": "two training pairs only; not validated",
        "post_test_status": "NOT YET EVALUATED",
        "diagnostics": diagnostics,
    }
    mpath = dest / "RESEARCH_PROVENANCE.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    os.chmod(mpath, 0o600)
    print(json.dumps({k: v for k, v in manifest.items() if k != "diagnostics"}, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--direction", type=Path, default=DEFAULT_VECTOR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--allow-smoke", action="store_true")
    args = p.parse_args()
    build(args.source, args.direction, args.output, args.layer,
          args.alpha, args.allow_smoke)


if __name__ == "__main__":
    main()
