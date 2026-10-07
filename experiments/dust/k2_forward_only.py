"""Isolated last-layer K2 o_proj, forward-only LoRA smoke (NOT full Dust).

Last decoder layer has no downstream attention, so position-aligned causal
next-token loss estimates the local o_proj activation error. This property
does NOT apply to earlier layers or attention Q/K/V projections.

Defaults to tiny random K2 architecture; real pretrained K2 is explicit opt-in.
Never writes model weights or changes the scheduler, NAS, or training pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time


def memory_available_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def build_model(model_dir: Path, *, tiny: bool, device: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError("Local K2 model configuration missing")
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    if getattr(config, "model_type", None) != "k2_horizon":
        raise ValueError("Refusing non-K2 architecture")
    if tiny:
        config.num_hidden_layers = 2
        config.hidden_size = 64
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 16
        config.rope_head_dim = 16
        config.intermediate_size = 128
        config.vocab_size = 256
        config.pad_token_id = 0
        config.max_position_embeddings = 128
        config.mlp_only_layers = [0, 1]
        config.rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
        config.use_cache = False
        return AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(device).eval()
    if device != "cpu":
        raise ValueError("Pretrained K2 smoke is CPU-only pending a separate ROCm memory gate")
    if memory_available_bytes() < 4 * 1024**3:
        raise MemoryError("Not enough memory headroom for the pretrained K2 forward smoke")
    return AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
        device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()

def position_losses(logits, labels):
    """Returns masked next-token CE aligned to the source logit position."""
    import torch
    from torch.nn import functional as F

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected [batch, sequence, vocab] logits and [batch, sequence] labels")
    if labels.shape[1] < 2:
        raise ValueError("At least two tokens required for causal CE")
    shifted = labels[:, 1:]
    valid = shifted.ne(-100)
    if not bool(valid.any()):
        raise ValueError("No unmasked next-token targets")
    losses = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        shifted.clamp_min(0).reshape(-1),
        reduction="none",
    ).reshape_as(shifted)
    masked = torch.where(valid, losses, torch.zeros_like(losses))
    return F.pad(masked, (0, 1)), F.pad(valid, (0, 1), value=False)


class FinalProjectionLoRA:
    """Temporary hook: captures K2 last-layer o_proj inputs; adds LoRA and noise."""
    def __init__(self, model, *, rank: int, seed: int):
        import torch

        self.module = model.model.layers[-1].self_attn.o_proj
        if not isinstance(self.module, torch.nn.Linear):
            raise TypeError("Expected actual K2 last-layer nn.Linear o_proj")
        if self.module.in_features <= 0 or self.module.out_features <= 0:
            raise ValueError("Invalid projection shape")
        dev = self.module.weight.device
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.a = (torch.randn(rank, self.module.in_features, generator=gen) /
                  math.sqrt(self.module.in_features)).to(dev)
        self.b = torch.zeros(self.module.out_features, rank, device=dev)
        self.rank = rank
        self.jitter = None
        self.inputs = None
        self.handle = self.module.register_forward_hook(self._hook)

    def _hook(self, _module, args, output):
        import torch
        inputs = args[0].detach().float()
        self.inputs = inputs
        update = (inputs @ self.a.T) @ self.b.T
        if self.jitter is not None:
            if tuple(self.jitter.shape) != tuple(output.shape):
                raise ValueError("Activation perturbation shape mismatch")
            update = update + self.jitter
        return output + update.to(dtype=output.dtype)

    def close(self):
        self.handle.remove()

    def apply(self, output_grad, *, count: int, lr: float):
        import torch
        if self.inputs is None or output_grad.shape != (*self.inputs.shape[:2], self.b.shape[0]):
            raise ValueError("Missing cached input or incorrect estimated gradient")
        with torch.no_grad():
            x = self.inputs.reshape(-1, self.inputs.shape[-1])
            dy = output_grad.reshape(-1, self.b.shape[0])
            z = x @ self.a.T
            grad_b = dy.T @ z / count
            grad_a = (dy @ self.b).T @ x / count
            self.b.add_(grad_b, alpha=-lr)
            self.a.add_(grad_a, alpha=-lr)

def run(*, model_dir: Path, tiny: bool = True, device: str = "cpu",
        seed: int = 42, steps: int = 1, draws: int = 16,
        rank: int = 4, sigma: float = 0.05, lr: float = 1.0) -> dict:
    """One bounded, temporary adapter experiment; no persistent weight writes."""
    import torch

    if not 1 <= steps <= (2 if tiny else 1):
        raise ValueError("Pretrained step limit is 1; tiny step limit is 2")
    if not 1 <= draws <= (128 if tiny else 4):
        raise ValueError("Pretrained draw limit is 4; tiny draw limit is 128")
    if rank not in (2, 4, 8) or not (0 < sigma <= 1) or not (0 < lr <= 2):
        raise ValueError("Invalid LoRA rank or perturbation hyperparameters")
    if not tiny and device != "cpu":
        raise ValueError("Pretrained K2 currently requires CPU until GPU admission is proven")
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Selected PyTorch device not available")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(seed)
    model = build_model(model_dir, tiny=tiny, device=device)
    model.requires_grad_(False)
    source_projection = model.model.layers[-1].self_attn.o_proj
    original_weight = source_projection.weight.detach().clone()
    adapter = FinalProjectionLoRA(model, rank=rank, seed=seed + 17)
    gen = torch.Generator(device=device).manual_seed(seed + 971)
    ids = torch.tensor([[1, 20, 5, 12, 3, 32, 17, 4]], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, :2] = -100  # known masked prompt prefix; explicit causal shift below
    history = []
    t0 = time.monotonic()
    try:
        with torch.no_grad():
            def score():
                if not tiny and memory_available_bytes() < 1 * 1024**3:
                    raise MemoryError("System available memory fell below 1 GiB")
                logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
                return position_losses(logits, labels)

            for step in range(steps):
                adapter.jitter = None
                clean, mask = score()
                if not bool(torch.isfinite(clean).all()):
                    raise ArithmeticError("Nonfinite clean token loss")
                history.append(clean.sum().item() / mask.sum().item())
                errors = torch.zeros(*ids.shape, adapter.b.shape[0], dtype=torch.float32,
                                     device=device)
                for draw in range(draws):
                    noise = torch.randn(errors.shape, generator=gen, device=device,
                                        dtype=torch.float32)
                    adapter.jitter = sigma * noise
                    try:
                        perturbed, changed_mask = score()
                    finally:
                        adapter.jitter = None
                    if not torch.equal(mask, changed_mask):
                        raise AssertionError("Mask changed between clean and perturbed pass")
                    if not bool(torch.isfinite(perturbed).all()):
                        raise ArithmeticError("Nonfinite perturbed token loss")
                    reward = (clean - perturbed) * mask
                    errors -= reward.unsqueeze(-1) * noise / (sigma * draws)
                adapter.apply(errors, count=int(mask.sum()), lr=lr)
            final, mask = score()
            history.append(final.sum().item() / mask.sum().item())
            weight_unchanged = bool(torch.equal(original_weight, source_projection.weight))
            gradient_absent = all(p.grad is None for p in model.parameters())
            if not weight_unchanged or not gradient_absent:
                raise AssertionError("Base model weights or autograd state changed")
            if not all(math.isfinite(v) for v in history):
                raise ArithmeticError("Nonfinite history")
            a_norm = adapter.a.norm().item()
            b_norm = adapter.b.norm().item()
    finally:
        adapter.close()
    return {
        "experiment": "K2Horizon_last_o_proj_forward_only_smoke",
        "architecture": type(model).__name__, "tiny_random_model": tiny,
        "pretrained_forward_smoke": not tiny, "last_layer_only": True,
        "real_dataset_used": False, "heldout_evaluated": False,
        "tokens": int(ids.numel()), "masked_prefix_tokens": 2,
        "seed": seed, "device": device, "draws_per_step": draws,
        "steps": steps, "rank": rank, "sigma": sigma, "lr": lr,
        "model_base_unchanged": weight_unchanged, "all_model_grads_absent": gradient_absent,
        "backward_calls": 0, "checkpoint_written": False,
        "loss_history": history, "loss_improved_on_synthetic_tokens": history[-1] < history[0],
        "adapter_a_norm": a_norm, "adapter_b_norm": b_norm,
        "elapsed_seconds": time.monotonic() - t0,
        "torch_version": torch.__version__, "torch_hip": torch.version.hip,
        "disclaimer": "Synthetic token IDs on same sequence; not Dust parity or K2 finetuning evidence",
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Existing LOCAL K2-Horizon directory; no downloads")
    parser.add_argument("--pretrained", action="store_true",
                        help="Explicit CPU-only load of real pretrained K2 weights")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--output", type=Path, help="Create new JSON evidence (never overwrite)")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Existing evidence must not be overwritten; refusing before loading model")
    result = run(
        model_dir=args.model_dir, tiny=not args.pretrained, device=args.device,
        seed=args.seed, steps=args.steps, draws=args.draws,
        rank=args.rank, sigma=args.sigma, lr=args.lr,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        try:
            with args.output.open("x", encoding="utf-8") as f:
                f.write(encoded)
        except FileExistsError:
            parser.error("Existing evidence must not be overwritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
