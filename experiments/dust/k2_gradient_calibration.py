"""Calibrate Dust-inspired last-o_proj activation gradients against exact local autograd.

Uses one real K2-Horizon training example selected deterministically by SHA-256.
Compares one-sided and antithetic Gaussian activation estimators over a bounded
population/sigma grid. The pretrained base is frozen and no checkpoint is saved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from experiments.dust.k2_data import read_pairs, tokenize_pair
from experiments.dust.k2_forward_only import position_losses
from experiments.dust.k2_matched_compare import digest, load_base, read_mem_available


class LocalOutputProbe:
    """Cut graph at final o_proj output, capture input, optionally add jitter."""

    def __init__(self, model):
        import torch
        self.module = model.model.layers[-1].self_attn.o_proj
        if not isinstance(self.module, torch.nn.Linear):
            raise TypeError("Expected final K2 attention o_proj nn.Linear")
        self.exact = False
        self.jitter = None
        self.leaf = None
        self.inputs = None
        self.handle = self.module.register_forward_hook(self._hook)

    def _hook(self, _module, args, output):
        value = output.detach().float()
        self.inputs = args[0].detach().float()
        self.leaf = None
        if self.exact:
            value = value.requires_grad_(True)
            self.leaf = value
        if self.jitter is not None:
            if tuple(self.jitter.shape) != tuple(value.shape):
                raise ValueError("Perturbation shape mismatch")
            value = value + self.jitter
        return value.to(dtype=output.dtype)

    def close(self):
        self.handle.remove()


def select_train_example(train_file: Path, tokenizer, *, index: int,
                         max_tokens: int) -> tuple[dict, dict]:
    entries, source = read_pairs(train_file)
    valid = []
    for pair_hash, (_, pair) in sorted(entries.items()):
        try:
            row = tokenize_pair(tokenizer, pair, max_tokens)
        except (ValueError, TypeError, KeyError, AttributeError):
            row = None
        if row:
            valid.append((pair_hash, row))
        if len(valid) > index:
            break
    if len(valid) <= index:
        raise ValueError("Not enough deterministic tokenizable train examples")
    pair_hash, row = valid[index]
    return row, {
        "source": source, "selected_pair_sha256": pair_hash,
        "selected_index": index, "sequence_tokens": len(row["tokens"]),
        "assistant_tokens": row["assistant_tokens"],
        "max_tokens": max_tokens,
    }


def tensors(sample: dict, device: str):
    import torch
    ids = torch.tensor([sample["tokens"]], device=device, dtype=torch.long)
    labels = torch.tensor([sample["labels"]], device=device, dtype=torch.long)
    return ids, labels


def cosine_and_error(estimate, exact) -> dict:
    import torch
    a = estimate.detach().float().flatten()
    b = exact.detach().float().flatten()
    nb = torch.linalg.vector_norm(b)
    na = torch.linalg.vector_norm(a)
    if float(nb) <= 1e-20:
        raise ArithmeticError("Exact gradient norm is effectively zero")
    cosine = torch.dot(a, b) / (torch.clamp(na, min=1e-20) * nb)
    rel = torch.linalg.vector_norm(a - b) / nb
    return {
        "cosine": float(cosine.item()),
        "relative_l2_error": float(rel.item()),
        "norm_ratio": float((na / nb).item()),
    }


def score_with_probe(model, probe: LocalOutputProbe, ids, labels):
    logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
    losses, mask = position_losses(logits, labels)
    if not bool(losses.isfinite().all()):
        raise ArithmeticError("Nonfinite token loss")
    return losses, mask


def exact_local_gradient(model, probe: LocalOutputProbe, ids, labels):
    import torch
    probe.exact = True
    probe.jitter = None
    with torch.enable_grad():
        losses, mask = score_with_probe(model, probe, ids, labels)
        count = int(mask.sum().item())
        if count <= 0 or probe.leaf is None or probe.inputs is None:
            raise AssertionError("Local gradient probe did not capture valid state")
        mean_ce = losses.sum() / count
        gradient = torch.autograd.grad(mean_ce, probe.leaf, retain_graph=False)[0]
    probe.exact = False
    return gradient.detach().float(), probe.inputs.detach().float(), mask.detach(), float(mean_ce.item())


def clean_token_losses(model, probe: LocalOutputProbe, ids, labels):
    import torch
    probe.exact = False
    probe.jitter = None
    with torch.no_grad():
        losses, mask = score_with_probe(model, probe, ids, labels)
    return losses.detach().float(), mask.detach()


def adapter_b_gradient(output_gradient, inputs, a):
    x = inputs.reshape(-1, inputs.shape[-1])
    z = x @ a.T
    dy = output_gradient.reshape(-1, output_gradient.shape[-1])
    return dy.T @ z


def calibrate(model, sample: dict, *, device: str, seeds: list[int],
              sigmas: list[float], populations: list[int], rank: int,
              max_seconds: int, direction_mode: str = "gaussian") -> dict:
    import torch
    if sorted(set(populations)) != populations or populations[0] < 1:
        raise ValueError("Populations must be unique, sorted positive integers")
    if populations[-1] > 2048:
        raise ValueError("Calibration population capped at 2048")
    if direction_mode not in ("gaussian", "orthogonal"):
        raise ValueError("Unknown perturbation direction mode")
    if any(not (0 < sigma <= 0.25) for sigma in sigmas):
        raise ValueError("Calibration sigma outside (0,0.25]")
    if rank not in (2, 4, 8):
        raise ValueError("Calibration rank must be 2, 4 or 8")
    if len(seeds) > 8 or len(sigmas) > 8:
        raise ValueError("Calibration grid too large")
    ids, labels = tensors(sample, device)
    probe = LocalOutputProbe(model)
    original_weight = probe.module.weight.detach().clone()
    started = time.monotonic()
    try:
        exact_dy, cached_inputs, exact_mask, exact_ce = exact_local_gradient(
            model, probe, ids, labels)
        clean, clean_mask = clean_token_losses(model, probe, ids, labels)
        if not torch.equal(exact_mask, clean_mask):
            raise AssertionError("Exact/clean causal masks differ")
        count = int(clean_mask.sum().item())
        # Keep the same LoRA A basis across estimator seeds/sigmas so B-gradient
        # alignment changes only with the activation estimator.
        a_gen = torch.Generator(device="cpu").manual_seed(20261007)
        a = (torch.randn(rank, probe.module.in_features, generator=a_gen) /
             math.sqrt(probe.module.in_features)).to(device)
        exact_b = adapter_b_gradient(exact_dy, cached_inputs, a)
        exact_output_norm = float(torch.linalg.vector_norm(exact_dy).item())
        exact_b_norm = float(torch.linalg.vector_norm(exact_b).item())
        points = []
        max_population = populations[-1]
        forward_count = 2  # exact + clean
        hidden = exact_dy.shape[-1]
        for seed in seeds:
            orthogonal = None
            if direction_mode == "orthogonal":
                if max_population > hidden:
                    raise ValueError("Orthogonal population cannot exceed hidden size")
                basis_gen = torch.Generator(device=device).manual_seed(seed)
                raw = torch.randn(
                    hidden, max_population, generator=basis_gen,
                    device=device, dtype=torch.float32)
                orthogonal = (
                    torch.linalg.qr(raw, mode="reduced").Q.T
                    * math.sqrt(hidden)
                ).contiguous()
                del raw
            for sigma in sigmas:
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("Calibration exceeded wall-clock budget")
                generator = torch.Generator(device=device).manual_seed(seed)
                one_sum = torch.zeros_like(exact_dy)
                anti_sum = torch.zeros_like(exact_dy)
                checkpoints = set(populations)
                for draw in range(1, max_population + 1):
                    if read_mem_available() < 1536 * 1024**2:
                        raise MemoryError("Host memory below calibration safety floor")
                    if orthogonal is None:
                        noise = torch.randn(
                            exact_dy.shape, generator=generator, device=device,
                            dtype=torch.float32)
                    else:
                        noise = orthogonal[draw - 1].view(1, 1, hidden).expand_as(exact_dy)
                    probe.jitter = sigma * noise
                    with torch.no_grad():
                        plus, plus_mask = score_with_probe(model, probe, ids, labels)
                    probe.jitter = -sigma * noise
                    with torch.no_grad():
                        minus, minus_mask = score_with_probe(model, probe, ids, labels)
                    probe.jitter = None
                    forward_count += 2
                    if not torch.equal(clean_mask, plus_mask) or not torch.equal(clean_mask, minus_mask):
                        raise AssertionError("Perturbation changed causal mask")
                    valid = clean_mask.float()
                    one_sum += (((plus.float() - clean) * valid).unsqueeze(-1)
                                * noise / sigma)
                    anti_sum += (((plus.float() - minus.float()) * valid).unsqueeze(-1)
                                 * noise / (2.0 * sigma))
                    if draw not in checkpoints:
                        continue
                    for method, raw in (("one_sided", one_sum), ("antithetic", anti_sum)):
                        estimate = raw / (draw * count)
                        b_estimate = adapter_b_gradient(estimate, cached_inputs, a)
                        point = {
                            "seed": seed, "sigma": sigma, "population": draw,
                            "estimator": method, "direction_mode": direction_mode,
                            "output_gradient": cosine_and_error(estimate, exact_dy),
                            "lora_b_gradient": cosine_and_error(b_estimate, exact_b),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                        points.append(point)
        if not torch.equal(original_weight, probe.module.weight):
            raise AssertionError("Base o_proj weight changed during calibration")
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError("Frozen base accumulated gradients")
        return {
            "exact_mean_train_token_ce": exact_ce,
            "scored_tokens": count,
            "sequence_tokens": int(ids.shape[1]),
            "rank": rank, "seeds": seeds, "sigmas": sigmas,
            "populations": populations, "direction_mode": direction_mode,
            "exact_output_gradient_norm": exact_output_norm,
            "exact_lora_b_gradient_norm": exact_b_norm,
            "points": points, "forward_count": forward_count,
            "elapsed_seconds": time.monotonic() - started,
            "base_unchanged": True, "base_model_grads_absent": True,
            "checkpoint_written": False,
        }
    finally:
        probe.jitter = None
        probe.close()


def parse_int_list(value: str) -> list[int]:
    items = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return items


def parse_float_list(value: str) -> list[float]:
    items = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected comma-separated floats")
    return items


def summarize(points: list[dict]) -> list[dict]:
    from collections import defaultdict
    from statistics import mean
    groups = defaultdict(list)
    for p in points:
        groups[(p["estimator"], p["sigma"], p["population"])].append(p)
    out = []
    for (estimator, sigma, population), rows in sorted(groups.items()):
        out.append({
            "estimator": estimator, "sigma": sigma, "population": population,
            "seeds": len(rows),
            "mean_output_cosine": mean(r["output_gradient"]["cosine"] for r in rows),
            "mean_output_relative_l2_error": mean(
                r["output_gradient"]["relative_l2_error"] for r in rows),
            "mean_lora_b_cosine": mean(
                r["lora_b_gradient"]["cosine"] for r in rows),
            "mean_lora_b_relative_l2_error": mean(
                r["lora_b_gradient"]["relative_l2_error"] for r in rows),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expected-sha256", default=(
        "6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--direction-mode", choices=("gaussian", "orthogonal"), default="gaussian")
    parser.add_argument("--seeds", type=parse_int_list, default=[7, 42, 1337])
    parser.add_argument("--sigmas", type=parse_float_list,
                        default=[0.01, 0.025, 0.05, 0.1])
    parser.add_argument("--populations", type=parse_int_list,
                        default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--max-seconds", type=int, default=360)
    parser.add_argument("--output", type=Path, help="Exclusive-create JSON evidence")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error("Refusing to overwrite existing evidence before model load")
    if not 0 <= args.sample_index <= 32:
        parser.error("sample-index outside [0,32]")
    if not 16 <= args.max_tokens <= 256:
        parser.error("max-tokens outside [16,256]")
    if not 30 <= args.max_seconds <= 900:
        parser.error("max-seconds outside [30,900]")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_dir), trust_remote_code=True, local_files_only=True)
    sample, selection = select_train_example(
        args.train_jsonl, tokenizer, index=args.sample_index,
        max_tokens=args.max_tokens)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, model_sha = load_base(
        args.model_dir, device=args.device, expected_sha=args.expected_sha256)
    try:
        result = calibrate(
            model, sample, device=args.device, seeds=args.seeds,
            sigmas=args.sigmas, populations=args.populations,
            rank=args.rank, max_seconds=args.max_seconds,
            direction_mode=args.direction_mode)
        result["summary"] = summarize(result["points"])
        report = {
            "schema": "auto-finetune.dust-k2-gradient-calibration.v1",
            "research_only": True, "production_promotion_authorized": False,
            "model_weights_sha256": model_sha,
            "model_config_sha256": digest(args.model_dir / "config.json"),
            "device": args.device, "torch": torch.__version__,
            "hip": torch.version.hip, "selection": selection,
            "calibration": result,
            "max_device_memory_allocated": (
                torch.cuda.max_memory_allocated() if args.device == "cuda" else None),
            "remaining_host_mem_available_bytes": read_mem_available(),
            "raw_user_data_in_report": False,
            "warning": (
                "Last-layer local-gradient calibration only; this does not validate "
                "earlier-layer causal credit, generalization, or upstream Dust parity."),
        }
    finally:
        del model
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        with args.output.open("x", encoding="utf-8") as f:
            f.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
