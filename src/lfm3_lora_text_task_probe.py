"""Offline real LFM3-1.7B LoRA optimizer comparison on disjoint text tasks.

Task prompts are procedurally generated, nonprivate, and deterministic.
This is a diagnostic text-task experiment, NOT a public benchmark, final
test, autonomous optimizer, or evidence of generalized improvement.
No adapters/checkpoints are written or promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

from src.lfm3_real_lora_probe import (
    BASE_SHA256, MODEL_PATH, _load, _preflight, _sha, validate,
)

SCHEMA = "sensorgraft.lfm3.lora-text-task-probe.v1"
MAX_STEPS = 20
FAMILIES = ("parity", "greater", "sum")


def _example(family, a, b):
    if family == "parity":
        prompt = f"Answer yes or no. Is {a} an even number? Answer: "
        answer = "yes" if a % 2 == 0 else "no"
        foil = "no" if answer == "yes" else "yes"
    elif family == "greater":
        prompt = f"Answer yes or no. Is {a} greater than {b}? Answer: "
        answer = "yes" if a > b else "no"
        foil = "no" if answer == "yes" else "yes"
    elif family == "sum":
        prompt = f"Write only the number. What is {a} plus {b}? Answer: "
        answer = str(a + b)
        foil = str(a + b + 1)
    else:
        raise ValueError("unsupported_task_family")
    return {"family": family, "prompt": prompt, "answer": answer,
            "foil": foil, "input_key": (family, a, b if family != "parity" else 0)}


def _digest(rows):
    """No private user content is included in this toy curriculum."""
    public = [{k: v for k, v in row.items() if k != "input_key"} for row in rows]
    return hashlib.sha256(json.dumps(
        public, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def dataset(seed):
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("invalid_seed")
    generator = random.Random(200003 + seed)
    sections = {"train": [], "selection": [], "untouched_test": []}
    seen = set()
    # 24 train, 9 selection and 9 untouched future examples. Each partition
    # has all 3 task families and distinct operand tuples.
    for section, count in (("train", 8), ("selection", 3), ("untouched_test", 3)):
        for family in FAMILIES:
            for _ in range(count):
                for _attempt in range(500):
                    a = generator.randrange(2, 81)
                    b = generator.randrange(2, 81)
                    if family == "greater" and a == b:
                        continue
                    example = _example(family, a, b)
                    if example["input_key"] not in seen:
                        seen.add(example["input_key"])
                        sections[section].append(example)
                        break
                else:
                    raise ValueError("dataset_sampling_exhausted")
    # Keep all three task families represented throughout a short run; a
    # contiguous family-block curriculum biases 10-20 step comparisons.
    random.Random(640017 + seed).shuffle(sections["train"])
    return sections


def _ids(tokenizer, prompt, answer):
    p = tokenizer.encode(prompt, add_special_tokens=False)
    a = tokenizer.encode(answer, add_special_tokens=False)
    if not 1 <= len(p) <= 64 or not 1 <= len(a) <= 8:
        raise ValueError("unexpected_text_task_encoding_length")
    if len(p) + len(a) > 72:
        raise ValueError("text_task_context_limit_exceeded")
    return p, a


def _completion_loss(torch, model, tokenizer, item):
    p, a = _ids(tokenizer, item["prompt"], item["answer"])
    ids = torch.tensor([p + a], dtype=torch.long, device="cpu")
    labels = torch.tensor([[-100] * len(p) + a], dtype=torch.long, device="cpu")
    loss = model(input_ids=ids, labels=labels, use_cache=False).loss
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite_text_task_loss")
    return loss


def _choice_logprob(torch, model, tokenizer, prompt, answer):
    p, a = _ids(tokenizer, prompt, answer)
    ids = torch.tensor([p + a], dtype=torch.long, device="cpu")
    with torch.inference_mode():
        output = model(input_ids=ids, use_cache=False)
        logits = output.logits[0, len(p) - 1:-1, :].float()
        chosen = logits.log_softmax(-1).gather(
            -1, torch.tensor(a, dtype=torch.long).unsqueeze(-1)
        ).squeeze(-1)
        score = float(chosen.mean())
    if not math.isfinite(score):
        raise RuntimeError("nonfinite_choice_score")
    return score


def _evaluate(torch, model, tokenizer, rows):
    model.eval()
    by_family = {name: [0, 0] for name in FAMILIES}
    margins = []
    for row in rows:
        good = _choice_logprob(torch, model, tokenizer, row["prompt"], row["answer"])
        bad = _choice_logprob(torch, model, tokenizer, row["prompt"], row["foil"])
        is_correct = good > bad
        counts = by_family[row["family"]]
        counts[0] += int(is_correct)
        counts[1] += 1
        margins.append(good - bad)
    return {"correct": sum(x[0] for x in by_family.values()),
            "total": sum(x[1] for x in by_family.values()),
            "mean_correct_minus_foil_logprob": sum(margins) / len(margins),
            "families": {n: {"correct": k[0], "total": k[1]}
                         for n, k in by_family.items()}}


def run(*, seeds=(1,), steps=10, optimizers=("adamw", "muon"),
        adamw_lr=0.0001, muon_lr=0.0001, experimental_muon=False,
        model_path=MODEL_PATH):
    validate(seeds=seeds, steps=steps, optimizers=optimizers,
             adamw_lr=adamw_lr, muon_lr=muon_lr,
             experimental_muon=experimental_muon)
    if steps > MAX_STEPS or set(optimizers) != {"adamw", "muon"}:
        raise ValueError("paired_text_probe_requires_both_optimizers_and_bounded_steps")
    _preflight()
    import torch
    from transformers import AutoTokenizer
    from src.lfm3_support import patch_transformers_for_lfm3
    torch.set_num_threads(2)
    patch_transformers_for_lfm3()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True)
    results = []
    for seed in seeds:
        parts = dataset(seed)
        hashes = {name: _digest(rows) for name, rows in parts.items()}
        if len(set(hashes.values())) != 3:
            raise RuntimeError("nonindependent_data_partition_hash")
        paired = []
        for name in optimizers:
            model, named = _load(torch, seed, str(model_path))
            params = [p for _, p in named]
            initial_adapter_hash = _sha(*params)
            backbone = next(p for n, p in model.named_parameters()
                            if n.endswith("model.layers.2.self_attn.q_proj.base_layer.weight"))
            backbone_copy = backbone.detach().reshape(-1)[:256].clone()
            optimizer = (torch.optim.AdamW(params, lr=adamw_lr, weight_decay=0)
                         if name == "adamw"
                         else torch.optim.Muon(params, lr=muon_lr, weight_decay=0))
            lr = adamw_lr if name == "adamw" else muon_lr
            before = _evaluate(torch, model, tokenizer, parts["selection"])
            events = []
            for step in range(steps):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                item = parts["train"][step % len(parts["train"])]
                t0 = time.perf_counter()
                loss = _completion_loss(torch, model, tokenizer, item)
                loss.backward()
                raw_grad = float(torch.nn.utils.clip_grad_norm_(params, max_norm=1.0))
                if not math.isfinite(raw_grad) or raw_grad > 1e5:
                    raise RuntimeError("unsafe_or_nonfinite_gradient")
                snapshot = [p.detach().clone() for p in params]
                optimizer.step()
                update_sq = sum(float((p.detach() - old).float().square().sum())
                                for p, old in zip(params, snapshot))
                if not math.isfinite(update_sq):
                    raise RuntimeError("nonfinite_optimizer_update")
                events.append({
                    "step": step, "optimizer": name, "learning_rate": lr,
                    "train_loss": float(loss.detach()), "unclipped_grad_norm": raw_grad,
                    "grad_clip_limit": 1.0, "update_norm": math.sqrt(update_sq),
                    "elapsed_s": max(time.perf_counter() - t0, 1e-9),
                })
            after = _evaluate(torch, model, tokenizer, parts["selection"])
            if not torch.equal(backbone.detach().reshape(-1)[:256], backbone_copy):
                raise RuntimeError("frozen_backbone_sample_changed")
            record = {
                "run_id": uuid.uuid4().hex, "seed": seed, "optimizer": name,
                "learning_rate": lr, "trainable_parameters": sum(p.numel() for p in params),
                "initial_adapter_sha256": initial_adapter_hash,
                "final_adapter_sha256": _sha(*params),
                "selection_before": before, "selection_after": after,
                "events": events, "weight_promotion_permitted": False,
            }
            if record["initial_adapter_sha256"] == record["final_adapter_sha256"]:
                raise RuntimeError("no_adapter_update")
            paired.append(record)
            print(json.dumps({"seed": seed, "optimizer": name,
                              "selection_before": before["correct"],
                              "selection_after": after["correct"],
                              "selection_total": after["total"]}), flush=True)
            del model, optimizer
        if len({a["initial_adapter_sha256"] for a in paired}) != 1:
            raise RuntimeError("unpaired_initial_adapters")
        results.extend(paired)
    manifests = {str(seed): {k: _digest(v) for k, v in dataset(seed).items()} for seed in seeds}
    return {
        "schema": SCHEMA, "status": "research_text_task_diagnostic_only",
        "base_sha256": BASE_SHA256, "model": "LiquidAI/LFM3-1.7B",
        "data_kind": "procedural_public_arithmetic_and_parity_text",
        "answer_metric": "two_choice_teacher_forced_logprob",
        "selection_used_for_reporting_not_optimizer_choice": True,
        "untouched_test_evaluated": False,
        "dataset_sha256_by_seed": manifests, "seeds": list(seeds), "steps": steps,
        "arms": results, "real_task_generalization_established": False,
        "weight_promotion_permitted": False, "trial_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-only", action="store_true", required=True)
    parser.add_argument("--allow-real-lora-training", action="store_true", required=True)
    parser.add_argument("--allow-experimental-muon-lora", action="store_true", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--adamw-lr", type=float, default=0.0001)
    parser.add_argument("--muon-lr", type=float, default=0.0001)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(seeds=args.seeds, steps=args.steps, adamw_lr=args.adamw_lr,
                 muon_lr=args.muon_lr, experimental_muon=True,
                 model_path=args.model_path)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps({"status": report["status"],
                      "weight_promotion_permitted": False, "output": str(args.output)}))


if __name__ == "__main__":
    main()
