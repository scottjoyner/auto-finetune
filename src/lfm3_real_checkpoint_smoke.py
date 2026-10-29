"""Real LFM3 BF16 forward-hook smoke, NOT a capability or latency benchmark.

Run ONLY in an isolated, unprivileged, network-disabled, read-only
research sandbox. Emits no prompts, raw logits, traces, or model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter


class ResearchRefused(ValueError):
    pass


def validate_plans(gated: dict, zero: dict) -> None:
    from src.lfm3_head_gate_hook import validate
    validate(gated)
    validate(zero)
    if (gated["model_config_sha256"] != zero["model_config_sha256"]
        or gated["source_vector_sha256"] != zero["source_vector_sha256"]
        or any(x != 0.0 for row in zero["layers"] for x in row["head_logit_deltas"])
        or not any(x != 0.0 for row in gated["layers"] for x in row["head_logit_deltas"])):
        raise ResearchRefused("gate plan and zero-control mismatch")


def verify_restricted_environment() -> None:
    if (os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or Path("/nas").exists()
        or Path("/home/scott/.ssh").exists()):
        raise ResearchRefused("requires offline sandbox excluding fleet mounts/credentials")


def run(snapshot: Path, gated_path: Path, zero_path: Path) -> dict:
    verify_restricted_environment()
    from src.lfm3_checkpoint_attestor import verify_snapshot
    from src.lfm3_head_gate_hook import strict_json, synthetic_head_gate_hooks, ATTENTION_LAYERS
    attest = verify_snapshot(snapshot)
    gated_bytes = gated_path.read_bytes()
    zero_bytes = zero_path.read_bytes()
    gated = strict_json(gated_bytes)
    zero = strict_json(zero_bytes)
    validate_plans(gated, zero)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("unsupported pinned Transformers build")
    torch.set_num_threads(4)
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16
    )
    model.eval()
    if model.__class__.__name__ != "Lfm3ForCausalLM":
        raise ResearchRefused("unexpected checkpoint architecture")
    prompt = [{"role": "user", "content": "What is 2 + 2? Reply with only the digit."}]
    inputs = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    weights = [
        model.model.layers[index].self_attn.gate_proj.weight.detach().clone()
        for index, _ in ATTENTION_LAYERS
    ]
    config_bytes = (snapshot / "config.json").read_bytes()
    source_bytes = (snapshot / "modeling_lfm3.py").read_bytes()
    t = perf_counter()
    with torch.inference_mode():
        base = model(**inputs, use_cache=False).logits[:, -1, :].float().clone()
        with synthetic_head_gate_hooks(
            model, zero, config_bytes, source_bytes, enabled=True
        ) as zero_info:
            zero_logits = model(**inputs, use_cache=False).logits[:, -1, :].float().clone()
        with synthetic_head_gate_hooks(
            model, gated, config_bytes, source_bytes, enabled=True
        ) as gated_info:
            gated_logits = model(**inputs, use_cache=False).logits[:, -1, :].float().clone()
        restored = model(**inputs, use_cache=False).logits[:, -1, :].float().clone()
    seconds = perf_counter() - t
    weight_unchanged = all(
        torch.equal(old, model.model.layers[index].self_attn.gate_proj.weight)
        for (index, _), old in zip(ATTENTION_LAYERS, weights)
    )
    zero_exact = torch.equal(base, zero_logits)
    restored_exact = torch.equal(base, restored)
    changed = not torch.equal(base, gated_logits)
    if not (zero_exact and restored_exact and changed and weight_unchanged):
        raise ResearchRefused("full-checkpoint A/B invariants failed")
    return {
        "schema": "sg-lfm3-real-hook-smoke.v1",
        "model_id": attest["model_id"],
        "revision": attest["revision"],
        "checkpoint_sha256": attest["checkpoint_sha256"],
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "attention_layers": gated_info["hooked_attention_layers"],
        "heads": sum(len(x["head_logit_deltas"]) for x in gated["layers"]),
        "zero_control_exact": zero_exact,
        "restored_exact": restored_exact,
        "gated_changes_logits": changed,
        "model_weights_unchanged": weight_unchanged,
        "max_abs_logit_delta": round((base - gated_logits).abs().max().item(), 6),
        "forward_four_pass_wall_seconds": round(seconds, 3),
        "load_and_four_pass_wall_seconds": round(perf_counter() - started, 3),
        "input_tokens": inputs["input_ids"].numel(),
        "gate_plan_sha256": hashlib.sha256(gated_bytes).hexdigest(),
        "zero_plan_sha256": hashlib.sha256(zero_bytes).hexdigest(),
        "verified_local_checkpoint": True,
        "answer_quality_measured": False,
        "research_only": True,
        "production_ready": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", action="store_true", required=True)
    p.add_argument("--ack-sandbox-research", action="store_true", required=True)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--gate-plan", required=True, type=Path)
    p.add_argument("--zero-plan", required=True, type=Path)
    args = p.parse_args()
    try:
        print(json.dumps(run(args.snapshot, args.gate_plan, args.zero_plan), sort_keys=True))
        return 0
    except (ResearchRefused, ValueError, OSError) as exc:
        raise SystemExit("LFM3 research refused: " + str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
