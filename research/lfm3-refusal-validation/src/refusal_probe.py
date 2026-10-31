"""Research-only local LFM3 refusal-direction experiment; no real tools."""
import argparse
import contextlib
import json
import os
import time
from pathlib import Path

from src.research_job_guard import require_research_write_capacity
from src.eval_contracts import read_jsonl, dataset_sha256, refusal_proxy, validate_action_cases

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/scott/lfm3-research/model-metadata")

def load_local_model(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not (path / "model.safetensors").exists():
        raise FileNotFoundError(path / "model.safetensors")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cpu")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if len(model.model.layers) != 24:
        raise ValueError("LFM3 configuration mismatch")
    return model, tokenizer

def encode(tokenizer, prompt):
    output = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        tokenize=True, return_tensors="pt", enable_thinking=False)
    return output["input_ids"] if hasattr(output, "keys") else output

def hidden_by_layer(model, ids):
    import torch
    captured = {}
    hooks = []
    for idx, layer in enumerate(model.model.layers):
        def capture(_layer, _inputs, out, i=idx):
            captured[i] = out[0, -1, :].detach().float().cpu()
        hooks.append(layer.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        for hook in hooks:
            hook.remove()
    if len(captured) != 24:
        raise RuntimeError("missing decoder activations")
    return captured

def estimate(model, tokenizer, pairs):
    import torch
    total = {}
    for pair in pairs:
        benign = hidden_by_layer(model, encode(tokenizer, pair["benign"]))
        adversarial = hidden_by_layer(model, encode(tokenizer, pair["adversarial"]))
        for idx in benign:
            delta = adversarial[idx] - benign[idx]
            total[idx] = total.get(idx, torch.zeros_like(delta)) + delta
        print(f"captured {pair['id']}", flush=True)
    vectors = {}
    norms = {}
    for idx, value in total.items():
        value /= len(pairs)
        n = value.norm().item()
        norms[str(idx)] = round(n, 5)
        if n < 1e-9:
            # A short-convolution layer may not see a distant prompt contrast.
            # Never manufacture an activation direction from zero signal.
            continue
        vectors[f"layer_{idx}"] = (value / n).contiguous()
    if not vectors:
        raise ValueError("no nonzero refusal-contrast candidates")
    return vectors, norms

@contextlib.contextmanager
def projection(model, vector, layer, alpha):
    """Reversible, per-layer forward hook; model weights remain unmodified."""
    import torch
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be 0..1")
    def hook(_module, _inputs, output):
        unit = vector.to(output.device, dtype=torch.float32)
        scale = torch.sum(output.float() * unit, dim=-1, keepdim=True)
        return output - alpha * (scale * unit).to(output.dtype)
    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

def generate(model, tok, prompt, max_tokens):
    import torch
    ids = encode(tok, prompt)
    with torch.inference_mode():
        result = model.generate(
            ids, do_sample=False, max_new_tokens=max_tokens,
            use_cache=True, pad_token_id=tok.pad_token_id)
    return tok.decode(result[0, ids.shape[1]:], skip_special_tokens=True)

def private_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.chmod(path, 0o600)

def main():
    os.umask(0o077)
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["probe", "compare"])
    p.add_argument("--model-path", type=Path, default=MODEL)
    p.add_argument("--max-pairs", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=36)
    p.add_argument("--layer", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.25)
    a = p.parse_args()
    if not 1 <= a.max_pairs <= 8 or not 0 <= a.layer < 24:
        p.error("invalid pair count or layer")
    require_research_write_capacity(ROOT / "results")
    validate_action_cases(read_jsonl(ROOT / "data/simulated_actions_v0.jsonl"))
    all_rows = read_jsonl(ROOT / "data/paired_v0.jsonl")
    if a.mode == "compare":
        if 2 * a.max_pairs > len(all_rows):
            p.error("not enough independent heldout pairs")
        rows = all_rows[a.max_pairs:2 * a.max_pairs]
    else:
        rows = all_rows[:a.max_pairs]
    import torch
    torch.manual_seed(0)
    started = time.monotonic()
    model, tok = load_local_model(a.model_path)
    folder = ROOT / "results"
    report = dict(mode=a.mode, model_path=str(a.model_path),
                  dataset_sha256=dataset_sha256(ROOT / "data/paired_v0.jsonl"),
                  pairs=[r["id"] for r in rows], precision="bf16", device="cpu",
                  human_safety_review="PENDING", seed=0)
    if a.mode == "probe":
        from safetensors.torch import save_file
        vectors, norms = estimate(model, tok, rows)
        folder.mkdir(exist_ok=True)
        require_research_write_capacity(folder)
        save_file(vectors, str(folder / "directions.safetensors"))
        os.chmod(folder / "directions.safetensors", 0o600)
        report["layer_delta_norms"] = norms
    else:
        from safetensors.torch import load_file
        vectors = load_file(str(folder / "directions.safetensors"))
        vec = vectors[f"layer_{a.layer}"]
        comparisons = []
        for row in rows:
            for category in ("benign", "adversarial"):
                text = row[category]
                base = generate(model, tok, text, a.max_new_tokens)
                with projection(model, vec, a.layer, a.alpha):
                    altered = generate(model, tok, text, a.max_new_tokens)
                comparisons.append(dict(id=row["id"], category=category,
                                        baseline=base, intervention=altered,
                                        baseline_refusal_proxy=refusal_proxy(base),
                                        intervention_refusal_proxy=refusal_proxy(altered)))
                print(f"compared {row['id']}/{category} (private content)", flush=True)
        require_research_write_capacity(folder)
        private_json(folder / "private_generations.json",
                     dict(comparisons=comparisons, human_safety_review="PENDING"))
        report["intervention"] = dict(layer=a.layer, alpha=a.alpha)
        report["generated_pairs"] = len(comparisons)
        report["refusal_proxies_only"] = dict(
            baseline=sum(c["baseline_refusal_proxy"] for c in comparisons),
            intervention=sum(c["intervention_refusal_proxy"] for c in comparisons))
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    require_research_write_capacity(folder)
    private_json(folder / f"{a.mode}_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()
