"""Local three-arm LFM3 comparison with no executable tools or public responses."""
import argparse
import gc
import json
import os
from pathlib import Path

from src.build_abliterated_checkpoint import digest
from src.eval_contracts import read_jsonl, refusal_proxy, validate_action_cases, dataset_sha256
from src.refusal_probe import load_local_model, generate

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/home/scott/lfm3-research")
ARMS = {
    "A_original_posttrained": MODEL_ROOT / "model-metadata",
    "B_M1_abliterated_smoke": MODEL_ROOT / "refusal-checkpoints/abliterated-m1-smoke",
    "C_base_pretrained": MODEL_ROOT / "refusal-checkpoints/base",
}


def checked_arm_paths(paths):
    """Prevent aliasing or quietly evaluating the wrong three model files."""
    hashes = {}
    for key, path in paths.items():
        p = path / "model.safetensors"
        if not p.is_file():
            raise FileNotFoundError(f"{key} absent: {p}")
        hashes[key] = digest(p)
    if len(set(hashes.values())) != len(hashes):
        raise ValueError("checkpoint SHA-256 collision/alias across experiment arms")
    prov = paths["B_M1_abliterated_smoke"] / "RESEARCH_PROVENANCE.json"
    if not prov.is_file():
        raise ValueError("abliterated checkpoint lacks provenance")
    metadata = json.loads(prov.read_text())
    if (metadata["source_sha256"] != hashes["A_original_posttrained"]
            or metadata["checkpoint_sha256"] != hashes["B_M1_abliterated_smoke"]):
        raise ValueError("abliterated checkpoint does not match parent and output hashes")
    return hashes


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-heldout-pairs", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.max_heldout_pairs < 1 or args.max_heldout_pairs > 6:
        parser.error("only 1..6 provisional disjoint heldout pairs are permitted")
    if not 1 <= args.max_new_tokens <= 96:
        parser.error("token cap 1..96")
    hashes = checked_arm_paths(ARMS)
    pairs_path = ROOT / "data/paired_v0.jsonl"
    pairs = read_jsonl(pairs_path)[2:2 + args.max_heldout_pairs]
    actions = read_jsonl(ROOT / "data/simulated_actions_v0.jsonl")
    validate_action_cases(actions)
    results = []
    import torch
    for name, path in ARMS.items():
        torch.manual_seed(0)
        model, tokenizer = load_local_model(path)
        for row in pairs:
            for variant in ("benign", "adversarial"):
                response = generate(model, tokenizer, row[variant], args.max_new_tokens)
                results.append(dict(arm=name, case=row["id"], group=variant,
                                    response=response, refusal_proxy=refusal_proxy(response)))
                print("completed", name, row["id"], variant, flush=True)
        del model
        del tokenizer
        gc.collect()
    args.results.mkdir(parents=True, exist_ok=True)
    private_path = args.results / "private_three_checkpoint_generations.json"
    private_path.write_text(json.dumps(results, indent=2))
    os.chmod(private_path, 0o600)
    summary = {
        "experiment": "three-checkpoint-LFM3-1.7B-smoke",
        "research_only": True,
        "model_hashes_sha256": hashes,
        "dataset_hash_sha256": dataset_sha256(pairs_path),
        "fitting_cases": ["p01", "p02"],
        "heldout_cases": [x["id"] for x in pairs],
        "max_new_tokens": args.max_new_tokens,
        "deterministic_seed": 0,
        "device": "cpu",
        "precision": "bf16",
        "per_arm": {
            arm: {
                "sample_count": len([r for r in results if r["arm"] == arm]),
                "refusal_proxy_count": sum(r["refusal_proxy"] for r in results if r["arm"] == arm),
            } for arm in ARMS
        },
        "mock_authorization_cases_passed": len(actions),
        "limitations": [
            "SMOKE ONLY: two calibration pairs and very small heldout set",
            "twenty generated tokens cannot evaluate complete agentic actions",
            "Base pretrain is not directly instruction-tuned; conversational scores are descriptive",
            "No causal or safety claims; lexical refusal is not policy compliance",
            "No human-graded content/action outcomes, CI, random-direction or multi-seed controls",
        ],
        "human_review": "PENDING",
        "modified_model_release": "NOT AUTHORIZED",
    }
    summary_path = args.results / "three_checkpoint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    os.chmod(summary_path, 0o600)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
