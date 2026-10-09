"""Bounded observation-only real K2 final-o_proj antithetic probe.

Runs all directions of the original shared-orthogonal estimator, writes a
fsynced keyed episode PRE/POST ledger and validated scalar labels. It never
executes a LoRA update, trains a classifier, stores user text, or writes a
model checkpoint. Historical labels are NOT used to choose directions.

Requires explicit local model SHA, private episode HMAC key file (mode 600),
local train source, and exclusively created local evidence paths. This is
only a producer-local timing witness; independent custody is NOT established.

Example:
  python -m experiments.dust.k2_real_direction_probe --observe-only \
    --model-dir /local/K2 --train-jsonl /local/clean/train.jsonl \
    --expected-sha256 <verified model weights SHA256> \
    --episode-key-file /local/private-hmac.key \
    --events /local/new.events.jsonl --derived /local/new.derived.jsonl \
    --output /local/new.summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def run(*, model_dir: Path, train_file: Path, expected_sha: str,
        episode_key_file: Path, events: Path, derived: Path,
        sample_index: int = 0, seed: int = 42, population: int = 64,
        sigma: float = 0.25, direction_batch: int = 4,
        max_tokens: int = 128, device: str = "cuda",
        receiver_run_id: str | None = None,
        preflight_manifest: Path | None = None,
        expected_preflight_sha256: str | None = None,
        collect_causal_history_v2: bool = False) -> dict:
    if preflight_manifest is None or expected_preflight_sha256 is None:
        raise ValueError("source-group preflight and pinned manifest hash required")
    if population not in (8, 16, 32, 64):
        raise ValueError("first witness is capped at 64 directions")
    if direction_batch != 4 or population % direction_batch:
        raise ValueError("first witness uses unchanged direction batch=4")
    if sigma != .25 or seed not in (7, 42, 1337):
        raise ValueError("predeclared sigma/seeds required")
    if not 0 <= sample_index < 64 or not 16 <= max_tokens <= 128:
        raise ValueError("source size or index out of bound")
    if device not in ("cpu", "cuda"):
        raise ValueError("unsupported device")
    if len(expected_sha) != 64 or any(c not in "0123456789abcdef"
                                      for c in expected_sha):
        raise ValueError("expected weight SHA256 required")
    from .k2_direction_witness import (
        LocalProbeWitness, SSHReceiver, pseudonym, read_private_key,
    )
    key = read_private_key(episode_key_file)
    import torch
    from transformers import AutoTokenizer
    from .k2_gradient_calibration import select_train_example
    from .k2_matched_compare import load_base, digest
    from .k2_tail_replay import (
        cache_base_sample, init_lora, tail_scored_structured_estimate,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True)
    sample, selection = select_train_example(
        train_file, tokenizer, index=sample_index, max_tokens=max_tokens)
    del tokenizer
    episode = pseudonym(key, sample)
    del key
    from .k2_cohort_preflight import authorize_index
    manifest_raw = preflight_manifest.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != expected_preflight_sha256:
        raise ValueError("tampered or unpinned source cohort manifest")
    selected_group = authorize_index(
        json.loads(manifest_raw),
        source_sha256=digest(train_file),
        model_config_sha256=digest(model_dir / "config.json"),
        max_tokens=max_tokens, sample_index=sample_index,
        episode_hmac_sha256=episode)
    del manifest_raw
    before_load = time.monotonic()
    model, observed_sha = load_base(
        model_dir, device=device, expected_sha=expected_sha)
    if observed_sha != expected_sha:
        raise AssertionError("weights changed after verification")
    reference_weight = model.model.layers[-1].self_attn.o_proj.weight.detach().clone()
    witness = None
    try:
        cache = cache_base_sample(model, sample, device, keep_logits=False)
        a, b = init_lora(cache, rank=4, seed=seed + 17)
        # Deliberately match initial rank-4 B=0 for the unbiased base
        # observation; no calibration-B change or hidden optimizer step.
        a_original, b_original = a.clone(), b.clone()
        receiver = (SSHReceiver("x1-370", receiver_run_id)
                    if receiver_run_id else None)
        witness = LocalProbeWitness(
            events, derived, episode_hmac_sha256=episode,
            model_revision_sha256=observed_sha, sigma=sigma,
            receiver=receiver,
            causal_history_population=(population if collect_causal_history_v2
                                       else None))
        start = time.monotonic()
        result = tail_scored_structured_estimate(
            model, cache, a, b, seed=seed, population=population,
            sigma=sigma, direction_batch=direction_batch,
            probe_observer=witness)
        observed = witness.finish(population)
        seconds = time.monotonic() - start
        if result["tail_forward_calls"] != 1 + population // direction_batch:
            raise AssertionError("observer changed population/forward calls")
        if not torch.equal(a, a_original) or not torch.equal(b, b_original):
            raise AssertionError("observer changed LoRA parameters")
        if not torch.equal(model.model.layers[-1].self_attn.o_proj.weight,
                           reference_weight):
            raise AssertionError("frozen base weight changed")
        if any(param.grad is not None for param in model.parameters()):
            raise AssertionError("model gradients populated")
        return {
            "schema": "auto-finetune.dust-k2-observation-only.v1",
            "mode": "PRODUCER_LOCAL_WITNESS_ONLY",
            "classifier_training_authorized": False,
            "production_dispatch_authorized": False,
            "model_revision_sha256": observed_sha,
            "train_dataset_digest_sha256": digest(train_file),
            "model_config_sha256": digest(model_dir / "config.json"),
            "sample_selection_pseudonymous": True,
            "source_group_schema": "masked-prompt-prefix-v2",
            "causal_history_v2_observed": (
                observed["causal_history_pre_events_only"]),
            "causal_history_v2_training_authorized": False,
            "preflight_manifest_sha256": expected_preflight_sha256,
            "preflight_group_split": selected_group["group_split"],
            "preflight_cluster_size": selected_group["cluster_members"],
            "near_duplicate_clustering_scope": "single pinned corpus only",
            "raw_source_identifiers_in_evidence": False,
            "source_episode_hmac_sha256": episode,
            "population": population, "sigma": sigma, "seed": seed,
            "direction_batch": direction_batch, "supervised_tokens":
                result["tokens"],
            "tail_forward_calls": result["tail_forward_calls"],
            "duration_seconds": seconds,
            "elapsed_including_model_load_seconds":
                time.monotonic() - before_load,
            "derived_evidence": observed,
            "original_estimator_unchanged": True,
            "base_weights_unchanged": True,
            "adapter_weights_unchanged": True,
            "backward_calls": 0, "optimizer_updates": 0,
            "checkpoint_written": False,
            "real_pretrained_K2_loaded": True,
            "training_data_contained_in_report": False,
            "independent_time_order_attestation": False,
            "independent_receiver_precommit_received":
                observed["receiver_precommit_receipts"] ==
                population // direction_batch if receiver_run_id else False,
            "independent_receiver_run_id": receiver_run_id,
            "independent_receipt_join_verified": False,
            "warning": "Pre/Post fsync and producer-local hash chain are "
                       "not an independent custody witness. Do not train "
                       "from these records until outside attestation, "
                       "privacy review and source-disjoint splits are proven.",
        }
    finally:
        if witness is not None:
            witness.close()
        del model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observe-only", action="store_true")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--train-jsonl", required=True, type=Path)
    ap.add_argument("--episode-key-file", required=True, type=Path)
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--derived", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--expected-sha256", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--population", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=.25)
    ap.add_argument("--direction-batch", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--collect-causal-history-v2", action="store_true",
                    help="opt in to history PRE event fields, no trainer changes")
    ap.add_argument("--receiver-run-id", help="hex32 run ID for x1 HMAC receipt")
    ap.add_argument("--preflight-manifest", required=True, type=Path,
                    help="producer-local, exact near-duplicate preflight JSON")
    ap.add_argument("--expected-preflight-sha256", required=True,
                    help="pinned SHA256 of that cohort preflight")
    args = ap.parse_args(argv)
    if not args.observe_only:
        ap.error("explicit --observe-only is required")
    for path in (args.events, args.derived, args.output):
        if path.exists() or path.is_symlink():
            ap.error("refuse to overwrite any prior research evidence")
    # Preserve HMAC custody and summary permissions even with permissive shell umask.
    os.umask(0o077)
    report = run(
        model_dir=args.model_dir, train_file=args.train_jsonl,
        expected_sha=args.expected_sha256,
        episode_key_file=args.episode_key_file,
        events=args.events, derived=args.derived,
        sample_index=args.sample_index, seed=args.seed,
        population=args.population, sigma=args.sigma,
        direction_batch=args.direction_batch,
        max_tokens=args.max_tokens, device=args.device,
        receiver_run_id=args.receiver_run_id,
        preflight_manifest=args.preflight_manifest,
        expected_preflight_sha256=args.expected_preflight_sha256,
        collect_causal_history_v2=args.collect_causal_history_v2)
    with args.output.open("x", encoding="utf-8") as target:
        target.write(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
