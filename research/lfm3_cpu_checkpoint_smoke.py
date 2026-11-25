#!/usr/bin/env python3
"""Opt-in, offline, resource-bounded CPU-only LFM3 checkpoint smoke test.

Never starts training, reads private prompts, makes network requests, or
changes model weights. Intended only for a separately admitted research host.
An external process timeout and resource limits should still be enforced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SCHEMA = "auto-finetune.lfm3.cpu-checkpoint-smoke.v1"


def blocked(reason):
    print(json.dumps({"schema": SCHEMA, "status": "blocked",
                      "reason": reason, "weight_promotion_permitted": False},
                     sort_keys=True))
    return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run", action="store_true",
                        help="explicitly opt in to loading full 3.46 GB BF16 model on CPU")
    args = parser.parse_args(argv)
    if not args.run:
        return blocked("full_checkpoint_load_not_authorized")
    # Cheap path gate is deliberately before any heavyweight ML imports.
    if args.checkpoint.is_symlink() or not args.checkpoint.is_dir():
        return blocked("checkpoint_directory_missing_or_linked")

    # Must precede torch, transformers, and dynamic remote-code module imports.
    # Caller should also isolate this process with external resource quotas.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        os.environ[key] = ""
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        from src.lfm3_support import patch_transformers_for_lfm3
        from src.lfm3_pretrained_guard import (
            load_pretrained_lfm3, verify_official_lfm3_snapshot,
        )
        from src.lfm3_chat_template_guard import ensure_native_lfm3_chat_template

        torch.set_num_threads(2)
        verified = verify_official_lfm3_snapshot(str(args.checkpoint))
        patch_transformers_for_lfm3()
        tok = AutoTokenizer.from_pretrained(verified, trust_remote_code=True,
                                           local_files_only=True)
        ensure_native_lfm3_chat_template(tok, verified)
        model = load_pretrained_lfm3(verified, AutoConfig, AutoModelForCausalLM,
                                     torch.bfloat16)
        if any(param.device.type != "cpu" for param in model.parameters()):
            return blocked("model_resided_outside_cpu")
        model.eval()
        ids = tok("Hello world", return_tensors="pt")["input_ids"]
        if not 1 <= ids.shape[1] <= 8:
            return blocked("unexpected_fixed_test_tokenization")
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits
        if (logits.ndim != 3 or logits.shape[0] != 1 or
                logits.shape[1] != ids.shape[1] or logits.shape[2] != 128000
                or not bool(torch.isfinite(logits).all())):
            return blocked("nonfinite_or_unexpected_logits")
        print(json.dumps({
            "schema": SCHEMA, "status": "cpu_functional_smoke_passed",
            "source_claim": "sha256_pinned_local_snapshot_not_publisher_signature",
            "model_source_signature_verified": False,
            "native_template_digest_verified": True,
            "pretrained_loading_keys_complete": True,
            "input_token_count": int(ids.shape[1]),
            "output_logit_shape": list(logits.shape),
            "finite_logits": True,
            "heldout_quality_verified": False,
            "gpu_execution_verified": False,
            "optimizer_improvement_verified": False,
            "weight_promotion_permitted": False,
        }, sort_keys=True))
        return 0
    except Exception:
        # Errors may contain private filesystem paths, URLs or prompts.
        return blocked("verification_or_runtime_failure")


if __name__ == "__main__":
    raise SystemExit(main())
