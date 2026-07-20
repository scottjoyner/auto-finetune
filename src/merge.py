"""Merge a finetuned LoRA adapter into the base model for standalone use.

PEFT's merge_and_unload() folds the LoRA weights into the base model so the
result is a plain HF model (no adapter needed at inference). This is the
deployment step: take the best bucket adapter (per `best --metric=...`) and
produce a merged model under outputs/checkpoints/.

NOTE: merging multiple *independently-trained* adapters by simple stacking is
not mathematically clean (LoRA deltas aren't additive across different data).
For combining sources, prefer training `train.combined.jsonl` instead. This
module merges ONE chosen adapter into base.
"""
from __future__ import annotations

import os


def merge_adapter(
    adapter_path: str,
    base_model: str,
    out_dir: str,
    rocm: bool = False,
) -> str:
    """Merge `adapter_path` (LoRA) into `base_model` and save a standalone model.

    Returns the output dir path. Uses bf16 on ROCm, fp16 otherwise.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(out_dir, exist_ok=True)
    dtype = torch.bfloat16 if rocm else torch.float16

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, attn_implementation="sdpa", device_map=None
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(base_model)

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[merge] merged adapter {adapter_path} -> {out_dir}")
    return out_dir
