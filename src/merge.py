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
import socket

from src import training_admission as admission
from src.training_admission import MIN_KIB

# Conservative floor for folding a bf16 base model into a standalone checkpoint.
# Must stay at or above the 16 GiB training-admission memory gate (MIN_KIB in
# src/training_admission.py): a merge loads the full base model in bf16 plus PEFT
# buffers, so it needs the same order of headroom as the training run it follows.
MIN_MERGE_AVAILABLE_KIB = MIN_KIB


class MergePreflightError(RuntimeError):
    """Host cannot safely hold the merge; raised before any model import/load."""


def preflight_merge() -> None:
    """Refuse a merge when the host cannot safely load the base model.

    Mirrors the training admission memory gate (MemAvailable from /proc/meminfo)
    and adds the resident-LLM check: merging loads the full base model, so an
    LLM already resident for inference would put the base load at risk. Never
    imports torch; non-xwing hosts keep the memory guard and skip the
    xwing-only inventory. Fails closed on unreadable telemetry.
    """
    try:
        available = admission.read_mem_available_kib()
    except (OSError, ValueError, TypeError) as exc:
        raise MergePreflightError(
            f"[merge] refused: cannot read MemAvailable from /proc/meminfo ({exc})"
        ) from exc
    if not isinstance(available, int) or available < MIN_MERGE_AVAILABLE_KIB:
        shown = f"{available // 1024} MiB" if isinstance(available, int) else "unreadable"
        raise MergePreflightError(
            f"[merge] refused: MemAvailable {shown} is below the safe minimum "
            f"{MIN_MERGE_AVAILABLE_KIB // 1024} MiB needed to load the base "
            "model in bf16 (free RAM or unload other jobs before merging)"
        )

    if socket.gethostname().lower().split(".")[0] != "xwing":
        return

    try:
        models = admission.read_model_inventory()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise MergePreflightError(
            f"[merge] refused: resident-model inventory unavailable ({exc}); "
            "cannot verify memory headroom on xwing"
        ) from exc
    problem = admission.inventory_error(models)
    if problem:
        raise MergePreflightError(
            f"[merge] refused: {problem}; cannot verify resident inference "
            "before loading the base model"
        )
    if admission.resident_llm(models):
        raise MergePreflightError(
            "[merge] refused: a resident LLM is loaded for inference; unload "
            "it before merging or the base model load may OOM"
        )


def merge_adapter(
    adapter_path: str,
    base_model: str,
    out_dir: str,
    rocm: bool = False,
) -> str:
    """Merge `adapter_path` (LoRA) into `base_model` and save a standalone model.

    Returns the output dir path. Uses bf16 on ROCm, fp16 otherwise.
    """
    # Must stay ahead of every heavy import and filesystem side effect.
    preflight_merge()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.lfm3_support import is_lfm3_model, patch_transformers_for_lfm3

    _lfm3 = is_lfm3_model(base_model)
    if _lfm3:
        patch_transformers_for_lfm3()

    os.makedirs(out_dir, exist_ok=True)
    dtype = torch.bfloat16 if rocm else torch.float16

    # trust_remote_code matters in a batch context: bases that ship custom
    # modeling files (e.g. K2-Horizon's modeling_k2_horizon.py) otherwise raise
    # and, without it, prompt "Do you wish to run the custom code? [y/N]" on
    # stdin -- which hangs the post-training merge hook with no terminal attached.
    # MERGE_DEVICE=cpu selects the CPU merge path (device_map=None); the default
    # for LFM3 is device_map="auto" (GPU placement for faster merging). The two
    # must not diverge: lock/serve treat MERGE_DEVICE=cpu as a CPU merge.
    merge_device = (os.environ.get("MERGE_DEVICE") or "").strip().lower()
    device_map = None if merge_device == "cpu" else ("auto" if _lfm3 else None)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, attn_implementation="sdpa",
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[merge] merged adapter {adapter_path} -> {out_dir}")
    return out_dir
