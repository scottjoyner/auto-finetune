"""DPO stage for tool-caller self-correction (CPU-safe prep; GPU train).

Consumes the mined repair pairs (``repairs.dpo.jsonl`` from
``src.repair_mix.build_dpo_mix``) and trains a preference
model with trl's ``DPOTrainer``, so the tool-caller learns to
*prefer* the corrected call over the errored one.

The data prep (``load_dpo_pairs`` / ``load_dpo_dataset``) is
pure-Python + a tokenizer and is CPU-testable; the actual
train step needs a GPU and is only reached when ``dry_run``
is False.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_dpo_pairs(path: str | Path) -> list[dict]:
    """Parse DPO pairs into (prompt, chosen, rejected) message triples.

    Each source row has ``prompt`` (list of messages) and ``chosen`` /
    ``rejected`` (each a single assistant message carrying a tool_call).
    Returns a list of ``{"prompt", "chosen", "rejected"}`` dicts ready
    for ``load_dpo_dataset`` / trl's ``DPOTrainer``.
    """
    out: list[dict] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.append({
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        })
    return out


def _render(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=False)


def load_dpo_dataset(tok, pairs: list[dict]):
    """Render pairs into a trl-ready Dataset of string columns.

    ``prompt`` / ``chosen`` / ``rejected`` are full conversations
    rendered with the model's chat template (so they are plain strings,
    which sidesteps pyarrow's refusal to infer a schema for
    heterogeneous nested message lists). This is trl's canonical
    text DPO format: ``chosen`` / ``rejected`` are the complete
    trajectories (prompt + the assistant turn), ``prompt`` is the
    shared prefix used for logit masking.
    """
    from datasets import Dataset
    rows = []
    for p in pairs:
        pr = p["prompt"]
        rows.append({
            "prompt": _render(tok, pr),
            "chosen": _render(tok, pr + p["chosen"]),
            "rejected": _render(tok, pr + p["rejected"]),
        })
    return Dataset.from_list(rows)


def train_dpo(cfg, pairs: list[dict], model_name: str,
             output_dir: Optional[str] = None, dry_run: bool = False,
             max_steps: int = 0) -> int:
    """Train a DPO model from ``pairs``. GPU required unless ``dry_run``."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dataset = load_dpo_dataset(tok, pairs)

    if dry_run:
        print(f"[dpo] dry-run: {len(dataset)} pairs tokenized via "
              f"{tok.__class__.__name__}; skipping model load")
        return 0

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto")
    peft_cfg = LoraConfig(
        r=cfg.get("train", "lora_r", default=16),
        lora_alpha=cfg.get("train", "lora_alpha", default=32),
        lora_dropout=cfg.get("train", "lora_dropout", default=0.05),
        target_modules=cfg.get("train", "lora_targets",
                            default=["q_proj", "v_proj"]),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    out = output_dir or cfg.get("train", "output_dir",
                                   default="outputs/checkpoints/dpo")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # trl builds a frozen copy of the base
        args=_dpo_args(cfg, out, max_steps),
        train_dataset=dataset,
        tokenizer=tok,
    )
    trainer.train()
    trainer.save_model(out)
    return 0


def _dpo_args(cfg, output_dir: str, max_steps: int):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.get(
            "train", "per_device_train_batch_size", default=4),
        gradient_accumulation_steps=cfg.get(
            "train", "gradient_accumulation_steps", default=8),
        learning_rate=cfg.get("train", "learning_rate", default=1e-4),
        num_train_epochs=cfg.get("train", "num_train_epochs", default=1),
        max_length=cfg.get("train", "max_seq_length", default=8192),
        warmup_ratio=cfg.get("train", "warmup_ratio", default=0.03),
        logging_steps=1,
        save_strategy="no" if max_steps else "epoch",
        max_steps=max_steps or None,
        report_to="none",
    )
