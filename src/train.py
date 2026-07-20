"""Finetuning with backend auto-selection.

Two backends are supported so the same pipeline runs on CUDA and AMD/ROCm:

- ``unsloth``  (CUDA only): fastest path. Needs ``pip install unsloth`` on a
  CUDA machine. Auto-selected when a CUDA GPU + unsloth are present.
- ``peft``     (CUDA + ROCm/AMD): HuggingFace PEFT + Transformers ``SFTTrainer``
  with 4-bit quantization via bitsandbytes (CUDA) or GPTQ/AWQ (ROCm). This is
  the path for AMD GPUs.

Backend is chosen by ``train.backend`` in config.yaml: ``auto`` picks unsloth
when available, else peft.
"""
from __future__ import annotations

import json
import os
import sys

from src.config import Config


def _detect_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _detect_rocm(rocm_path: str | None = None, hip: str | None = None) -> bool:
    # ROCm exposes HSA/ROCM_PATH or the gfx target; torch may also report a ROCm
    # build. Best-effort detection. `rocm_path`/`hip` are injectable for tests.
    if rocm_path is not None:
        return bool(rocm_path)
    if os.environ.get("ROCM_PATH") or os.environ.get("HSA_PATH"):
        return True
    try:
        import torch
        if hasattr(torch.version, "hip") and (hip if hip is not None else torch.version.hip) is not None:
            return True
    except Exception:
        pass
    return False


class TrainError(RuntimeError):
    """Raised for user-facing training failures (missing dataset, etc.)."""


def _resolve_backend(cfg: Config) -> str:
    requested = (cfg.get("train", "backend", default="auto") or "auto").lower()
    if requested in ("unsloth", "peft"):
        return requested
    # auto: prefer unsloth on CUDA
    if _detect_cuda():
        try:
            import unsloth  # noqa: F401
            return "unsloth"
        except Exception:
            pass
    return "peft"


def load_dataset(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise TrainError(f"invalid JSON on line {ln} of {path}: {e}")
    return rows


def _to_messages(example: dict) -> list[dict]:
    if example.get("conversations"):
        # sharegpt -> chatml-style list
        return [{"role": ("assistant" if m["from"] == "gpt" else
                          ("system" if m["from"] == "system" else "user")),
                 "content": m["value"]} for m in example["conversations"]]
    if example.get("instruction") is not None:  # alpaca
        return [{"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["output"]}]
    return example.get("messages", [])


def _build_texts(dataset: list[dict], tokenizer, max_seq: int) -> list[str]:
    texts = []
    for ex in dataset:
        msgs = _to_messages(ex)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return texts


# ── Unsloth (CUDA) backend ────────────────────────────────────────────────────
def _train_unsloth(cfg: Config, data: list[dict]) -> int:
    from datasets import Dataset
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    t = cfg.get("train", default={})
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=t.get("model_name", "unsloth/Qwen3-8B-Instruct"),
        max_seq_length=t.get("max_seq_length", 8192),
        load_in_4bit=t.get("load_in_4bit", True),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=t.get("lora_r", 32),
        lora_alpha=t.get("lora_alpha", 64),
        lora_dropout=t.get("lora_dropout", 0.0),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    texts = _build_texts(data, tokenizer, t.get("max_seq_length", 8192))
    ds = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        args=_training_args(t, model, rocm=_detect_rocm()),
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    _save(model, tokenizer, t)
    return 0


# ── PEFT (CUDA + ROCm) backend ────────────────────────────────────────────────
def _train_peft(cfg: Config, data: list[dict]) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer

    t = cfg.get("train", default={})
    model_name = t.get("model_name", "Qwen/Qwen3-8B-Instruct")
    max_seq = t.get("max_seq_length", 8192)
    load_4bit = t.get("load_in_4bit", True)
    grad_ckpt = t.get("gradient_checkpointing", False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if load_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if _detect_rocm() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if _detect_rocm() else torch.float16,
        attn_implementation="sdpa",
        quantization_config=quant_config,
        # device_map="auto" wedges the ROCm runtime on this gfx1151 iGPU after a
        # long run (core dump at from_pretrained). Single-GPU so None is equivalent.
        device_map=None,
    )
    if load_4bit:
        model = prepare_model_for_kbit_training(model)

    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora = LoraConfig(
        r=t.get("lora_r", 32),
        lora_alpha=t.get("lora_alpha", 64),
        lora_dropout=t.get("lora_dropout", 0.0),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    texts = _build_texts(data, tokenizer, max_seq)
    ds = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        args=_training_args(t, model, rocm=_detect_rocm()),
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    _save(model, tokenizer, t)
    return 0


def _training_args(t: dict, model=None, rocm: bool = False) -> "TrainingArguments":  # noqa: F821
    from transformers import TrainingArguments
    bf16 = rocm  # AMD/ROCm prefers bf16
    # adamw_8bit requires bitsandbytes which may not have ROCm binaries —
    # fall back to the standard adamw when bnb is unavailable.
    use_bnb_optim = not rocm
    return TrainingArguments(
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        num_train_epochs=t.get("num_train_epochs", 3),
        learning_rate=float(t.get("learning_rate", 2e-4)),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.01),
        lr_scheduler_type="cosine",
        optim="adamw_torch" if not use_bnb_optim else "adamw_8bit",
        fp16=(not bf16),
        bf16=bf16,
        logging_steps=1,
        output_dir=_output_dir(t),
        # Honor config so long runs are observable (gotcha: a shared
        # save_strategy="epoch" left the output dir empty for ~8h before).
        save_strategy=t.get("save_strategy", "epoch"),
        save_steps=t.get("save_steps", 500),
        report_to="none",
    )


def _save(model, tokenizer, t: dict) -> None:
    out_dir = _output_dir(t)
    os.makedirs(out_dir, exist_ok=True)
    # PEFT models expose save_pretrained; unwrap if needed.
    getattr(model, "save_pretrained", lambda p: None)(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[train] saved adapter to {out_dir}")
    if t.get("push_to_hub", False) and t.get("hub_model_id"):
        model.push_to_hub(t["hub_model_id"])
        tokenizer.push_to_hub(t["hub_model_id"])
        print(f"[train] pushed to {t['hub_model_id']}")


def validate_dataset(data_path: str) -> list[dict]:
    """Load and sanity-check the training dataset; raise TrainError if bad."""
    if not os.path.exists(data_path):
        raise TrainError(
            f"dataset not found: {data_path}. Run extract/clean/format first."
        )
    data = load_dataset(data_path)
    if not data:
        raise TrainError(f"dataset is empty: {data_path}")
    return data


def _render_sample_texts(data, tokenizer, max_seq, n: int = 3) -> list[str]:
    """Render the first `n` examples to plain text (used by dry-run)."""
    sample = data[:n]
    return _build_texts(sample, tokenizer, max_seq)


def _output_dir(t: dict) -> str:
    """Resolve the checkpoint output dir, honoring TRAIN_OUTPUT_DIR override."""
    env = os.environ.get("TRAIN_OUTPUT_DIR")
    if env:
        return env
    return t.get("output_dir", "outputs/checkpoints")


def main(cfg: Config, dry_run: bool = False, source: str | None = None, label: str | None = None, max_examples: int | None = None) -> int:
    dataset_dir = os.environ.get("TRAIN_DATASET_DIR") or cfg.path("dataset_dir")
    fn_parts = ["train"]
    if label:
        fn_parts.append(label)
    if source:
        fn_parts.append(source)
    data_path = os.path.join(dataset_dir, ".".join(fn_parts) + ".jsonl")
    data = validate_dataset(data_path)
    if max_examples is not None and max_examples > 0:
        data = data[:max_examples]
        print(f"[train] --max-examples={max_examples}: using first {len(data)} examples")
    print(f"[train] {len(data)} examples loaded from {data_path}")

    backend = _resolve_backend(cfg)
    print(f"[train] backend = {backend}  (cuda={_detect_cuda()}, rocm={_detect_rocm()})")

    if dry_run:
        # Validate the full pipeline up to (but excluding) the GPU model load:
        # load a tiny tokenizer and confirm text rendering works on CPU.
        print("[train] dry-run: validating dataset + tokenization only")
        try:
            from transformers import AutoTokenizer
            tok_name = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
            tok = AutoTokenizer.from_pretrained(tok_name)
            sample = _render_sample_texts(
                data, tok, cfg.get("train", "max_seq_length", default=8192)
            )
            print(f"[train] rendered {len(sample)} sample texts; first len={len(sample[0])} chars")
            print("[train] dry-run OK — install a GPU/ROCm torch build to actually train")
        except Exception as e:
            print(f"[train] dry-run tokenizer check failed: {e}")
        return 0

    if backend == "unsloth":
        try:
            return _train_unsloth(cfg, data)
        except ImportError as e:
            print(f"[train] unsloth unavailable ({e}); falling back to peft")
            return _train_peft(cfg, data)
    return _train_peft(cfg, data)


if __name__ == "__main__":
    from cli import main as _cli_main  # pragma: no cover
    sys.exit(_cli_main(sys.argv))
