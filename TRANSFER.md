# Transfer manifest — moving the finetune run to a faster machine

This run was started on the local box (Ryzen AI 9 HX 370 / Radeon 890M iGPU,
shared-memory bandwidth-bound, ~115 s/step => ~27 h for the full Hermes set).
It was stopped at step 100/896 with **no usable checkpoint written** (see
"Gotchas" below). Everything needed to reproduce / continue on a faster box
lives on the local NVMe `SSD_4TB` (`/dev/nvme1n1p1`, ext4 — NOT network).

The other machine must copy the items below over, then run from a clean output
dir.

## Locations on this machine (source)

| What | Path | Size | Notes |
|------|------|------|-------|
| **Repo + code + config** | `/home/scott/git/auto-finetune` | ~few MB | git repo; commit the working-tree edits first |
| **Training dataset (Hermes)** | `/media/scott/SSD_4TB/finetune-staging/data/datasets/train.hermes.jsonl` | 149 MB | 7,153 examples; the real asset to carry |
| **All formatted splits** | `/media/scott/SSD_4TB/finetune-staging/data/datasets/` | ~600 MB | `train.jsonl` (merged), `train.opencode.jsonl`, `train.ssd.jsonl`, `train.nas5-*.jsonl`, etc. |
| **Raw extracted sessions** | `/media/scott/SSD_4TB/finetune-staging/data/raw/` | large | per-source subdirs (ssd, nas5-*, hermes) |
| **Cleaned sessions** | `/media/scott/SSD_4TB/finetune-staging/data/cleaned/` | large | per-source subdirs |
| **Base model** | `/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b` | 17 GB | 3B Qwen2, bf16 safetensors |
| **Python venv** | `/media/scott/SSD_4TB/finetune-staging/venv/` | large | ROCm torch 2.10 + peft/trl/datasets; rebuild on the new box instead |
| **Logs** | `/media/scott/SSD_4TB/finetune-staging/logs/` | tiny | `train-hermes-*.log` |
| **Output dir (checkpoints)** | `/media/scott/SSD_4TB/finetune-staging/outputs/checkpoints/toolcall-v5-3b-hermes/` | ~120 MB | only smoke-test adapter present; NOT a real run checkpoint |

## Minimal copy for a fresh run on the new machine

You do NOT need the raw/cleaned dirs or the venv — just regen or rebuild.

```bash
# 1) the code
rsync -aP /home/scott/git/auto-finetune/  <NEW>:/path/auto-finetune/

# 2) the dataset (the only data you strictly need — format is already done)
rsync -aP /media/scott/SSD_4TB/finetune-staging/data/datasets/ \
          <NEW>:/path/finetune-staging/data/datasets/

# 3) the base model (or just let HF download it if it's on the Hub)
rsync -aP /media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b/ \
          <NEW>:/path/models/RefinedToolCallV5-3b/
```

Then on the new machine, point `config.yaml` at the copied paths:
```yaml
train.model_name: /path/models/RefinedToolCallV5-3b
train.output_dir: /path/finetune-staging/outputs/checkpoints/toolcall-v5-3b-hermes
# (paths in config.yaml use absolute SSD paths — update them)
```

The dataset dir paths in `config.yaml` are:
```yaml
raw_dir:     /media/scott/SSD_4TB/finetune-staging/data/raw
cleaned_dir: /media/scott/SSD_4TB/finetune-staging/data/cleaned
dataset_dir: /media/scott/SSD_4TB/finetune-staging/data/datasets
```
If you only copied `datasets/`, you can skip extract/clean/format and go
straight to `train --source=hermes`.

## Launch

```bash
cd /path/auto-finetune
export PATH="/path/venv/bin:$PATH" PYTHONPATH="/path/auto-finetune"
# ROCm only: export ROCM_PATH=/opt/rocm HSA_OVERRIDE_GFX_VERSION=11.5.1
nohup python -m src.cli train --source=hermes \
  > /path/finetune-staging/logs/train-hermes-full.log 2>&1 &
```

Smoke-test first to confirm the pipeline before the long run:
```bash
python -m src.cli train --source=hermes --max-examples=100 --dry-run   # tokenizer only
python -m src.cli train --source=hermes --max-examples=100             # ~real 100-ex run
```

## Gotchas (learned the hard way)

1. **`save_steps` / resume-offset bug.** The full run reached step 100/896 but
   never wrote `checkpoint-50` / `checkpoint-100`. The output dir was shared
   with the smoke test, which left a `trainer_state.json` recording
   `global_step=14`. HF `Trainer` appears to offset its save counter against
   that. **Fix: start the new run in a FRESH, empty output dir** (no leftover
   `trainer_state.json` / `checkpoint-*`). Verify a checkpoint appears at step
   50. The smoke test's `checkpoint-7`/`checkpoint-14` are useless for resume.

2. **`load_in_4bit: false` is REQUIRED here.** `bitsandbytes` has NO ROCm
   binary installed (`libbitsandbytes_rocm83.so` missing); setting it `true`
   crashes at model load with `RuntimeError: Configured ROCm binary not found`.
   On a CUDA box you can flip it `true` (QLoRA) for less VRAM, or switch
   `train.backend: auto` + install `unsloth` for a big speedup.

3. **Long silent load (~6 min).** `device_map="auto"` streams 434 safetensor
   shards from disk into the (i)GPU one at a time. No progress reaches the
   terminal until training starts. This is normal, not a hang.

4. **`save_strategy="epoch"` made the old run look dead.** With 2 epochs and no
   step-saving, the first checkpoint only appeared after ~8 h and the output dir
   stayed empty the whole time — that's what looked like "ran for hours and did
   nothing". Config now uses `save_strategy: steps, save_steps: 50` so a
   checkpoint drops every ~1.5 h.

5. **Speed is a shared-memory-bandwidth wall, not VRAM.** On the 890M iGPU with
   49 GB unified RAM, ~115 s/step. A "beefier iGPU with 8 GB VRAM + 25 GB RAM"
   would be SLOWER or broken (8 GB < the ~6 GB bf16 weights; less system RAM =
   less unified pool). Real speedup needs a discrete GPU with its own HBM
   (10-30x), or fewer epochs / smaller effective batch.

6. **`train.hermes.jsonl` is selected by `--source=hermes`** (cli builds the
   filename `train.<source>.jsonl`). Without `--source`, `train` loads the
   merged `train.jsonl`.
