# Skill: Auto-Finetune Pipeline

Training, evaluation, benchmarking, merging, and deployment for the auto-finetune system. Training requires a GPU (CUDA or AMD/ROCm).

## Environment Setup

```bash
export PATH="/media/scott/data/finetune-venv/bin:$PATH"
export PYTHONPATH=/home/scott/git/auto-finetune
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=11.5.1  # AMD gfx1151 only
export HF_HOME=/media/scott/data/finetune-staging/hf-home
export TMPDIR=/media/scott/data/finetune-staging/tmp
cd /home/scott/git/auto-finetune
```

## Hardware & GPU

- **GPU**: AMD Radeon 8050S (gfx1151, Strix Point), 12 GB VRAM, unified RAM
- **ROCm**: `/opt/rocm` -> 7.0.0
- **torch**: 2.12.0+rocm7.14.0
- **Speed**: ~45-48 s/step (shared-memory-bandwidth bound)

### GPU Recovery After Reboot

```bash
# kfd is built into amdgpu -- just load amdgpu:
sudo modprobe amdgpu

# verify:
ls /dev/kfd  # should exist
bash /media/scott/data/finetune-staging/post-reboot-check.sh
```

**CRITICAL**: Keep kernel `7.0.0-28-generic`. Do NOT boot `6.17.0-1028-oem` (breaks gfx1151 dispatch).

## Storage Layout

| Type | Path |
|------|------|
| Datasets | `/media/scott/data/finetune-staging/data/datasets/` |
| Checkpoints | `/media/scott/data/finetune-staging/outputs/checkpoints/` |
| Logs | `/media/scott/data/finetune-staging/logs/` |
| Eval reports | `/media/scott/data/finetune-staging/data/eval-reports/` |

## Training Queue

The `launch-next.sh` script runs datasets in order:

| # | Label | Dataset | Status |
|---|-------|---------|--------|
| 1 | ssd | opencode live sessions | Done |
| 2 | nas5-main | NAS5 main snapshot | Done |
| 3 | nas5-20260717 | NAS5 July 17 snapshot | Done |
| 4 | opencode-all | Merged opencode sources | Done |
| 5 | opencode-portfolio | Portfolio-filtered opencode | Done |
| 6 | hermes-reasoning | Hermes agent sessions | Running |
| 7 | combined | Deduped union of all | Pending |

State file: `/media/scott/data/finetune-staging/launch-next.state`

## Training Commands

### Single Training Run

```bash
python -m src.cli train --label=combined         # Train on specific label
python -m src.cli train --source=hermes           # Train on hermes source
python -m src.cli train --label=ssd --max-examples=100  # Quick test
python -m src.cli train --source=hermes --dry-run       # Tokenizer-only validation
```

### Chained Training Queue

```bash
./launch-next.sh              # Run ONE next dataset, then exit
./launch-next.sh --loop       # Keep launching until queue empty
./launch-next.sh --reset      # Forget completion state (re-run everything)
```

### Launch with Setsid (Survives Session End)

```bash
setsid nohup bash launch-next.sh --loop > /media/scott/data/finetune-staging/logs/launch-next-$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

### Monitor Training

```bash
tail -f /media/scott/data/finetune-staging/logs/train-toolcall-v5-3b-<label>.log
ps aux | grep "src.cli train" | grep -v grep
```

## Training Config (config.yaml)

```yaml
train:
  backend: peft              # peft (ROCm/CUDA) | unsloth (CUDA only) | auto
  model_name: /media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  max_seq_length: 8192
  gradient_checkpointing: true
  load_in_4bit: false        # REQUIRED false on ROCm (no bitsandbytes binary)
  learning_rate: 1e-4
  num_train_epochs: 2
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_ratio: 0.03
  save_strategy: steps
  save_steps: 50
  output_dir: /media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-<label>
```

### Backend Selection

| Backend | GPU | Speed | Notes |
|---------|-----|-------|-------|
| `peft` | CUDA + ROCm | Moderate | HuggingFace PEFT + bitsandbytes (CUDA) or auto-gptq (ROCm) |
| `unsloth` | CUDA only | Fastest | Requires `pip install unsloth` |
| `auto` | Either | Auto | Uses unsloth if CUDA + unsloth present, else peft |

## Evaluation Commands

### Held-Out Loss Evaluation

```bash
python -m src.cli eval-all                       # Evaluate all adapters
python -m src.cli eval-all --loss-only           # Loss only (safe during training)
python -m src.cli eval-all --report              # Write report to disk
python -m src.cli eval --label=combined          # Evaluate specific adapter
```

### Pick Best Adapter

```bash
python -m src.cli best --metric=loss
python -m src.cli best --metric=tool_exact
```

### Qualitative Probe

```bash
python -m src.cli probe --label=combined         # Base vs adapter comparison
python -m src.cli compare                        # All adapters vs base
python -m src.cli compare --label=combined       # Scoped to one bucket
```

### Sanity Check

```bash
python -m src.cli sanity                         # Check all adapter directories
```

### Consolidated Report

```bash
python -m src.cli report --label=combined        # Loss + probe + best -> report file
```

## Benchmarking Commands

### Single Model Benchmark

```bash
python -m src.cli bench --runner=self --model=/path/to/ckpt --tasks=eval/tasks/auto-verified.jsonl
python -m src.cli bench --runner=subagent --model=/path/to/ckpt
python -m src.cli bench --runner=api --fleet --tasks=eval/tasks/auto-verified.jsonl
```

### Benchmark Matrix

```bash
python -m src.cli bench-matrix --preset=all --report    # Full comparison
python -m src.cli bench-matrix --preset=fast            # Quick smoke gate
python -m src.cli bench-matrix --preset=local-refs      # Local only
python -m src.cli bench-matrix --preset=lmstudio        # GGUF models
python -m src.cli bench-matrix --preset=fleet           # LAN fleet models
```

### Bench-Compare (Base vs Adapter)

```bash
python -m src.cli bench-compare \
  --base=/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b \
  --adapter=/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-<label> \
  --tasks=eval/tasks/auto-verified.jsonl
```

### Build Benchmark Tasks

```bash
python -m src.cli bench-build \
  --tasks=analysis/auto-tasks.jsonl \
  --verify-report=analysis/verify-report.jsonl \
  --out=eval/tasks/auto-verified.jsonl
```

### Five Benchmark Runners

| Runner | Description | Model Source |
|--------|-------------|--------------|
| `self` | Self-contained harness, parses `<tool_call>`, runs tools in sandbox | Local HF dir |
| `subagent` | Optimized loop for RefinedToolCallV5 variants | Local HF dir |
| `local-chat` | Standard HF chat model in native function-call format | Local HF dir |
| `api` | OpenAI-compatible endpoint | `--base-url` + `--api-model` |
| `hermes` | Delegates to hermes-agent harness | Hermes config |

## Merge & Quantize

### Merge LoRA Adapter

```bash
python -m src.cli merge --label=combined
```

- Fuses LoRA into standalone model
- Output: `outputs/checkpoints/toolcall-v5-3b-<label>-merged/`

### Quantize

```bash
python -m src.cli quantize --label=combined --bits=4 --method=gptq
python -m src.cli quantize --label=combined --bits=4 --method=awq
python -m src.cli quantize-status
```

## DPO Training

```bash
python -m src.cli dpo --model=<base-or-sft-checkpoint> --pairs=analysis/repairs.dpo.jsonl
python -m src.cli dpo --model=<checkpoint> --dry-run --max-steps=10
```

## Deployment Commands

### Deploy to Nodes

```bash
python -m src.cli deploy --label=combined --nodes=local,nas5,laptop
python -m src.cli deploy --label=combined --nodes=local,nas5 --quorum=2
```

### Check Deployment Status

```bash
python -m src.cli multi-deploy-status
python -m src.cli deploy-status
python -m src.cli discover-nodes
```

### Rollback

```bash
python -m src.cli rollback --label=combined --nodes=local,nas5
```

## Operations Commands

### Metrics Tracking

```bash
python -m src.cli metrics-record --label=combined --eval-loss=0.42 --tool-exact=0.85
python -m src.cli metrics-compare --label=combined
python -m src.cli metrics-regression --label=combined
python -m src.cli metrics-history --label=combined
python -m src.cli metrics-summary
```

### Cost Tracking

```bash
python -m src.cli cost-record --label=combined --hours=8.5 --gpu="Radeon 890M"
python -m src.cli cost-summary
python -m src.cli cost-history
```

### Notifications

```bash
python -m src.cli notify --event=training_complete --message="combined model trained"
python -m src.cli notify-history --limit=10
```

### Registry

```bash
python -m src.cli registry-list
python -m src.cli registry-add --label=combined --checkpoint=/path/to/model
```

### Scheduler (Continuous Automation)

```bash
python -m src.cli scheduler-status
python -m src.cli scheduler-run                  # One complete cycle
python -m src.cli scheduler-loop --interval=3600 # Continuous monitoring
```

## Gotchas

1. **`load_in_4bit: false` REQUIRED on ROCm.** No `libbitsandbytes_rocm83.so` binary. Setting `true` crashes at model load.
2. **Long silent load (~6 min).** Model loads 434 safetensor shards from NFS. Not a hang.
3. **`save_strategy: steps, save_steps: 50`** ensures checkpoints appear every ~1.5h. `save_strategy: epoch` leaves output dir empty for hours.
4. **Fresh output dir for new runs.** Stale `trainer_state.json` from prior runs causes checkpoint offset bugs.
5. **Speed is shared-memory-bandwidth bound.** ~45-48 s/step on Radeon 8050S. Real speedup needs discrete GPU with HBM.
6. **`torch_dtype` deprecated** in transformers 5.14.1. Use `dtype` instead.
7. **`warmup_ratio` deprecated** in transformers 5.14.1 (still works, just warns).
8. **Keep `7.0.0-28-generic` kernel.** `6.17.0-1028-oem` breaks gfx1151 dispatch.

## Quick Reference: Full Training Pipeline

```bash
# 1. Verify GPU
sudo modprobe amdgpu
bash /media/scott/data/finetune-staging/post-reboot-check.sh

# 2. Smoke test
python -m src.cli train --source=hermes --max-examples=100 --dry-run
python -m src.cli train --source=hermes --max-examples=100

# 3. Launch full queue
setsid nohup bash launch-next.sh --loop > /media/scott/data/finetune-staging/logs/launch-next-$(date +%Y%m%d-%H%M%S).log 2>&1 &

# 4. Monitor
tail -f /media/scott/data/finetune-staging/logs/train-toolcall-v5-3b-<label>.log

# 5. After training completes
python -m src.cli eval-all --loss-only --report
python -m src.cli best --metric=loss
python -m src.cli merge --label=combined
python -m src.cli bench-matrix --preset=fast --report

# 6. Deploy
python -m src.cli deploy --label=combined --nodes=local,nas5
```

## Reboot Recovery

See `REBOOT-RECOVERY.md` for the full procedure. Key steps:

1. Ensure kernel is `7.0.0-28-generic`
2. `sudo modprobe amdgpu` (loads kfd too)
3. Verify GPU: `bash post-reboot-check.sh`
4. Resume queue: `setsid nohup bash launch-next.sh --loop`
