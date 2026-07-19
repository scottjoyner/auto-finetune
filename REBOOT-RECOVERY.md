# Reboot recovery — ROCm kernel / GPU dispatch crash (2026-07-19)

## Symptom (before reboot)
After a reboot to bring up the ROCm 7.0 kernel, any GPU work segfaulted in the
ROCm runtime — NOT a code bug.

- `hipSetDevice` + `hipMalloc` succeeded (agent enumerated fine).
- `hipMemcpy` host→device and even a trivial `__global__` kernel launch
  **segfaulted inside `libhsa-runtime64.so.1`** (ROCr queue dispatch).
- Reproduced both via raw HIP (`/tmp/opencode/t4`, `t5`) and via torch
  `torch.randn(...).to("cuda")` → SIGSEGV.
- `rocminfo` / `rocm-smi` worked (sensors read fine), so the device was
  visible but compute dispatch was dead.

## Root cause
The machine was running a **non-default kernel**: `6.17.0-1028-oem`,
while `/etc/default/grub` sets `GRUB_DEFAULT` to
`gnulinux-7.0.0-28-generic-advanced-...` (i.e. `7.0.0-28-generic`).
The `6.17.0-1028-oem` amdgpu driver mismatched ROCm 7.0 for the
`gfx1151` (Ryzen AI MAX / Radeon 8050S) compute engine → broken queue
submission. The known-good kernel is `7.0.0-28-generic`.

## Fix (user action — needs root)
Reboot into the GRUB default (`7.0.0-28-generic`):

```bash
# A) plain reboot uses GRUB_DEFAULT (7.0.0-28-generic):
sudo reboot

# B) if it comes back on 6.17, force the good one then reboot:
sudo grub-reboot "gnulinux-7.0.0-28-generic-advanced-bb13b3dd-737f-4736-af88-a1c6cb20d48f"
sudo reboot
```

After reboot, verify with:
```bash
bash /media/scott/data/finetune-staging/post-reboot-check.sh
```
Expected: `uname -r` → `7.0.0-28-generic`, `H2D ok`, `DONE`.

## Environment facts (so we don't re-derive them)
- GPU: `AMD Radeon 8050S` (gfx1151, Strix Point). 12 GB VRAM, unified RAM.
- ROCm: `/opt/rocm` → `/opt/rocm-7.0.0`, version `7.0.0`.
- Required env (set in check script + launch):
  ```bash
  export ROCM_PATH=/opt/rocm
  export HSA_OVERRIDE_GFX_VERSION=11.5.1   # gfx1151
  ```
- venv: `/media/scott/data/finetune-venv`  (LOCAL `/media/scott/data` drive,
  NOT the NFS SSD_4TB). Built with ROCm torch.
  - **torch must be `2.11.0.dev20260206+rocm7.0`** (the cached wheel at
    `/media/scott/data/pip-tmp/torch-2.11.0.dev20260206+rocm7.0-*.whl`).
    A plain `pip install torch` pulled CUDA torch 2.13.0 and broke the GPU —
    re-pin with `--index-url https://download.pytorch.org/whl/rocm7.0` and
    `--no-deps` if it ever gets clobbered again.
  - deps: peft 0.19.1, transformers 5.14.1, trl 1.8.0, datasets 5.0.0.
- Storage:
  - Local workspace: `/media/scott/data` (1.6T, 604G free) — USE THIS for
    outputs/venv/swap. Has `finetune-venv`, `pip-tmp`, `swap-xwing-500g.img`.
  - NFS network drive: `100.64.43.123:/media/scott/SSD_4TB` mounted at
    `/media/scott/SSD_4TB` (TRANSFER.md says "NOT network" but it IS NFS —
    fine for reading datasets/model, slower for heavy I/O).
  - CIFS: `/media/scott/NAS5` (opencode DBs, neo4j).
- Data asset: `/media/scott/SSD_4TB/finetune-staging/data/datasets/train.hermes.jsonl`
  (149 MB, 7153 examples) — selected by `--source=hermes`.
- Base model: `/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b`.

## Code changes already made (staged, uncommitted)
1. `src/train.py` `_training_args`: now honors `save_strategy` / `save_steps`
   from config instead of hardcoding `save_strategy="epoch"` (was gotcha #4 —
   output dir looked dead for 8h). Defaults: `save_strategy="epoch"`,
   `save_steps=500` if config omits them.
2. `config.yaml` `train.output_dir` →
   `/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-hermes`
   (local disk; was on NFS). Fresh empty dir created (clears gotcha #1 stale
   `trainer_state.json` from the prior smoke test).

## Launch (after GPU verified)
```bash
cd /home/scott/git/auto-finetune
export PATH="/media/scott/data/finetune-venv/bin:$PATH"
export PYTHONPATH=/home/scott/git/auto-finetune
export ROCM_PATH=/opt/rocm HSA_OVERRIDE_GFX_VERSION=11.5.1
nohup python -m src.cli train --source=hermes \
  > /media/scott/data/finetune-staging/logs/train-hermes-full.log 2>&1 &
```
- Model load is SILENT ~6 min (434 shards streamed over NFS). Not a hang.
- `save_steps: 50` → first checkpoint ~1.5 h in. Watch:
  `tail -f /media/scott/data/finetune-staging/logs/train-hermes-full.log`
- Smoke test first if unsure:
  `python -m src.cli train --source=hermes --max-examples=100`

## Gotchas (carried from TRANSFER.md + this incident)
- #1 stale output dir → start fresh (done).
- #2 `load_in_4bit: false` required (no ROCm bitsandbytes binary). Keep false.
- #3 long silent load (~6 min) is normal.
- #4 save_strategy now = steps/50 (fixed in code).
- #5 speed is shared-memory-bandwidth bound; ~65-115 s/step observed.
- #6 `--source=hermes` picks `train.hermes.jsonl`.
- NEW: keep the `7.0.0-28-generic` kernel; do NOT boot `6.17.0-1028-oem`
  for ROCm work (breaks gfx1151 dispatch).
