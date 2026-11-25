# LFM3-1.7B training runbook (xwing)

Everything below is ready; only the operator gates (P0 hold, GPU, venv) block execution.

## Pre-flight (must all pass)
1. Recovery latch promoted via reviewed policy (see docs/XWING_QWEN_DUAL_RECOVERY_HOLD_20261024.md).
   Training admission must return `PASS` (not `HOLD`).
2. GPU healthy: `rocm-smi` shows card 0, and a 256x256 bf16 matmul succeeds:
   ```bash
   PYTHONPATH=... /media/scott/data/venvs/lfm3-disposable/bin/python -c \
     "import torch; x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16); print((x@x).shape)"
   ```
   If it fails with `hipErrorInvalidImage`, terminate wedged pids (`rocm-smi --showpids`) and reboot.
3. MemAvailable >= 16 GiB (unload Ornith / wait for low-inference window).
4. No resident LLM on 127.0.0.1:1234 (`loaded_instances` empty for llm models).
5. Disposable venv present: /media/scott/data/venvs/lfm3-disposable (Python 3.12.13).

## Run
```bash
cd /home/scott/git/auto-finetune
export TRAIN_BIN=/media/scott/data/venvs/lfm3-disposable/bin/python
export PYTHONPATH=/home/scott/git/auto-finetune:/media/scott/data/finetune-venv/lib/python3.12/site-packages:/home/scott/.local/lib/python3.12/site-packages
# native dialect (preferred):
TRAIN_DATASET_DIR=/media/scott/data/finetune-staging/data/future-runs/lfm3-mixed-seed42/datasets \
  $TRAIN_BIN -u -m src.cli train --label=lfm3-mixed
# or queued:
./launch-next.sh   # LFM3 entries are in the queue
```

## Verify
- Dry-run: `... src.cli train --label=lfm3-mixed --dry-run` -> "holdout verify: clean".
- Checkpoints land in /media/scott/data/finetune-staging/outputs/checkpoints/lfm3-1.7b-sft-r1/.
- Post-run: `src.cli merge` (MERGE=1 in queue) writes ...-master.

## Constraints honored
No SSD_4TB/models changes, no lms load/unload, no lm-studio restarts, no disk-reclaim,
no edits to harvest-opencode-fleet.sh or assistx-projection-keepalive.py.
EOF
echo done