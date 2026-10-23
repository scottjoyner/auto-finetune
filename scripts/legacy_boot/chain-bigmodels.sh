#!/bin/bash

# Fail closed BEFORE scratch, GPU probing, model loading or other side effects.
if ! PYTHONPATH=/home/scott/git/auto-finetune \
    /media/scott/data/finetune-venv/bin/python -m src.training_admission; then
  echo "[training-hold] xwing fine-tuning/inference admission HOLD." >&2
  exit 78
fi

# Wait for Thinking SFT to finish, then relaunch launch-next so the NEW queue
# entries (qwen3-8b, ornith15-9b) are picked up; post-queue waits behind them.
STATE=/media/scott/data/finetune-staging/launch-next.state
while ! grep -qx "done-lfm25-sft-think-r1" "$STATE" 2>/dev/null; do sleep 120; done
sleep 30   # let the old loop exit cleanly
cd /home/scott/git/auto-finetune
export PATH=/media/scott/data/finetune-venv/bin:$PATH PYTHONPATH=/home/scott/git/auto-finetune
export HF_HOME=/media/scott/data/finetune-staging/hf-home TMPDIR=/media/scott/data/finetune-staging/tmp
# Fail closed BEFORE scratch, GPU probing, model loading or other side effects.
if ! PYTHONPATH=/home/scott/git/auto-finetune \
    /media/scott/data/finetune-venv/bin/python -m src.training_admission; then
  echo "[training-hold] xwing fine-tuning/inference admission HOLD." >&2
  exit 78
fi

exec setsid ./launch-next.sh --loop >> /media/scott/data/finetune-staging/logs/launch-next.log 2>&1
