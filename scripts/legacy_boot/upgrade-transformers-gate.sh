#!/bin/bash

# Fail closed BEFORE scratch, GPU probing, model loading or other side effects.
if ! PYTHONPATH=/home/scott/git/auto-finetune \
    /media/scott/data/finetune-venv/bin/python -m src.training_admission; then
  echo "[training-hold] xwing fine-tuning/inference admission HOLD." >&2
  exit 78
fi

# Fires when qwen3-8b SFT completes: upgrades main venv transformers to 5.x
# (required for ornith's qwen3_5 arch — validated in test-venv-qwen35).
# Runs BEFORE ornith starts; post-queue evals come later and LFM2 support is
# retained in 5.15.1 (validated: ORNITH_OK + LFM2_STILL_PRESENT).
STATE=/media/scott/data/finetune-staging/launch-next.state
LOG=/media/scott/data/finetune-staging/logs/tf-upgrade.log
while ! grep -qx "done-qwen38b-sft-r1" "$STATE" 2>/dev/null; do sleep 120; done
{
  echo "[$(date)] qwen38b done — upgrading transformers"
# Fail closed BEFORE scratch, GPU probing, model loading or other side effects.
if ! PYTHONPATH=/home/scott/git/auto-finetune \
    /media/scott/data/finetune-venv/bin/python -m src.training_admission; then
  echo "[training-hold] xwing fine-tuning/inference admission HOLD." >&2
  exit 78
fi

  /media/scott/data/finetune-venv/bin/pip install -q "transformers==5.15.1"
  TRANSFORMERS_OFFLINE=1 HF_HOME=/media/scott/data/finetune-staging/hf-home \
    /media/scott/data/finetune-venv/bin/python -c "
from transformers import AutoConfig
import transformers
c = AutoConfig.from_pretrained('/media/scott/data/finetune-staging/models/Ornith-1.5-9B')
assert c.model_type == 'qwen3_5'
import transformers.models.lfm2 as l
print('UPGRADE OK', transformers.__version__, '| lfm2 present')"
} >> "$LOG" 2>&1
