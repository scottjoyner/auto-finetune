#!/usr/bin/env bash
# post-reboot-gpu-train.sh — runs on @reboot (via user crontab).
# Verifies the ROCm GPU is alive after reboot; if healthy, auto-launches the
# finetune run. Override TRAIN_ARGS to switch datasets (default: Hermes).
# All output is logged to /media/scott/data/finetune-staging/logs/.
set -u
V=/media/scott/data/finetune-venv/bin
export PATH="$V:$PATH"
export PYTHONPATH=/home/scott/git/auto-finetune
TRAIN_ARGS=${TRAIN_ARGS:---source=hermes}
TRAIN_TAG=${TRAIN_TAG:-hermes}

# Keep ALL scratch off the NFS mount and the small root: redirect HF/transformers
# and torch temp dirs to the local data drive.
export HF_HOME=/media/scott/data/finetune-staging/hf-home
export TRANSFORMERS_CACHE=/media/scott/data/finetune-staging/hf-home
export HF_DATASETS_CACHE=/media/scott/data/finetune-staging/hf-home/datasets
export TORCH_HOME=/media/scott/data/finetune-staging/torch-home
export TMPDIR=/media/scott/data/finetune-staging/tmp
export TEMP=/media/scott/data/finetune-staging/tmp
export TMP=/media/scott/data/finetune-staging/tmp
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"

LOGD=/media/scott/data/finetune-staging/logs
mkdir -p "$LOGD"
LOG="${LOG:-$LOGD/post-reboot-$(date +%Y%m%d-%H%M%S).log}"
exec >"$LOG" 2>&1

echo "[$(date)] post-reboot GPU check starting (kernel $(uname -r))"

# 1) GPU health: allocate on device + H2D copy + matmul.
$V/python - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda_ok", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("NO GPU")
print("device", torch.cuda.get_device_name(0))
x = torch.randn(1024,1024,dtype=torch.bfloat16,device="cpu")
y = x.to("cuda")
print("H2D ok", y.device, (y@y).shape)
PY
if [ $? -ne 0 ]; then
    echo "[$(date)] GPU NOT healthy — aborting training launch. See $LOG"
    exit 1
fi
echo "[$(date)] GPU healthy."

# 2) Dry-run sanity (dataset + tokenizer).
cd /home/scott/git/auto-finetune
$V/python -m src.cli train $TRAIN_ARGS --dry-run 2>&1 | tail -5

# 3) Launch the real run (fresh local output dir).
# guard: don't launch if a train process is already running
if pgrep -f "src.cli train" >/dev/null 2>&1; then
    echo "[$(date)] train already running — skipping launch."
    exit 0
fi

echo "[$(date)] launching finetune with args: $TRAIN_ARGS"
nohup $V/python -m src.cli train $TRAIN_ARGS \
    > "$LOGD/train-${TRAIN_TAG}.log" 2>&1 &
echo "[$(date)] launched PID $! — tail $LOGD/train-${TRAIN_TAG}.log"
