#!/usr/bin/env bash
# post-reboot-gpu-train.sh — runs on @reboot (via user crontab).
# Verifies the ROCm GPU is alive after reboot; if healthy, auto-launches the
# Hermes finetune run. All output is logged to /media/scott/data/finetune-staging/logs/.
set -u
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=11.5.1
V=/media/scott/data/finetune-venv/bin
export PATH="$V:$PATH"
export PYTHONPATH=/home/scott/git/auto-finetune

LOGD=/media/scott/data/finetune-staging/logs
mkdir -p "$LOGD"
LOG="$LOGD/post-reboot-$(date +%Y%m%d-%H%M%S).log"
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
$V/python -m src.cli train --source=hermes --dry-run 2>&1 | tail -5

# 3) Launch the real run (fresh local output dir).
echo "[$(date)] launching Hermes finetune..."
nohup $V/python -m src.cli train --source=hermes \
    > "$LOGD/train-hermes-full.log" 2>&1 &
echo "[$(date)] launched PID $! — tail $LOGD/train-hermes-full.log"
