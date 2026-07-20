#!/usr/bin/env bash
# Post-reboot GPU + finetune verification. Run AFTER rebooting into the
# GRUB-default kernel (7.0.0-28-generic).
set -u
V=/media/scott/data/finetune-venv/bin
export PATH="$V:$PATH"
export PYTHONPATH=/home/scott/git/auto-finetune
TRAIN_ARGS=${TRAIN_ARGS:---source=hermes}

# Keep scratch off NFS + small root: redirect HF/torch temp to local data drive.
export HF_HOME=/media/scott/data/finetune-staging/hf-home
export TRANSFORMERS_CACHE=/media/scott/data/finetune-staging/hf-home
export HF_DATASETS_CACHE=/media/scott/data/finetune-staging/hf-home/datasets
export TORCH_HOME=/media/scott/data/finetune-staging/torch-home
export TMPDIR=/media/scott/data/finetune-staging/tmp
export TEMP=/media/scott/data/finetune-staging/tmp
export TMP=/media/scott/data/finetune-staging/tmp
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"

echo "=== kernel ==="; uname -r
echo "=== GPU visible? ==="
$V/python - <<'PY'
import torch
print("torch", torch.__version__, "hip", torch.version.hip, "cuda_ok", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(1024,1024,dtype=torch.bfloat16,device="cpu")
    y = x.to("cuda")
    print("H2D ok", y.device, (y@y).shape)
else:
    raise SystemExit("NO GPU")
PY
echo "=== dry-run train ==="
cd /home/scott/git/auto-finetune
$V/python -m src.cli train $TRAIN_ARGS --dry-run 2>&1 | tail -5
echo "DONE"
