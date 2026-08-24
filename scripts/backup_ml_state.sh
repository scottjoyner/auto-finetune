#!/bin/bash
# Backup the irreplaceable ML state to NAS3 when it is mounted.
# Safe to cron weekly: exits 0 silently if NAS3 is unavailable.
#
# Covers (small, high-value only — datasets/merged models are regenerable
# from raw stores and base weights):
#   * launch-next.state            (queue progress markers)
#   * fleet-power.db               (power/usage time series)
#   * provenance/                  (run manifests)
#   * per-label final adapter files (adapter_model.safetensors etc.,
#     excluding checkpoint-N intermediates)
set -uo pipefail

STAGING=/media/scott/data/finetune-staging
DEST_ROOT=/media/scott/NAS3/fileserver/ml-state-backups
[ -d /media/scott/NAS3 ] || { echo "[ml-backup] NAS3 not mounted — skipping"; exit 0; }
mkdir -p "$DEST_ROOT" || { echo "[ml-backup] cannot write $DEST_ROOT"; exit 0; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$DEST_ROOT/xwing/$STAMP"
mkdir -p "$RUN"

# 1) small critical files
cp "$STAGING/launch-next.state" "$RUN/" 2>/dev/null || true
cp /media/scott/data/fleet-power/fleet-power.db "$RUN/" 2>/dev/null || true
rsync -a "$STAGING/provenance" "$RUN/" 2>/dev/null || true

# 2) final adapters per label (skip intermediate checkpoint-* dirs)
CKPT="$STAGING/outputs/checkpoints"
for label in "$CKPT"/*/; do
  name=$(basename "$label")
  [ -f "$label/adapter_model.safetensors" ] || continue
  mkdir -p "$RUN/adapters/$name"
  rsync -a --exclude 'checkpoint-*' "$label" "$RUN/adapters/$name/"
done

echo "$STAMP" > "$DEST_ROOT/xwing/latest.txt"
du -sh "$RUN" | awk '{print "[ml-backup] "$0}'
