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
# Primary: SSD_4TB (x1-370 NFS, reachable). Fallback: NAS3 (down since Jun 24).
if [ -d /media/scott/SSD_4TB/fileserver ]; then
  DEST_ROOT=/media/scott/SSD_4TB/fileserver/ml-state-backups
elif [ -d /media/scott/SSD_4TB ]; then
  DEST_ROOT=/media/scott/SSD_4TB/agent-state-backups/ml-state-backups
elif [ -d /media/scott/NAS3 ]; then
  DEST_ROOT=/media/scott/NAS3/fileserver/ml-state-backups
else
  echo "[ml-backup] no backup target mounted - skipping"; exit 0
fi

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
