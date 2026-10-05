#!/bin/bash
# Backup the irreplaceable ML state to the first writable backup target.
# Safe to cron weekly: exits 0 silently if no target is usable.
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

# Pick the first target we can actually WRITE, not merely one that exists.
# The 2026-09-21 run picked /media/scott/SSD_4TB/fileserver (a symlink to the
# NAS3 export) because the path resolved, then every mkdir failed with
# "Read-only file system" and the whole backup was silently lost -- the
# -d test cannot see that condition.
#
# Order: proven SSD_4TB NFS target, then the CIFS NAS, then the legacy NAS3.
pick_dest() {
  local candidate
  for candidate in "$@"; do
    if mkdir -p "$candidate" 2>/dev/null && [ -w "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    echo "[ml-backup] unusable target: $candidate" >&2
  done
  return 1
}

DEST_ROOT=$(pick_dest \
  /media/scott/SSD_4TB/agent-state-backups/ml-state-backups \
  /nas/ml-state-backups \
  /media/scott/NAS3/fileserver/ml-state-backups) || {
  echo "[ml-backup] no writable backup target - skipping"
  exit 0
}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN="$DEST_ROOT/xwing/$STAMP"
if ! mkdir -p "$RUN"; then
  echo "[ml-backup] cannot create $RUN - skipping"
  exit 0
fi

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

# 2b) adapters from the candidate pipeline.
#
# Since the scheduler was wired to run_candidate, adapters are trained inside
# analysis/candidates/<id>/adapters/<label>/ and only the *merged* winner is
# published to outputs/checkpoints. So step 2 no longer sees the adapter behind
# a promotion, and a merged model cannot be re-derived without it. Back up the
# adapters of candidates that actually completed (candidate.json present), which
# bounds this to real promotions rather than every abandoned attempt.
CANDS="$STAGING/data/analysis/candidates"
if [ -d "$CANDS" ]; then
  for cand in "$CANDS"/*/; do
    [ -d "$cand" ] || continue
    [ -f "$cand/candidate.json" ] || continue
    cid=$(basename "$cand")
    for adir in "$cand"adapters/*/; do
      [ -f "$adir/adapter_model.safetensors" ] || continue
      label=$(basename "$adir")
      dest="$RUN/candidates/$cid/$label"
      mkdir -p "$dest"
      rsync -a --exclude 'checkpoint-*' "$adir" "$dest/"
    done
    # The manifest is the only record of which candidate won and why.
    [ -f "$cand/candidate.json" ] && cp "$cand/candidate.json" \
      "$RUN/candidates/$cid/" 2>/dev/null || true
  done
fi

echo "$STAMP" > "$DEST_ROOT/xwing/latest.txt"
du -sh "$RUN" | awk '{print "[ml-backup] "$0}'
