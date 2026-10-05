#!/bin/bash
# Mirror the irreplaceable parts of finetune-staging to the NAS (/nas, CIFS).
# Safe to cron: exits 0 silently if the NAS is unavailable or too full.
#
# Throughput RE-MEASURED 2026-10-05: 58-66 MB/s (dd probe, 500MB direct +
# cached read-back), not the 1.7-4.3 MB/s recorded on 2026-10-04. The NAS is
# also 53T free against a 1.6T /data at 81% capacity.
#
# The old rate is why this stayed scoped so tightly: at 2 MB/s the full 240G
# staging tree was a multi-day job. At 60 MB/s it is ~70min, so bulk that is
# genuinely regenerable is now handled separately by
# scripts/offload_cold_models_to_nas.sh, which verifies before deleting.
# This script keeps mirroring only what cannot be regenerated: run manifests,
# logs, queue state, built corpora and final adapters.
#
# Scope (~5.3G, ~2min at the current rate; converged runs take ~25s):
#
#   * provenance/                    run manifests (the audit trail)
#   * launch-next.state               queue progress markers
#   * logs/                           training/bench logs
#   * launch/*.sh, launch/*.py        the scripts cron actually invokes
#   * outputs/checkpoints/*/          final adapters, NOT checkpoint-N/*
#   * data/datasets/*.jsonl           built training corpora
#   * data/clean-rebuild/             the benchmark-held-out rebuild
#   * data/eval/, data/future-runs/   eval reports + disjoint eval partitions
#
# Deliberately NOT mirrored:
#   models/                base weights for in-flight runs (Qwen3-8B now,
#                          Ornith-1.5-9B next) plus regenerable downloads.
#                          Superseded iterations are handled by
#                          offload_cold_models_to_nas.sh -> /nas/archive/cold-models
#   outputs/.../checkpoint-* (44G)  resumable training intermediates
#   data/raw, data/cleaned, data/analysis, pip-cache, downloads, tmp,
#   hf-home, torch-home, npu-xclbins, k2-tl-overlay   all regenerable
#
# Never uses --delete: remote fleet nodes push into $DEST/traces/inbox and
# $DEST/data, and a mirror must not reap files it does not own.
set -uo pipefail

STAGING=/media/scott/data/finetune-staging
DEST=/nas/finetune-staging
LOG=/media/scott/data/fleet-power/nas-mirror.log

# The scoped payload is ~5.3G, ~2min at the re-measured 58-66 MB/s; converged
# incremental runs take ~25s. 12G of headroom covers a partial retry.
MIN_FREE_KB=$(( 12 * 1024 * 1024 ))

log() { echo "[nas-mirror] $*" >>"$LOG"; }

# Accumulate failures so the run cannot stamp success over a broken transfer.
RC=0
push() { # push <label> <rsync-args...>
  local label=$1; shift
  # --no-p/-o/-g: /nas is CIFS with noperm and a forced file_mode=0755, so it
  # cannot store Unix modes. Leaving -p (implied by -a) on makes rsync re-report
  # and re-copy every single file forever without ever converging.
  if ! rsync -a --no-p --no-o --no-g --partial --timeout=600 --info=stats2 "$@" \
        >>"$LOG" 2>&1; then
    log "FAIL: $label"
    RC=1
  fi
}

if [ ! -d "$STAGING" ]; then
  log "staging root $STAGING missing - skipping"
  exit 0
fi

if ! mkdir -p "$DEST" 2>/dev/null || [ ! -w "$DEST" ]; then
  log "NAS $DEST not available or not writable - skipping"
  exit 0
fi

FREE_KB=$(df -Pk "$DEST" | awk 'NR==2 {print $4}')
if [ -z "$FREE_KB" ] || [ "$FREE_KB" -lt "$MIN_FREE_KB" ]; then
  log "insufficient free space on $DEST (${FREE_KB:-unknown}K) - skipping"
  exit 0
fi

log "start (free ${FREE_KB}K)"

# --partial keeps a half-written file so an interrupted run resumes instead of
# restarting the whole file; that is exactly how the 2026-10-03 attempt left a
# 0-byte train.combined.jsonl behind. Deletion is off by default in rsync and
# is deliberately never enabled: remote fleet nodes push into $DEST/traces/inbox
# and $DEST/data, and a mirror must not reap files it does not own.

# 1) small critical state
push provenance/ "$STAGING/provenance/" "$DEST/provenance/"
push logs/ "$STAGING/logs/" "$DEST/logs/"
cp "$STAGING/launch-next.state" "$DEST/launch-next.state" 2>/dev/null \
  || log "warn: could not copy launch-next.state"
cp "$STAGING/RUNBOOK.md" "$DEST/RUNBOOK.md" 2>/dev/null || true

# 2) the scripts cron invokes (not the multi-hundred-MB launch artifacts)
push launch-scripts/ --include='*/' --include='*.sh' --include='*.py' --exclude='*' \
  "$STAGING/launch/" "$DEST/launch/"

# 3) final adapters per label, skipping checkpoint-N intermediates
mkdir -p "$DEST/outputs/checkpoints" || { log "FAIL: mkdir checkpoints"; RC=1; }
for label in "$STAGING"/outputs/checkpoints/*/; do
  [ -f "$label/adapter_model.safetensors" ] || continue
  name=$(basename "$label")
  push "adapter:$name" --exclude='checkpoint-*' "$label" "$DEST/outputs/checkpoints/$name/"
done

# 4) built corpora + eval artifacts
mkdir -p "$DEST/data/datasets" "$DEST/data/eval" "$DEST/data/future-runs" \
  || { log "FAIL: mkdir data"; RC=1; }
push data/datasets/ "$STAGING/data/datasets/" "$DEST/data/datasets/"
push data/eval/ "$STAGING/data/eval/" "$DEST/data/eval/"
push data/future-runs/ "$STAGING/data/future-runs/" "$DEST/data/future-runs/"

# Stamp last, and only when every transfer succeeded, so the stamp never lies
# about coverage.
if [ "$RC" -eq 0 ]; then
  date -u +%Y%m%dT%H%M%SZ > "$DEST/.last-mirror-utc"
  log "done $(cat "$DEST/.last-mirror-utc")"
else
  log "done WITH ERRORS - stamp not advanced"
fi
exit "$RC"
