#!/bin/bash
# Offload cold, unreferenced model weights from local NVMe to the NAS.
#
# Why this exists: /nas measured 58-66 MB/s on 2026-10-05 (dd probe, 500MB),
# not the 1.7-4.3 MB/s the mirror script's comment still claims. At 60 MB/s the
# full 187G models tree is ~50min, so moving genuinely dead weight is now cheap
# and /data is at 81% capacity. The old rate is why this was previously skipped.
#
# Safety model, in order:
#   1. Only paths named in COLD_MODELS are ever considered. That list is
#      deliberately conservative -- each entry was checked for references in
#      config.yaml, launch-next.sh, src/*.py, fleet-power/, and against any
#      running llama-server's --model argument.
#   2. Copy, then verify the destination by file count AND byte count.
#   3. Only then remove the source. Never before verification passes.
#   4. Never --delete on the destination: remote nodes push into /nas too.
#
# Models needed by the live pipeline are NOT here and must never be added:
#   Qwen3-8B            currently training (pid 9571)
#   Ornith-1.5-9B       next in the launch-next queue
#   RefinedToolCallV5-3b  config.yaml train.model_name (all toolcall runs)
#   MiniCPM5-2B         base for the iter chain
#   MiniCPM5-2B-iter15-merged  newest iteration, kept as the live pointer
set -uo pipefail

STAGING=/media/scott/data/finetune-staging
DEST=/nas/archive/cold-models
LOG=/media/scott/data/fleet-power/nas-offload.log
MIN_FREE_KB=$(( 30 * 1024 * 1024 ))   # 30G headroom on the NAS

# Superseded MiniCPM5 merge iterations. iter15 is the newest and stays local.
# 12 x ~4.7G ~= 53G.
COLD_MODELS=(
  MiniCPM5-2B-iter2-merged
  MiniCPM5-2B-iter3-merged
  MiniCPM5-2B-iter4-merged
  MiniCPM5-2B-iter5-merged
  MiniCPM5-2B-iter6-merged
  MiniCPM5-2B-iter7-merged
  MiniCPM5-2B-iter8-merged
  MiniCPM5-2B-iter9-merged
  MiniCPM5-2B-iter10-merged
  MiniCPM5-2B-iter11-merged
  MiniCPM5-2B-iter12-merged
  MiniCPM5-2B-iter13-merged
  MiniCPM5-2B-iter14-merged
)

log() { echo "[nas-offload] $(date '+%F %T') $*" >>"$LOG"; }

mkdir -p "$DEST" 2>/dev/null || true
# Create the log's parent, not the log itself: `mkdir -p "$LOG"` would make a
# directory named nas-offload.log and every write to it would fail.
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
if [ -d "$LOG" ]; then
  log "ERROR: $LOG is a directory, refusing to run"
  exit 1
fi

avail_kb=$(df -Pk /nas 2>/dev/null | awk 'NR==2{print $4}')
if [ -z "${avail_kb:-}" ]; then
  log "SKIP: /nas not available"
  exit 0
fi
if [ "$avail_kb" -lt "$MIN_FREE_KB" ]; then
  log "SKIP: /nas has $((avail_kb/1024/1024))G free, below 30G headroom"
  exit 0
fi

moved=0
skipped=0
for m in "${COLD_MODELS[@]}"; do
  src="$STAGING/models/$m"
  if [ ! -d "$src" ]; then
    skipped=$((skipped+1))
    continue
  fi

  # Refuse if anything is serving this exact path right now.
  if tr '\0' ' ' </proc/*/cmdline 2>/dev/null | grep -q -- "$src"; then
    log "SKIP $m: referenced by a running process"
    skipped=$((skipped+1))
    continue
  fi

  log "copy $m ..."
  # -a preserves structure; --partial survives an interrupted copy;
  # no --delete, no -p/-o/-g (CIFS noperm rejects them).
  if ! rsync -a --partial --no-p --no-o --no-g --timeout=600 \
        "$src/" "$DEST/$m/" 2>>"$LOG"; then
    log "FAIL copy $m: left source intact"
    continue
  fi

  # Verify before removing anything. Compare file count and total bytes.
  s_files=$(find "$src" -type f | wc -l)
  d_files=$(find "$DEST/$m" -type f | wc -l)
  s_bytes=$(du -sb "$src" | cut -f1)
  d_bytes=$(du -sb "$DEST/$m" | cut -f1)
  if [ "$s_files" != "$d_files" ] || [ "$s_bytes" != "$d_bytes" ]; then
    log "FAIL verify $m: src $s_files/$s_bytes vs dst $d_files/$d_bytes -- source kept"
    continue
  fi

  rm -rf "$src"
  log "moved $m ($s_files files, $((s_bytes/1024/1024))MB)"
  moved=$((moved+1))
done

log "done: moved=$moved skipped=$skipped"
echo "moved=$moved skipped=$skipped"