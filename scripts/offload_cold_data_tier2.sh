#!/bin/bash
# Second-tier cold data offload: quantized model exports and resumable
# checkpoint state.
#
# Tier 1 (offload_cold_models_to_nas.sh) moved superseded merged weights. This
# moves the two remaining large cold categories on /data:
#
#   1. MiniCPM5 quantized exports (f16 / Q4_K_M / lora ggufs) for iterations
#      whose merged dirs tier 1 already moved. Nothing references these:
#      config.yaml, launch-next.sh, src/*.py and fleet-power/ have no hits, and
#      LM Studio serves from /home/scott/.lmstudio/models/, a different tree.
#      iter15 is kept as the newest iteration.
#
#   2. checkpoint-N/ directories from completed runs. These hold resumable
#      trainer state -- useful only to resume an *interrupted* run. A run that
#      already finished has no use for them, and its final adapter is kept
#      locally (tier 0 / the mirror script handles those).
#
# The live run is identified from the running trainer's own TRAIN_OUTPUT_DIR
# rather than a hardcoded name, so this stays correct as the queue advances.
set -uo pipefail

STAGING=/media/scott/data/finetune-staging
MODELS="$STAGING/models"
CKPT="$STAGING/outputs/checkpoints"
DEST=/nas/archive/cold-models
CKPT_DEST=/nas/archive/cold-checkpoints
LOG=/media/scott/data/fleet-power/nas-offload2.log
MIN_FREE_KB=$(( 30 * 1024 * 1024 ))

log() { echo "[nas-offload2] $(date '+%F %T') $*" >>"$LOG"; }

mkdir -p "$(dirname "$LOG")" "$DEST" "$CKPT_DEST" 2>/dev/null || true
if [ -d "$LOG" ]; then
  echo "ERROR: $LOG is a directory, refusing to run" >&2
  exit 1
fi

avail_kb=$(df -Pk /nas 2>/dev/null | awk 'NR==2{print $4}')
if [ -z "${avail_kb:-}" ] || [ "$avail_kb" -lt "$MIN_FREE_KB" ]; then
  log "SKIP: /nas unavailable or below 30G headroom"
  exit 0
fi

# The run currently training: its checkpoint-N dirs are live resume state.
LIVE_CKPT=""
for p in $(pgrep -f 'src.cli train' 2>/dev/null); do
  d=$(tr '\0' '\n' <"/proc/$p/environ" 2>/dev/null \
      | sed -n 's/^TRAIN_OUTPUT_DIR=//p' | head -1)
  [ -n "$d" ] && LIVE_CKPT="$d"
done
log "live run output dir: ${LIVE_CKPT:-<none detected>}"

verify_and_remove() {   # src, dest, label
  local src="$1" dst="$2" label="$3"
  # A trailing slash makes rsync insist the source is a directory, so single
  # files (the gguf exports) and directories need different invocations.
  local srcarg="$src"
  if [ -d "$src" ]; then
    srcarg="$src/"
  elif [ -d "$dst" ]; then
    # A directory sitting where a file belongs: rsync would silently write the
    # file *inside* it and the byte-count check below would then compare a
    # file against a directory. Clear it first.
    log "  removing stale directory at destination $dst"
    rm -rf "$dst"
  fi
  # rsync will not create intermediate destination directories (mkdir of
  # <dest>/<run>/checkpoint-N fails with ENOENT when <dest>/<run> is absent).
  mkdir -p "$(dirname "$dst")" 2>/dev/null || true
  log "copy $label ..."
  if ! rsync -a --partial --no-p --no-o --no-g --timeout=900 \
        "$srcarg" "$dst" 2>>"$LOG"; then
    log "FAIL copy $label: source kept"
    return 1
  fi
  local sf df_ sb db
  if [ -d "$src" ]; then
    sf=$(find "$src" -type f | wc -l); df_=$(find "$dst" -type f | wc -l)
  else
    sf=1; df_=$([ -f "$dst" ] && echo 1 || echo 0)
  fi
  sb=$(du -sb "$src" | cut -f1)
  if [ -d "$src" ]; then
    db=$(du -sb "$dst" | cut -f1)
  else
    db=$([ -f "$dst" ] && stat -c %s "$dst" || echo 0)
  fi
  if [ "$sf" != "$df_" ] || [ "$sb" != "$db" ]; then
    log "FAIL verify $label: src $sf/$sb vs dst $df_/$db -- source kept"
    return 1
  fi
  rm -rf "$src"
  log "moved $label ($sf files, $((sb/1024/1024))MB)"
  return 0
}

moved=0

# ── 1. quantized exports for superseded iterations ─────────────────────────
# Keep iter15 (newest). Move iter2..iter14 ggufs.
for f in "$MODELS"/MiniCPM5-2B-iter*-*.gguf; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  case "$base" in
    *iter15-*) continue ;;                       # newest iteration stays
  esac
  if tr '\0' ' ' </proc/*/cmdline 2>/dev/null | grep -q -- "$f"; then
    log "SKIP $base: referenced by a running process"
    continue
  fi
  verify_and_remove "$f" "$DEST/$base" "$base" && moved=$((moved+1))
done

# Superseded LoRA adapters (small, but same reasoning).
for d in MiniCPM5-2B-adapter MiniCPM5-2B-adapter-iter2; do
  [ -d "$MODELS/$d" ] || continue
  verify_and_remove "$MODELS/$d" "$DEST/$d" "$d" && moved=$((moved+1))
done

# ── 2. checkpoint-N state from completed runs ──────────────────────────────
if [ -d "$CKPT" ]; then
  for run_dir in "$CKPT"/*/; do
    run=$(basename "$run_dir")
    [ -d "$run_dir" ] || continue
    if [ "$run_dir" = "${LIVE_CKPT%/}/" ] || [ "$run_dir" = "$LIVE_CKPT" ]; then
      log "SKIP $run: this is the live run"
      continue
    fi
    for cp in "$run_dir"checkpoint-*; do
      [ -d "$cp" ] || continue
      label="$run/$(basename "$cp")"
      verify_and_remove "$cp" "$CKPT_DEST/$run/$(basename "$cp")" "$label" \
        && moved=$((moved+1))
    done
    # Tidy the now-empty parent if nothing but checkpoints lived there.
    rmdir "$run_dir" 2>/dev/null && log "removed empty $run_dir"
  done
fi

log "done: moved=$moved"
echo "moved=$moved"