#!/bin/bash
# Offload the K2-Horizon artifacts to the NAS (/nas).
#
# Safe to run repeatedly and safe to cron: rsync is resumable, nothing is ever
# deleted at the destination, and --bwlimit keeps the NAS write rate low enough
# that a training run sharing the same NVMe is not starved.
#
# Why this is tiered. /nas writes at roughly 1.7-4.3 MB/s (measured 2026-10-04).
# The K2-Horizon weights total ~13.7G, which is 1-2 hours of transfer. The code,
# configs, templates and manifests total well under a megabyte and are the part
# that cannot be re-fetched from anywhere: modeling_k2_horizon.py is the MoVA
# implementation, and it is the file every grafting attempt depends on (see the
# decision-attention notes: the additive-branch, zero-init and gate_proj traps
# are all in that one file). So the small tier always runs; the weights are opt-in.
#
# Note finetune-staging/models is deliberately excluded from
# mirror_staging_to_nas.sh (187G, mostly regenerable base weights). That
# exclusion is why this script exists -- it pulls in the one subtree under
# models/ that holds unrecoverable source rather than weights.
#
# Usage:
#   offload_k2horizon_to_nas.sh            # code/configs/manifests/bench only
#   offload_k2horizon_to_nas.sh --weights  # also push model weights (throttled)
#   offload_k2horizon_to_nas.sh --verify   # checksum-compare code tier, no writes
set -uo pipefail

DEST=/nas/models/k2-horizon
LOG=/media/scott/data/fleet-power/k2horizon-offload.log
STAMP_FILE="$DEST/.last-offload-utc"

MODEL_DIR=/media/scott/data/finetune-staging/models/K2-Horizon-0.9B
KERNEL=/home/scott/ROCmFPX/src/models/k2_horizon.cpp
BENCH_BIN=/home/scott/fleet-data/bin
BENCH_RUNS=$BENCH_BIN/runs

# ~2 MB/s keeps a single CIFS writer from saturating the link the trainer and
# harvest cron share. Raise deliberately, not by accident.
BWLIMIT_KB=${K2H_BWLIMIT_KB:-2048}

WANT_WEIGHTS=0
VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --weights) WANT_WEIGHTS=1 ;;
    --verify)  VERIFY_ONLY=1 ;;
    *) echo "usage: $0 [--weights] [--verify]" >&2; exit 2 ;;
  esac
done

log() { echo "[k2h-offload] $*" >>"$LOG"; }

# /nas is CIFS with noperm and a forced file_mode=0755, so -p can never be
# satisfied and would make rsync re-copy every file forever. Delete is never
# enabled: other nodes may hold references into this tree.
RSYNC=(rsync -a --no-p --no-o --no-g --partial --timeout=600 --info=stats2
       --bwlimit="$BWLIMIT_KB")

copy() { # copy <label> <src> <dest-dir>
  local label=$1 src=$2 dst=$3
  if [ ! -e "$src" ]; then
    log "skip (absent): $label -> $src"
    return 0
  fi
  if "$VERIFY_ONLY" = 1; then
    log "verify-only, not copying: $label"
    return 0
  fi
  if "${RSYNC[@]}" "$src" "$dst/"; then
    log "ok: $label"
  else
    log "FAIL: $label"
    return 1
  fi
}

RC=0

if ! mkdir -p "$DEST" 2>/dev/null || [ ! -w "$DEST" ]; then
  echo "[k2h-offload] $DEST unavailable or not writable - skipping" | tee -a "$LOG"
  exit 0
fi

# --- tier 1: source, configs, templates, manifests, bench reproducibility ---
# Explicit include list: never mirror the whole model dir, because that is where
# the 2.1G safetensors live and they are re-downloadable.
mkdir -p "$DEST/model-code" "$DEST/bench"

if [ -d "$MODEL_DIR" ]; then
  # Excludes must precede --include='*/': rsync applies the FIRST matching rule,
  # so listing the directory include first makes every later exclude unreachable.
  "${RSYNC[@]}" \
    --exclude='__pycache__/' --exclude='.cache/' \
    --include='*/' \
    --include='*.py' --include='*.json' --include='*.jinja' \
    --include='*.md' --include='*.txt' --include='.gitattributes' \
    --exclude='*' \
    "$MODEL_DIR/" "$DEST/model-code/" >>"$LOG" 2>&1 \
    && log "ok: model-code" || { log "FAIL: model-code"; RC=1; }
else
  log "skip (absent): model-code -> $MODEL_DIR"
fi

copy "k2_horizon.cpp" "$KERNEL" "$DEST/model-code" || RC=1

for f in k2h_benchmark.py k2h_terminal_bench.py k2h-bench.sh \
         k2h_benchmark_results_latest.json; do
  copy "bench/$f" "$BENCH_BIN/$f" "$DEST/bench" || RC=1
done

# Per-task run outputs (pareto/summary/metadata) so a benchmark can be re-read
# without re-running it.
if [ -d "$BENCH_RUNS" ]; then
  # Same precedence rule: exclude first, then allow directories.
  "${RSYNC[@]}" --exclude='__pycache__/' --include='*/' --include='*.json' --exclude='*' \
    "$BENCH_RUNS/" "$DEST/bench/runs/" >>"$LOG" 2>&1 \
    && log "ok: bench/runs" || { log "FAIL: bench/runs"; RC=1; }
fi

# --- tier 2: weights (opt-in, throttled) ----------------------------------
if [ "$WANT_WEIGHTS" = 1 ] && [ "$VERIFY_ONLY" = 0 ]; then
  copy "K2-Horizon-0.9B weights" "$MODEL_DIR/model-00000-of-00001.safetensors" "$DEST/weights"
  # /nas cannot hold arbitrary symlinks (CIFS), so resolve before copying.
  for gguf_dir in /home/scott/.lmstudio/models/IFM/K2-Horizon-3.7B-GGUF \
                  /home/scott/.lmstudio/models/IFM/K2-Horizon-0.9B-GGUF; do
    [ -d "$gguf_dir" ] || continue
    name=$(basename "$gguf_dir")
    mkdir -p "$DEST/gguf/$name" 2>/dev/null
    "${RSYNC[@]}" "$gguf_dir/" "$DEST/gguf/$name/" >>"$LOG" 2>&1 \
      && log "ok: gguf/$name" || { log "FAIL: gguf/$name"; RC=1; }
  done
fi

# --- manifest + stamp ------------------------------------------------------
if [ "$VERIFY_ONLY" = 0 ] && [ "$RC" -eq 0 ]; then
  {
    echo "# k2-horizon offload manifest"
    echo "generated_utc: $(date -u +%Y%m%dT%H%M%SZ)"
    echo "dest: $DEST"
    echo "weights_included: $WANT_WEIGHTS"
    echo "source_model_dir: $MODEL_DIR"
    echo "--- sha256 (code tier) ---"
    find "$DEST/model-code" "$DEST/bench" -type f 2>/dev/null | sort | while read -r f; do
      sha256sum "$f"
    done
  } > "$DEST/MANIFEST.txt" 2>/dev/null
  date -u +%Y%m%dT%H%M%SZ > "$STAMP_FILE"
  log "done $(cat "$STAMP_FILE")"
else
  log "done WITH ERRORS or verify-only - stamp not advanced"
fi
exit "$RC"
