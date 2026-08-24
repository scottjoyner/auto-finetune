#!/usr/bin/env bash
# launch-next.sh — hands-off chained finetune runner.
#
# Runs the next NOT-YET-DONE dataset from the queue, in order. Each run gets
# its own checkpoint dir + log. State lives in STATE_FILE so a reboot (or a
# crash) resumes at the right place instead of repeating finished runs.
#
# Usage:
#   ./launch-next.sh            # run ONE next dataset, then exit
#   ./launch-next.sh --loop     # keep launching the next dataset until queue empty
#   ./launch-next.sh --reset    # forget completion state (re-run everything)
#
# Env overrides:
#   TRAIN_BIN  python interpreter (default: finetune-venv)
#   STOP_FILE  touch this path to make the loop stop after the current run

set -u
REPO=/home/scott/git/auto-finetune
V=/media/scott/data/finetune-venv/bin
export PATH="$V:$PATH"
export PYTHONPATH="$REPO"

# scratch on local data drive (off NFS / small root)
export HF_HOME=/media/scott/data/finetune-staging/hf-home
export TRANSFORMERS_CACHE=/media/scott/data/finetune-staging/hf-home
export HF_DATASETS_CACHE=/media/scott/data/finetune-staging/hf-home/datasets
export TORCH_HOME=/media/scott/data/finetune-staging/torch-home
export TMPDIR=/media/scott/data/finetune-staging/tmp
export TEMP=/media/scott/data/finetune-staging/tmp
export TMP=/media/scott/data/finetune-staging/tmp
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME"

STATE_FILE=/media/scott/data/finetune-staging/launch-next.state
STOP_FILE=/media/scott/data/finetune-staging/launch-next.stop
LOGD=/media/scott/data/finetune-staging/logs
mkdir -p "$LOGD"

# Queue: "label:output_name:done_marker"  (Hermes already DONE — commented out)
QUEUE=(
  "ssd:toolcall-v5-3b-ssd:done-ssd"
  "nas5-main:toolcall-v5-3b-nas5-main:done-nas5-main"
  "nas5-20260717:toolcall-v5-3b-nas5-20260717:done-nas5-20260717"
  "opencode-all:toolcall-v5-3b-opencode-all:done-opencode-all"
  "opencode-portfolio:toolcall-v5-3b-opencode-portfolio:done-opencode-portfolio"
  "hermes-reasoning:toolcall-v5-3b-hermes-reasoning:done-hermes-reasoning"
  "combined:toolcall-v5-3b-combined:done-combined"
  # cron-filtered corpus rebuild (2026-08-23): fresh eval-split partition,
  # retrained on the deduplicated combined corpus
  "combined:toolcall-v5-3b-combined-r2:done-combined-r2"
  # LFM2.5-Base SFT on the same corpus rewritten into LFM-native tool dialect
  # (scripts/make_lfm25_sft.py). Small base: tighter seq window.
  "lfm-combined:lfm2.5-1.2b-sft-r1:done-lfm25-sft-r1:TRAIN_MODEL_NAME=/media/scott/data/finetune-staging/models/LFM2.5-1.2B-Base,TRAIN_MAX_SEQ_LENGTH=4096"
  # A/B experiment: same corpus on Thinking-Instruct — does SFT teach its
  # reasoning loops to converge on tool calls? (probed non-convergent stock)
  "lfm-combined:lfm2.5-thinking-sft-r1:done-lfm25-sft-think-r1:TRAIN_MODEL_NAME=/media/scott/data/finetune-staging/models/LFM2.5-Thinking-Instruct,TRAIN_MAX_SEQ_LENGTH=4096"
  # comparison-only (low priority) — uncomment to include
  # "nas5-old-broken:toolcall-v5-3b-nas5-old-broken:done-nas5-old-broken"
  # "nas5-recover-old:toolcall-v5-3b-nas5-recover-old:done-nas5-recover-old"
)

# each run gets its own output_dir via env so config.yaml doesn't need editing
OUT_BASE=/media/scott/data/finetune-staging/outputs/checkpoints

if [ "${1:-}" = "--reset" ]; then
  rm -f "$STATE_FILE"; echo "state reset"
fi
LOOP=0
if [ "${1:-}" = "--loop" ]; then LOOP=1; fi

# GPU pre-flight check: confirm the ROCm device is visible and a tiny matmul
# works. This catches the "amdgpu not modprobed / /dev/kfd missing" cold-boot
# case. It is intentionally lightweight — the real health signal is the actual
# training run (which we verify separately). Non-fatal: if it fails we warn but
# still proceed, since the heavy probe had a bash quirk that caused false
# negatives.
gpu_ok() {
  "$V/python" - <<'PY' 2>/dev/null
import torch, sys
if not torch.cuda.is_available():
    sys.exit(1)
try:
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    (x @ x).cpu()
except Exception:
    sys.exit(1)
PY
  return $?
}

while true; do
  # pick next undone item
  # Entry format: label:out_name:done_marker[:KEY=VAL,KEY=VAL]
  NEXT=""
  for item in "${QUEUE[@]}"; do
    IFS=':' read -r -a parts <<< "$item"
    label="${parts[0]}"; out="${parts[1]}"; marker="${parts[2]}"
    runenv="${parts[3]:-}"
    if [ ! -f "$STATE_FILE" ] || ! grep -qx "$marker" "$STATE_FILE" 2>/dev/null; then
      NEXT="$label|$out|$marker|$runenv"; break
    fi
  done

  if [ -z "$NEXT" ]; then
    echo "[launch-next] queue empty — all datasets done."
    exit 0
  fi

  label="${NEXT%%|*}"; rest="${NEXT#*|}"; out="${rest%%|*}"; rest="${rest#*|}"
  marker="${rest%%|*}"; runenv="${rest#*|}"

  # stop file check
  if [ -f "$STOP_FILE" ]; then
    echo "[launch-next] STOP_FILE present — halting before $label."
    exit 0
  fi

  # GPU liveness probe (warning-only): runs a real tiny training step to
  # confirm the ROCm runtime isn't wedged. If it fails we log a warning but
  # DO NOT abort — the probe's bash env handling is fragile and the device is
  # verified healthy out-of-band; manual `reboot` remains the real remedy.
  if ! gpu_ok; then
    echo "[launch-next] WARNING: GPU liveness probe failed — device may be wedged. Continuing anyway (verify with rocm-smi)."
  else
    echo "[launch-next] GPU liveness probe OK."
  fi

  LOG="$LOGD/train-$out.log"
  export TRAIN_OUTPUT_DIR="$OUT_BASE/$out"
  mkdir -p "$TRAIN_OUTPUT_DIR"

  # Prefer the held-out-excluded partition built by `eval-split --label=<x>`
  # (future-runs/<label>-seed42/datasets). Training on the raw dataset file
  # would contaminate every later eval with rows it was scored on.
  FR_DATADIR="/media/scott/data/finetune-staging/data/future-runs/${label}-seed42/datasets"
  if [ -f "$FR_DATADIR/train.${label}.jsonl" ]; then
    export TRAIN_DATASET_DIR="$FR_DATADIR"
    echo "[launch-next] dataset: held-out-excluded partition ($FR_DATADIR)"
  else
    unset TRAIN_DATASET_DIR
    echo "[launch-next] dataset: raw (no eval-split partition found — run eval-split to avoid contamination)"
  fi

  echo "[launch-next] starting dataset=$label  -> $out  (log: $LOG)"
  echo "[launch-next] $(date)"

  # Optional per-run env overrides from the queue entry (4th colon field).
  if [ -n "$runenv" ]; then
    IFS=',' read -r -a _kvps <<< "$runenv"
    for _kv in "${_kvps[@]}"; do
      export "${_kv?}"
      echo "[launch-next] override: $_kv"
    done
  fi

  # train reads output_dir from TRAIN_OUTPUT_DIR env if set.
  # Launch as a plain background job of THIS shell (NOT setsid) so `wait`
  # actually blocks until training completes. Only one dataset trains at a
  # time. The whole launch-next.sh is itself started under `setsid` by the
  # caller, so it survives the launching tool session ending.
  "$V/python" -m src.cli train --label="$label" \
    > "$LOG" 2>&1 < /dev/null &
  TRAIN_PID=$!
  echo "[launch-next] train pid=$TRAIN_PID"

  # Block until this run finishes, then record state and proceed.
  wait "$TRAIN_PID"
  RC=$?

  if [ $RC -eq 0 ]; then
    echo "$marker" >> "$STATE_FILE"
    echo "[launch-next] $label finished OK (rc=$RC)."
  else
    echo "[launch-next] $label FAILED (rc=$RC) — stopping. See $LOG"
    exit $RC
  fi

  # Scope per-run overrides: clear anything set by this entry.
  if [ -n "$runenv" ]; then
    IFS=',' read -r -a _kvps <<< "$runenv"
    for _kv in "${_kvps[@]}"; do unset "${_kv%%=*}"; done
  fi

  [ "$LOOP" -eq 1 ] || break
  # tiny pause so a crash loop can't spin instantly
  sleep 5
done
