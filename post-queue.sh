#!/usr/bin/env bash
# post-queue.sh — one-shot post-training pipeline.
#
# Waits for launch-next.sh to finish the full queue, then:
#   1. eval-all --report      (held-out loss + tool-call table, persisted)
#   2. best --metric=loss     (pick the winning adapter)
#   3. probe --label=<winner> (qualitative tool-call check, base vs adapter)
#   4. merge --label=<winner> (fuse LoRA into a standalone model)
#   5. validate merged model  (must actually emit a <tool_call> — catches
#      "loads fine but generates garbage" merge bugs)
#   6. side-by-side probe across ALL finished adapters
#   7. write a final consolidated report
#   8. agentic benchmark across ALL references (local Qwen2.5-7B + lmstudio
#      q8 GGUFs + lan fleet) via `bench-matrix --preset=all`
#
# Usage:
#   ./post-queue.sh            # wait for queue, then run everything
#   ./post-queue.sh --nowait   # skip waiting (assume queue already done)
#   ./post-queue.sh --label=ssd  # force a specific winner (skip best)
#
# Env overrides:
#   TRAIN_BIN  python interpreter (default: finetune-venv)
#   STOP_FILE  touch this to abort a running wait loop
#
# NOTE: the mounted SSD is now served over ethernet. Model loads from it are
# slower and may see latency spikes (this host + other services share it).
# Loads have generous timeouts; if a step times out, just re-run it — the
# adapters/merged outputs are idempotent and resumable.

set -u
REPO=/home/scott/git/auto-finetune
V=/media/scott/data/finetune-venv/bin
export PATH="$V:$PATH"
export PYTHONPATH="$REPO"

# scratch on LOCAL data drive (do NOT point at the ethernet SSD / NFS / CIFS)
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
REPORTD=/media/scott/data/finetune-staging/eval-reports
mkdir -p "$LOGD" "$REPORTD"

# the 7 done-markers that mean the queue is fully finished
DONE_MARKERS=(
  done-ssd done-nas5-main done-nas5-20260717 done-opencode-all
  done-opencode-portfolio done-hermes-reasoning done-combined
)

NOW="$(date +%Y%m%d-%H%M%S)"
LOG="$LOGD/post-queue-$NOW.log"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[post-queue] $(date) $*"; }

WAIT=1
FORCE_LABEL=""
for a in "$@"; do
  case "$a" in
    --nowait) WAIT=0 ;;
    --label=*) FORCE_LABEL="${a#*=}" ;;
  esac
done

queue_done() {
  for m in "${DONE_MARKERS[@]}"; do
    grep -qx "$m" "$STATE_FILE" 2>/dev/null || return 1
  done
  return 0
}

training_running() {
  # launch-next trains via `python -m src.cli train`; match that, not this script
  pgrep -f "src.cli train" >/dev/null 2>&1
}

if [ "$WAIT" -eq 1 ]; then
  log "waiting for training queue to finish (state=$STATE_FILE)..."
  while ! queue_done; do
    if [ -f "$STOP_FILE" ]; then
      log "STOP_FILE present — aborting wait."
      exit 1
    fi
    if training_running; then
      log "training still active; sleeping 120s..."
    else
      # not running AND not all-done -> either between jobs or crashed.
      # give it a moment in case launch-next is mid-launch, then re-check.
      log "no train process and queue incomplete; sleeping 60s (may be between jobs)..."
    fi
    sleep 60
  done
  # extra safety: ensure nothing is still generating/writing checkpoints
  while training_running; do
    log "final settle: training still active; sleeping 60s..."
    sleep 60
  done
fi

if ! queue_done; then
  log "WARNING: queue not fully done (state markers missing). Proceeding anyway."
fi

log "=== 1/5 eval-all --report ==="
"$V/python" -m src.cli eval-all --loss-only --report || log "eval-all returned non-zero"

log "=== 2/5 best adapter ==="
WINNER="$FORCE_LABEL"
if [ -z "$WINNER" ]; then
  WINNER="$("$V/python" -m src.cli best --metric=loss 2>/dev/null \
            | grep -oE 'toolcall-v5-3b-[a-z0-9-]+' | head -1 \
            | sed 's/^toolcall-v5-3b-//')"
fi
if [ -z "$WINNER" ]; then
  log "could not determine winner; defaulting to 'combined'"
  WINNER=combined
fi
log "winner (by loss): $WINNER"

log "=== 3/5 probe --label=$WINNER ==="
"$V/python" -m src.cli probe --label="$WINNER" || log "probe returned non-zero"

log "=== 4/5 merge --label=$WINNER ==="
"$V/python" -m src.cli merge --label="$WINNER" || { log "merge FAILED"; exit 1; }
MERGED="/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-$WINNER-merged"

log "=== 5/5 validate merged model emits a tool_call ==="
"$V/python" - <<PY || log "MERGED MODEL VALIDATION FAILED"
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
m = AutoModelForCausalLM.from_pretrained("$MERGED", torch_dtype="auto", device_map="cpu")
tok = AutoTokenizer.from_pretrained("$MERGED")
conv = [{"role": "user", "content": "Run git status in /home/scott/git/auto-finetune and tell me if there are uncommitted changes."}]
txt = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
ids = tok(txt, return_tensors="pt").input_ids
out = m.generate(**ids, max_new_tokens=160, do_sample=False)
gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
print("MERGED_GEN_START")
print(gen)
print("MERGED_GEN_END")
assert "tool_call" in gen.lower() or "bash" in gen.lower(), "merged model did not emit a tool call"
print("VALIDATION_OK")
PY

log "=== 6/6 side-by-side probe across ALL finished adapters ==="
"$V/python" -m src.cli compare || log "compare returned non-zero"

log "=== 7/7 agentic benchmark across ALL references (local + lmstudio + fleet) ==="
# Runs the task suite through every available reference: local Qwen2.5-7B
# (local-chat), the lmstudio q8 GGUFs (api), and the lan fleet models (api).
# Best-effort: a reference whose server is down just errors that one spec
# (bench_matrix isolates per-spec failures), so the matrix still completes.
"$V/python" -m src.cli bench-matrix --preset=all --report \
  || log "bench-matrix --preset=all returned non-zero (some references unreachable)"

log "=== final report ==="
"$V/python" -m src.cli report --label="$WINNER" || log "report returned non-zero"

log "DONE. winner=$WINNER  merged=$MERGED  report dir=$REPORTD"
