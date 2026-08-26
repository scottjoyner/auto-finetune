#!/bin/bash
# serve_lfm25.sh — post-SFT pipeline: merge -> GGUF -> serve -> smoke gate.
#
# Run AFTER launch-next finishes the lfm-combined SFT (marker done-lfm25-sft-r1):
#   cd ~/git/auto-finetune && bash scripts/serve_lfm25.sh
#
# Steps:
#   1. merge the LoRA adapter into a standalone model
#   2. convert to GGUF (q8_0) with the local llama.cpp fork
#   3. restart llama-server on :8095 hosting the finetuned model
#   4. run one harness task as a smoke gate; report PASS/FAIL
set -euo pipefail

REPO=/home/scott/git/auto-finetune
LLAMA=/home/scott/git/llama.cpp
VENV=/media/scott/data/finetune-venv/bin
STAGING=/media/scott/data/finetune-staging
LABEL=lfm-combined
PORT=8095
export PATH="$VENV:$PATH" PYTHONPATH="$REPO"
export HF_HOME="$STAGING/hf-home"
# merge_adapter() is device_map=None (CPU): safe alongside a GPU training run
# once the lease manager knows we are CPU-only.
export MERGE_DEVICE=cpu

ADAPTER="$STAGING/outputs/checkpoints/toolcall-v5-3b-combined-r2/../lfm2.5-1.2b-sft-r1"
OUT="$STAGING/outputs/checkpoints/lfm2.5-1.2b-sft-r1-merged"
GGUF_DIR="$STAGING/models/lfm25"
GGUF="$GGUF_DIR/LFM2.5-1.2B-sft-r1-Q8_0.gguf"

echo "[serve-lfm25] 1/4 merging adapter..."
TRAIN_MODEL_NAME="$STAGING/models/LFM2.5-1.2B-Base" \
TRAIN_ADAPTER="$STAGING/outputs/checkpoints/lfm2.5-1.2b-sft-r1" \
python -m src.cli merge --label=$LABEL
OUT="$STAGING/outputs/checkpoints/lfm2.5-1.2b-sft-r1-merged"
echo "[serve-lfm25] merged at: $OUT"

echo "[serve-lfm25] 2/4 converting to GGUF q8_0..."
mkdir -p "$GGUF_DIR"
python "$LLAMA/convert_hf_to_gguf.py" "$OUT" --outfile "$GGUF" --outtype q8_0

echo "[serve-lfm25] 3/4 restarting llama-server on :$PORT..."
pkill -f "llama-server.*8095" 2>/dev/null || true
sleep 2
# GPU auto-detect: when no exclusive trainer holds the GPU, serve with the
# ROCm build at full offload (~10-20x decode vs CPU). During training, fall
# back to the CPU build so serving never contends for the device.
LLAMA_BIN="$LLAMA/build-cpu/bin/llama-server"
NGL=0
if ! pgrep -f "src.cli train" >/dev/null 2>&1 \
   && [ -x "$LLAMA/build-rocm/bin/llama-server" ]; then
  LLAMA_BIN="$LLAMA/build-rocm/bin/llama-server"
  NGL=999
  echo "[serve-lfm25] GPU free -> ROCm backend, full offload"
fi
setsid "$LLAMA_BIN" \
  --model "$GGUF" --host 127.0.0.1 --port $PORT \
  --ctx-size 8192 --parallel 1 --jinja --no-webui \
  --special -ngl $NGL \
  </dev/null >>"$STAGING/logs/lfm-serve.log" 2>&1 &
sleep 8

echo "[serve-lfm25] 4/4 smoke gate: one harness task..."
PROMPT='Create a file named hello.txt containing exactly: hello world'
# 900s: the CPU llama-server shares cores with any concurrent training/eval
# (observed 4 t/s under load); a multi-turn harness task needs headroom.
RESULT=$(timeout 900 python - "$PROMPT" <<'PY'
import json, sys
from src.lfm_harness import HarnessContext, handle_request
from src.drivers_lfm25 import LFM25Driver
from src.bench import run_task
ctx = HarnessContext(make_driver=lambda **kw: LFM25Driver(**kw), run_one=run_task)
resp = handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "lfm_task",
                                  "arguments": {"prompt": sys.argv[1]}}}, ctx)
text = resp["result"]["content"][0]["text"]
print(text)
PY
) || RESULT="smoke harness error"
echo "[serve-lfm25] smoke: $(echo "$RESULT" | head -1)"
# Gate intent: prove the served model issues parseable tool calls that the
# harness executes. completed=True is ideal; >=1 executed [tool] action also
# proves the serving path end-to-end (young adapters often loop before
# converging on a final no-tool answer).
case "$RESULT" in
  *completed=True*|*"\\n[tool] "*|*"[tool] "*)
    echo "[serve-lfm25] PASS — finetuned LFM2.5 serving on :$PORT (tool execution verified)";;
  *)
    echo "[serve-lfm25] FAIL — inspect logs"; exit 1;;
esac
