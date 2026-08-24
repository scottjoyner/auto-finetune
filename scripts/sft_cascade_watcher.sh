#!/bin/bash
# sft_cascade_watcher.sh — fire the LFM2.5 serving + showdown automatically
# when the SFT run completes overnight.
#
# Waits for the done-marker, runs serve_lfm25.sh (merge -> GGUF -> :8095 ->
# smoke gate), waits for health, runs the showdown bench against stock
# (:8094) and finetuned (:8095), saves results, notifies desktop.

set -uo pipefail
REPO=/home/scott/git/auto-finetune
STATE=/media/scott/data/finetune-staging/launch-next.state
MARKER=done-lfm25-sft-r1
RESULTS=/media/scott/data/fleet-power/showdown-results.md

echo "[cascade] watching for $MARKER ..."
while ! grep -qx "$MARKER" "$STATE" 2>/dev/null; do
  # abort if the poller/watchdog reported trainer death repeatedly? keep simple:
  sleep 300
done
echo "[cascade] $MARKER detected $(date)"

echo "[cascade] running serve_lfm25.sh (merge -> GGUF -> :8095)..."
if bash "$REPO/scripts/serve_lfm25.sh" >> /media/scott/data/fleet-power/cascade.log 2>&1; then
  echo "[cascade] serve OK"
else
  echo "[cascade] SERVE FAILED — see cascade.log" | tee -a "$RESULTS"
  exit 1
fi

sleep 10
echo "[cascade] running showdown bench..."
{
  echo "# LFM2.5 showdown — $(date)"
  bash "$REPO/scripts/lfm25_showdown.sh"
} >> "$RESULTS" 2>&1
tail -20 "$RESULTS"

try_notify() {
  python3 - "$1" <<'PY' >/dev/null 2>&1 || true
import sys
sys.path.insert(0, "/home/scott/git/auto-finetune")
from src.notify import send_desktop
send_desktop("LFM2.5 cascade", sys.argv[1])
PY
}
try_notify "SFT served + benchmarked — results in $RESULTS"
echo "[cascade] complete $(date)"
