#!/bin/bash
# LFM2.5 showdown: stock (8094) vs finetuned (8095) through identical harness.
# Run after serve_lfm25.sh has deployed the SFT model on 8095.
set -uo pipefail
REPO=/home/scott/git/auto-finetune
TASKS=/media/scott/data/finetune-staging/data/analysis/bench-tasks-lfm25.jsonl
export PATH="/media/scott/data/finetune-venv/bin:$PATH" PYTHONPATH="$REPO"
cd "$REPO"

for port in 8094 8095; do
  tag=$([ "$port" = "8094" ] && echo stock || echo finetuned)
  echo "=== $tag (:8094/:8095 -> $port) ==="
  # 2026-08-25: keep per-task rows, not just the completion headline —
  # partial-pass detail is where finetune-vs-stock knowledge gains show.
  # Scratch honours TMPDIR and defaults to the staging tmp on the data mount;
  # /tmp is a separate filesystem that is cleared on reboot.
  DETAIL_DIR=${TMPDIR:-/media/scott/data/finetune-staging/tmp}
  mkdir -p "$DETAIL_DIR" 2>/dev/null || DETAIL_DIR=/tmp
  DETAIL="$DETAIL_DIR/showdown-detail-$tag.txt"
  python -m src.cli bench --runner=lfm25 --base-url=http://127.0.0.1:$port \
    --tasks=$TASKS 2>&1 | tee "$DETAIL" | grep -E "completion:" || true
  grep -E "task|PASS|FAIL|✓|✗|passed" "$DETAIL" | grep -v "completion:" | head -40 || true
done
echo "compare smoke-tier rows above; mined tasks show knowledge gains."
