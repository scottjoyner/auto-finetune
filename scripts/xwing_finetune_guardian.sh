#!/usr/bin/env bash
# Active external systemd unit calls ~/.local/bin/xwing-finetune-guardian.sh
# Delegate to versioned, tested, hold-aware Python observer. Never starts work.
set -euo pipefail
umask 077
exec /media/scott/data/finetune-venv/bin/python \
  /home/scott/git/auto-finetune/scripts/finetune_guardian.py "$@"
