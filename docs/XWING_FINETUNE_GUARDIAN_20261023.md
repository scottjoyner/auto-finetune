# xwing fine-tuning guardian: hold-aware process and temperature monitoring

The previous five-minute systemd guardian counted lfm_harness
processes as training. On October 23 two resident harnesses were
active, while the actual fine-tuning trainer was stopped. Process
death detection based on a serving harness was unreliable.

The replacement scripts/finetune_guardian.py reads /proc command lines
and counts only exact Python training entrypoints: -m src.cli train,
-m src.cli dpo, -m src.train and -m src.dpo. Shell, grep, watchdog,
scheduler and -m src.lfm_harness do not count.

An idle held trainer is not a death alarm. A trainer active with
launch-next.stop present triggers an anomaly alert. A real
active-to-absent transition triggers an alert once, not on every
five-minute poll. GPU edge temperature >=88C on two consecutive
read-only rocm-smi samples triggers one warning until cooling.
Missing sensor readings are null and do not imply temperature 0.

The implementation preserves the existing user systemd five-minute
service and timer, local Signal note-to-self destination, and
guardian.log path. Atomic receipt is mode0600 in the user's local
state directory; JSONL log includes counts, hold, temperature, event
names and notification outcomes, not model prompts or full argv.
--dry-run performs no writes and sends no notifications.

Versioned scripts/xwing_finetune_guardian.sh is the intended
replacement for ~/.local/bin/xwing-finetune-guardian.sh. Only deploy
after CI and backup the original with exact-byte verification. Do
not restart training, models, fleet lease ownership, or timers.

Acceptance: xwing dry-run stopfile=true, training_processes=0,
temperature approximately 67C, no alarms. Tests cover held idle,
held violation, actual trainer departure, consecutive heat alerts,
process-recognition strictness, receipt permission and dry-run.
