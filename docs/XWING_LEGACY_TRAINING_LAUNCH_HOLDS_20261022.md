# xwing legacy training and post-processing launch holds

The primary CLI trainer and launch-next queue previously had fail-closed
admission protection. This patch closes additional pre-load/post-queue
entrypoints which could still probe a GPU, merge and load large models,
restart serving, or upgrade the shared training environment without
first checking the held xwing inference residency.

Before **any scratch/log state creation or GPU allocation** the following
paths now call python -m src.training_admission with the xwing virtualenv
and absolute PYTHONPATH. Every non-success decision exits 78; the live
stop-file keeps xwing held regardless of startup or stale job marker:
- post-reboot-gpu-train.sh (rechecked before the real launch)
- post-reboot-check.sh
- post-queue.sh (also rechecked after its queue-wait / --nowait branch)
- scripts/sft_cascade_watcher.sh (also rechecked before loading serving)
- scripts/serve_lfm25.sh (also rechecked immediately before any pkill)
- scripts/lfm25_showdown.sh
- scripts/legacy_boot/chain-bigmodels.sh (start and after completion wait)
- scripts/legacy_boot/upgrade-transformers-gate.sh (start and after wait,
  BEFORE shared venv pip install)

The two bootchain files are original snapshots of existing live scripts
in /media/scott/data/finetune-staging/launch, with the guards inserted.
They must be deployed with byte-verified per-file backups, without changing
the already-disabled legacy crontab entries.

Synthetic tests replace ONLY the absolute interpreter path in an isolated
copy to simulate admission denial; they exercise all eight entrypoints
and verify exit78 before side effects. This protects against missing live
dependencies in hosted CI and never invokes a real GPU, model, power, NAS,
trainer or queue. It is not proof of training readiness, fencing, or
ongoing scheduling after a physical GPU reconnect.

Existing hold state remains: xwing launch-next.stop, three disabled boot
jobs, no training process, LM Studio Ornith serving healthy, R9700
physically absent on x1. **Never** remove the stop-file, restart training,
or re-enable @reboot entries as part of this hardening.
