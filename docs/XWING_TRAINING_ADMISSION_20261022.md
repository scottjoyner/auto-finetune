# xwing training admission — 2026-10-22

After the Ubuntu reboot, xwing ran Qwen3-8B fine-tuning and resident Ornith
35B-A3B simultaneously. Approximately 66 MiB RAM was available, 9.8 GiB of swap
was occupied, and ten-second full memory-stall PSI was over 10%. Training
progressed poorly and the host became unresponsive. The training queue was
launched by cron at reboot even though a previous comment warned of conflict.

## Default behavior

On host xwing, pre-start admission fails closed with exit 78 when any condition
holds:

- The operator recovery stop-file exists:
  /media/scott/data/finetune-staging/launch-next.stop
- MemAvailable is below 16 GiB.
- 10-second Linux memory PSI some exceeds 3%, or full exceeds 1%.
- The local LM Studio native API /api/v1/models cannot be read, is malformed,
  or returns a model inventory with missing loaded_instances information.
- Any LM Studio LLM is resident, including an IDLE model. Embedding-only
  instances are not treated as heavyweight LLMs.

The check uses Python standard library and requests only loopback LM Studio.
It does not load models, remove checkpoints, turn off swap, mutate GPU leases,
query external endpoints, or log credentials. It does not claim independent
GPU resource authority: resource lease/fencing still needs separate validation.

The guard executes in the public train/dpo CLI, src.train.main for direct
Python callers, and twice in launch-next.sh (before --reset and before each
queued job). The live stop-file and three boot crontab entry holds must remain
in place until a later explicit restart decision. Training on other machines
keeps its existing admission policy.

## Test and rollout

Run synthetic regressions with:
  python -m pytest -o addopts= tests/test_training_admission.py tests/test_train.py tests/test_cli.py -q

Before installation, back up modified host files and verify exact insertion
points. The operator-owned live checkout has unrelated changes; do not force
checkout, reset it, or replace entire modified files. Use a narrow insert-only
patch and recheck syntax, headroom, and the model API.

Do NOT run the training queue merely to validate this change; a denial
can be verified via python -m src.training_admission and python -m src.cli train.
A green admission result authorizes only a preflight, not training deployment.
