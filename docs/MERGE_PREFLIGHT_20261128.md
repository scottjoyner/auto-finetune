# Merge memory + resident-LLM preflight — November 28, 2026

`src merge` (post-training adapter fold into a standalone base checkpoint) now
runs a fail-closed preflight before any model import or filesystem side effect:

- `src.merge.preflight_merge()` refuses when `MemAvailable` from
  `/proc/meminfo` is below `MIN_MERGE_AVAILABLE_KIB` (16 GiB — the same
  conservative floor the xwing training admission gate enforces via
  `training_admission.MIN_KIB`). A merge loads the full base model in bf16 plus
  PEFT buffers, so it needs the same order of headroom as the training run it
  follows.
- On xwing it additionally reads the local model inventory (LM Studio
  `/api/v1/models` loopback) and refuses when the inventory is unusable or when
  an LLM is resident for inference (`resident_llm`). An idle embedding instance
  is harmless; an LLM that is loaded is not.
- The preflight never imports torch and fails closed on unreadable telemetry.

Denials raise `src.merge.MergePreflightError`; the CLI prints `[error] <reason>`
and exits 78 before any adapter merge work starts. (Training admission HOLD also
uses exit 78; the CLI message prefix keeps the two distinguishable in operator
logs.)

## Merge device placement

`MERGE_DEVICE=cpu` selects the CPU merge path (`device_map=None`) for any base,
including LFM3. The default for LFM3 remains `device_map="auto"` (GPU placement,
faster merging). The two must not diverge: lease handling
(`src.locking.command_resources`) and `scripts/serve_lfm25.sh` treat
`MERGE_DEVICE=cpu` as a CPU merge, so the merge must honor it.

## Relationship to training admission

The merge preflight reuses the training-admission memory and inventory helpers
(`training_admission.read_mem_available_kib`, `read_model_inventory`,
`inventory_error`, `resident_llm`). Training admission itself is governed by the
operator-retired recovery latch plus the stop-file, memory and resident-LLM gates.
See `docs/XWING_TRAINING_ADMISSION_20261022.md` and
`docs/XWING_QWEN_DUAL_RECOVERY_HOLD_20261024.md`.

## Tests

Synthetic regressions live in `tests/test_merge_preflight.py` (fake `/proc/meminfo`,
stubbed inventory, stubbed model loaders; no network, no GPU, no model load).
The admission workflow (`.github/workflows/xwing-training-admission.yml`) runs it
alongside `tests/test_training_admission.py`.
