# Skill: Auto-Harvester Pipeline

Data harvesting, cleaning, analysis, and dataset preparation for the auto-finetune system. All operations in this skill are **CPU-only** and safe to run while GPU training is active.

## Environment Setup

```bash
export PATH="/media/scott/data/finetune-venv/bin:$PATH"
export PYTHONPATH=/home/scott/git/auto-finetune
export HF_HOME=/media/scott/data/finetune-staging/hf-home
export TMPDIR=/media/scott/data/finetune-staging/tmp
cd /home/scott/git/auto-finetune
```

## Storage Layout

| Type | Path | Notes |
|------|------|-------|
| Raw sessions | `/media/scott/data/finetune-staging/data/raw/` | Per-source subdirs |
| Cleaned sessions | `/media/scott/data/finetune-staging/data/cleaned/` | Normalized conversations |
| Datasets | `/media/scott/data/finetune-staging/data/datasets/` | Final train/*.jsonl files |
| Analysis | `/media/scott/data/finetune-staging/data/analysis/` | Buckets, tasks, metrics |

**CRITICAL**: Never overwrite `datasets/` while training is running. HuggingFace loaders memory-map the file; replacing it mid-run crashes training.

## Data Sources

| Source | Path | Type | Notes |
|--------|------|------|-------|
| Live opencode | `/media/scott/SSD_4TB/opencode/opencode.db` | SQLite (corrupt) | Only growing source. Read-only via CorruptDB |
| Live Hermes | `/media/scott/SSD_4TB/hermes-home/.hermes/state.db` | SQLite | Canonical Hermes store |
| NAS5 snapshots | `/media/scott/NAS5/opencode/opencode*.db` | SQLite | Static, already harvested. Skip. |

Both live DBs are in WAL mode; read-only extraction does not block writes.

## Pipeline Flow

```
extract  ->  clean  ->  format  ->  combine  ->  eval-split
 (raw)      (cleaned) (datasets/*.jsonl)          (eval/held-out-<label>.jsonl)
```

## Core Commands

### Extract (opencode)

```bash
python -m src.cli extract --label=ssd          # Live SSD_4TB store only
python -m src.cli extract --label=nas5-main    # Specific NAS5 snapshot
python -m src.cli extract --project=portfolio  # Filter by project
```

- Writes `raw/<label>/<session>.json` per session
- Idempotent (files named by session ID); adds only new sessions
- Handles corrupted DBs via row-by-row reading

### Extract (Hermes)

```bash
python -m src.cli hermes
```

- Writes `raw/hermes_*.json` files
- Source: `config.yaml` -> `sources.hermes.state_db`

### Clean

```bash
python -m src.cli clean                        # Clean all raw sessions
python -m src.cli clean --label=ssd            # Clean specific label
python -m src.cli clean --keep-reasoning       # Keep reasoning as context
```

- Redacts secrets/tokens (`clean.redact_secrets: true`)
- Drops empty turns (`clean.drop_empty_turns: true`)
- Truncates long messages (`clean.max_chars_per_message: 32000`)
- Deduplicates identical conversations (`clean.dedupe: true`)
- Output: `cleaned/*.json` (flat) + `cleaned/<label>/` subdirs

### Format

```bash
python -m src.cli format                       # Merged train.jsonl
python -m src.cli format --source=hermes       # train.hermes.jsonl only
python -m src.cli format --label=ssd           # train.ssd.jsonl only
python -m src.cli format --all-split           # hermes + opencode + merged
```

- Template: `hermes` (uses tokenizer chat_template)
- Max chars per example: 24000 (~8k tokens)
- Sliding window for long sessions

### Combine

```bash
python -m src.cli combine
```

- Deduped union of all formatted labels -> `train.combined.jsonl`

### Eval Split

```bash
python -m src.cli eval-split --label=ssd --frac=0.1
# Repeat for all labels:
for L in ssd nas5-main nas5-20260717 opencode-all opencode-portfolio hermes-reasoning combined; do
  python -m src.cli eval-split --label=$L --frac=0.1 || true
done
```

- Carves held-out splits -> `eval/held-out-<label>.jsonl`

## Analysis Commands

### Analyze (Task Bucket Classification)

```bash
python -m src.cli analyze
python -m src.cli analyze --out=/custom/path
```

- CPU-only heuristic classifier (no LLM needed)
- Features: tool-call histogram, file extensions, intent keywords, turn count
- Buckets: `shell`, `file-edit`, `multi-file-refactor`, `code-search`, `debug`, `web-research`, `data-analysis`, `docs`, `reasoning/math`, `mixed`
- Output: `analysis/buckets.json`, `analysis/corpus.json`, `analysis/auto-tasks.jsonl`, `analysis/failures.jsonl`

### Strata (Bucket-Based Training Sets)

```bash
python -m src.cli strata --out=analysis/ --balance --cap=500
python -m src.cli strata --holdout=analysis/auto-tasks.jsonl --out=analysis/
```

- Emits `train.<bucket>.jsonl` per task bucket
- `--balance` equalizes buckets to N examples (upsamples minority, downsamples majority)
- `--holdout` excludes benchmark source sessions

### Dedup (Near-Duplicate Detection)

```bash
python -m src.cli dedup --threshold=0.85
```

- MinHash + LSH based
- Threshold: 0.95 (strict) / 0.85 (default) / 0.75 (loose)

### Profile (Dataset Statistics)

```bash
python -m src.cli profile
python -m src.cli profile --out=report.json
```

- Token length distributions, language detection, code complexity, topic clustering

### Auto-Balance

```bash
python -m src.cli auto-balance --cap=500
```

- Weighted balancing across task buckets
- Weights: multi-file-refactor 2.5x, file-edit/data-analysis 2.0x, shell/code-search 1.5x, debug/docs 1.0x, reasoning 0.5x

### Dataset Versioning

```bash
python -m src.cli dataset-version-create --label=combined
python -m src.cli dataset-version-list
python -m src.cli dataset-version-diff --v1=<id> --v2=<id>
python -m src.cli dataset-version-restore --v2=<id>
```

### Pre-tokenize

```bash
python -m src.cli pretokenize --label=combined --max-length=2048
```

- Batch tokenize to Arrow/Parquet for faster training

### Binarize

```bash
python -m src.cli binarize --src=<jsonl> --out=<arrow> --model=<path>
```

- Full pre-tokenization with real base tokenizer
- Consumed via `TRAIN_TOKENIZED_DIR` env var

## Verification Commands

### Verify (Static Replay)

```bash
python -m src.cli verify
python -m src.cli verify --tasks=analysis/auto-tasks.jsonl --out=analysis/verify-report.jsonl
```

- Replays recorded file writes into temp workspace
- Validates task-source linkage and check satisfiability
- CPU-only, never touches `datasets/`

### Verify-Exec (Guarded Execution Replay)

```bash
python -m src.cli verify-exec
python -m src.cli verify-exec --timeout=30
```

- Also runs recorded bash/code commands in isolated temp dir
- Denylist blocks destructive/network commands (`rm -rf`, `sudo`, `ssh`, `curl`, etc.)
- CPU-only, never touches `datasets/`

### Verify-Gap (Diagnose Benchmark Ceiling)

```bash
python -m src.cli verify-gap
```

- Categorizes why failing tasks fail (file_not_materialized vs snippet_missing)

### Audit (Leakage Detection)

```bash
python -m src.cli audit
```

- Confirms no benchmark task instruction appears in training data

### Mine Repairs (Contrastive Pairs)

```bash
python -m src.cli mine-repairs --include-commands --all-sessions
```

- Mines in-session error->success self-repairs from failures
- Output: `analysis/repairs.jsonl` (DPO-style pairs)
- `--include-commands`: adds shell-tool self-repairs (26 -> 605 pairs)
- `--all-sessions`: scans every session, not just final-failure ones (605 -> 702)

### Validate Classifier

```bash
python -m src.cli validate-classifier
python -m src.cli validate-classifier --n-fail=15 --n-ok=15
```

- Builds hand-label sheet for heuristic classifier validation

## Data Drift Detection

```bash
python -m src.cli harvest-status
python -m src.cli harvest-plan --min-new=50
```

- `harvest-status`: shows new sessions per source since last harvest
- `harvest-plan`: decides if harvest/train should run based on `min_new_sessions`

## Safety Rules

1. **Never overwrite `datasets/` while training is running.** During training, only run `extract` + `clean` (they touch `raw/` and `cleaned/`, never `datasets/`).
2. **Read source DBs read-only.** The extractor uses `?mode=ro` URI parameter.
3. **All outputs go to local `/media/scott/data`.** Never write to network mounts.
4. **Confirm GPU health before/after**: `rocm-smi --showuse` and `df -h /media/scott/data`.

## Quick Reference: Full Harvest Pipeline

```bash
# 1. Extract new sessions
python -m src.cli extract --label=ssd
python -m src.cli hermes

# 2. Clean
python -m src.cli clean

# 3. Analyze (CPU-only, safe during training)
python -m src.cli analyze

# 4. Format (ONLY when training is idle)
python -m src.cli format --all-split
python -m src.cli combine

# 5. Create held-out splits
for L in ssd nas5-main nas5-20260717 opencode-all opencode-portfolio hermes-reasoning combined; do
  python -m src.cli eval-split --label=$L --frac=0.1 || true
done

# 6. Build benchmark tasks
python -m src.cli bench-build
```
