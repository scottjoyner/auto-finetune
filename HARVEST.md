# Harvesting procedure (data prep, CPU-only)

"Harvesting" is the data-collection half of the pipeline: pull real agent
sessions out of the opencode / Hermes session stores, normalize them, and turn
them into training datasets. It is **pure CPU work** — no GPU — so it is the
right thing to run *while a training run occupies the GPU*.

```
extract  →  clean  →  format  →  combine  →  eval-split   →  (training consumes the datasets)
 (raw)      (cleaned) (datasets/*.jsonl)            (eval/held-out-<label>.jsonl)
```

This file documents the *procedure*, the *safety rules* learned the hard way,
and a *future extension* (task-bucket classification). See `README.md` for the
DB schema and the corruption workaround.

---

## 0. Environment

```bash
export PATH="/media/scott/data/finetune-venv/bin:$PATH"
export PYTHONPATH=/home/scott/git/auto-finetune
export HF_HOME=/media/scott/data/finetune-staging/hf-home
export TMPDIR=/media/scott/data/finetune-staging/tmp
cd /home/scott/git/auto-finetune
```

All outputs land on the **local** data drive (`/media/scott/data/finetune-staging/data/{raw,cleaned,datasets}`),
never on the network mounts, so harvesting never contends with training's
checkpoint writes.

---

## 1. Where the data lives (and what is worth reading)

| Source | Path | New data? | Notes |
|--------|------|-----------|-------|
| Live opencode | `/media/scott/SSD_4TB/opencode/opencode.db` (26 GB) | **Yes** | The only growing opencode store. Corrupt (`database disk image is malformed`) — `CorruptDB` reads it row-by-row. |
| Live Hermes | `/media/scott/SSD_4TB/hermes-home/.hermes/state.db` (1.2 GB) | **Yes** | Hermes canonical store; harvest with the `hermes` command. |
| NAS5 snapshots | `/media/scott/NAS5/opencode/opencode*.db` (157 GB + 55 GB + …) | **No** | Static (last modified Jul 13–17). Already harvested. Re-reading hammers NAS5 for zero new sessions — **skip**. |

**Key rule:** the *only* sources of genuinely new sessions are the two live
`SSD_4TB` DBs. Target them; do not re-scan the NAS5 snapshots.

Both live DBs are in **WAL mode** (`opencode.db-wal` / `-shm` present), so a
read-only extractor does **not** block opencode's own writes to your live
session store. Reading them is safe.

---

## 2. Safety rules (GPU training must stay live)

1. **Never overwrite `datasets/` while a training run is reading it.**
   HuggingFace jsonl loaders memory-map the file; replacing
   `train.<label>.jsonl` mid-run can crash the live training. So:

   - **During training:** run only `extract` + `clean` (they touch `raw/` and
     `cleaned/`, never `datasets/`).
   - **When training is idle:** run `format` / `combine` / `eval-split` to
     actually produce the `train.*.jsonl` files.

2. **Read the source DBs read-only; never write to a network mount.** All
   harvested artifacts go to local `/media/scott/data`.

3. **Confirm the GPU is live and there is disk headroom** before/after:
   ```bash
   rocm-smi --showuse 2>/dev/null | grep -A1 "GPU use"   # expect ~100%
   df -h /media/scott/data                                # ~588G free as of last run
   ```

---

## 3. Commands (exact)

### 3a. During training — harvest + clean only (safe)

```bash
# opencode: just the live SSD_4TB store (skip the static NAS5 snapshots)
python -m src.cli extract --label=ssd

# Hermes: the live state.db
python -m src.cli hermes

# Normalize everything in raw/ -> cleaned/  (~1 min for ~2500 sessions)
python -m src.cli clean
```

`extract --label=ssd` writes `raw/ssd/`; `hermes` writes `raw/hermes_*.json`.
`clean` emits `cleaned/` (flat, mostly Hermes) plus per-source subdirs
(`ssd`, `nas5-*`, `opencode-portfolio`, `hermes-reasoning`).

> If a prior harvest already exists, `extract` is idempotent (files are named
> by session id); it adds only sessions created since the last pass. The live
> opencode store is usually already caught up; the live Hermes store is where
> the new sessions appear.

### 3b. When training is idle — turn cleaned data into datasets

```bash
python -m src.cli format                 # train.jsonl (merged) + train.<subdir>.jsonl
python -m src.cli format --source=hermes # train.hermes.jsonl
python -m src.cli format --label=opencode-all   # train.opencode-all.jsonl (merged opencode subdirs)
python -m src.cli combine                # train.combined.jsonl (deduped union of all labels)
# carve held-out splits the evaluator expects, per label:
for L in ssd nas5-main nas5-20260717 nas5-old-broken nas5-recover-old \
         opencode-portfolio hermes-reasoning opencode-all combined; do
  python -m src.cli eval-split --label=$L --frac=0.1 || true
done
```

`launch-next.sh` then trains each label in turn
(`ssd → nas5-main → nas5-20260717 → opencode-all → opencode-portfolio →
hermes-reasoning → combined`) and `post-queue.sh` finishes with a
`bench-matrix --preset=all` sweep.

---

## 4. Verify a harvest

```bash
echo "raw/ssd:       $(ls /media/scott/data/finetune-staging/data/raw/ssd 2>/dev/null | wc -l)"
echo "raw/hermes:    $(ls /media/scott/data/finetune-staging/data/raw/hermes_* 2>/dev/null | wc -l)"
echo "cleaned:       $(ls /media/scott/data/finetune-staging/data/cleaned/*.json 2>/dev/null | wc -l)"
echo "datasets (untouched during training):"
ls /media/scott/data/finetune-staging/data/datasets/*.jsonl | xargs -n1 basename | tr '\n' ' '
```

---

## 5. Future: classify sessions into task buckets

The current buckets are by **source** (`ssd`, `nas5-main`, `hermes-reasoning`,
…). A more useful axis is **task type** — what the agent was actually doing —
so we can train or weight specialized corpora (e.g. a "shell-heavy" adapter, a
"multi-file refactor" adapter) and build a balanced combined corpus.

This is doable on CPU with a **deterministic heuristic classifier** (no LLM
needed while the GPU is busy):

- **Features per cleaned session:** tool-call histogram (`bash`, `read`,
  `write`, `edit`/`patch`, `grep`/`glob`, `web`/`fetch`, `python`, …); file
  extensions touched (`*.py`, `*.md`, `*.json`, `*.csv`, `*.sh`, …); the
  user's intent keywords in the first message/title (`fix`, `debug`,
  `refactor`, `implement`, `search`, `explain`, `test`, …); turn count;
  tool-call diversity; presence of error/traceback text (failure signal).
- **Buckets:** `shell`, `file-edit`, `multi-file-refactor`, `code-search`,
  `debug`, `web-research`, `data-analysis`, `docs`, `reasoning/math`,
  `mixed`/`general`.
- **Output:** tag each session with `task_type` and emit
  `train.<bucket>.jsonl` (into a *staging* dir during training, not
  `datasets/`, to respect rule §2.1), plus a stats report (bucket sizes,
  overlap, dedup rate).

This is implemented today as **`python -m src.cli analyze`** (`src/analyze.py`).
It is fully CPU-safe and writes a staging dir (default
`<data>/analysis`, **never `datasets/`**) with:

- `buckets.json` — `session_id → {source, bucket, difficulty, keep, quality_reason}`
- `corpus.json` — aggregate stats (bucket/source/difficulty counts, top tools &
  file-types, avg turns, error rate, dedup rate, opencode↔hermes overlap)
- `auto-tasks.jsonl` — benchmark tasks mined from successful file-edit sessions
- `failures.jsonl` — sessions containing error/traceback text (candidate
  negative-mining set)

Turn the manifest into training corpora with **`python -m src.cli strata`**
(`src.format_dataset.emit_strata`, also CPU-safe, writes to staging only):

- emits `train.<bucket>.jsonl` per task-bucket into `<data>/analysis`
- with `--balance [--cap=N]` it equalizes every bucket to `N` examples
  (upsampling small buckets, **stride-downsampling** the dominant `debug`/
  `reasoning` buckets) and also writes a combined `train.balanced.jsonl`.
- with `--holdout=<tasks.jsonl>` it excludes the source sessions behind
  mined auto-tasks (recovered from each `task_id`) so the benchmark
  is a true held-out of the training corpus.

This directly addresses the merged corpus being dominated by `reasoning`+`debug`:
the actionable tool-use buckets (`file-edit`, `data-analysis`, `code-search`, …)
are otherwise a tiny minority. The balanced set is a ready-to-train staging
artifact — promote it into `datasets/` only when the GPU is idle (see §4).

### Verify the mined tasks (`verify`)

`python -m src.cli verify` (`src/verify.py`, CPU-only, **safe** — it
executes no shell/code, only replays recorded file writes) replays each
`auto-tasks.jsonl` entry's source session into a temp workspace and runs the
`file_contains` checks. It validates two things end-to-end: the
task↔source-session linkage, and that the mined check is actually satisfiable
by the recorded solution. Output: `verify-report.jsonl` + a pass-rate summary.

On the live corpus this surfaces a real data-quality signal: ~61% of mined
tasks reconstruct from structured `write_file`/`edit`/`patch` tool calls; the
rest create files via bash heredocs / `process` and need an **execution
sandbox** to verify (out of scope for the static replay). The pass-rate is a
trackable metric — re-run after any `derive_task` change.

### Exec replay (`verify-exec`, opt-in, guarded)

`python -m src.cli verify-exec` (`src/verify_exec.py`) goes one step
further: after the static replay it ALSO runs the recorded
`bash`/`terminal`/`execute_code` tool calls in an **isolated temp dir**
(cwd = throwaway dir, deleted after; per-command timeout). A denylist
refuses to run destructive or network-egress patterns (`rm -rf`, `sudo`,
`dd`, `mkfs`, `git push`, `curl`, `wget`, `ssh`, `scp`, `pip`/`npm`/
`apt`, `/media` writes, …) — those tasks are reported as `blocked`
rather than executed. It is CPU-only (no GPU) and never touches
`datasets/`.

**Honest ceiling:** on the live 80-task corpus, `verify-exec` does NOT
raise the pass-rate above 61%. The remaining 31 tasks create their
files at **absolute / remote paths** (`/home/.../fleet.py`, `ssh`/
`docker`/`curl`) which a cwd-isolated replay cannot relocate into the
check workspace. Reaching 100% needs a real **container sandbox** that
bind-mounts the original project directories (so absolute writes land
inside the sandbox) — deliberately out of scope here. The guarded
executor is still useful for future harvests whose tasks use relative
shell/python file creation.

Later, when the GPU is free, an LLM-based classifier (or a small fine-tuned
tagger) can replace the heuristics for finer buckets. The heuristic pass is
the right first iteration and is fully CPU-safe.

### Turn the verifiable tasks into a real benchmark (`bench-build`)

This is the payoff of the whole harvest: a **"did the task actually get
done?"** benchmark set derived from real operator sessions, consumable by
the agentic bench harness (`src/bench.py`).

`python -m src.cli bench-build` (`src.bench.build_auto_bench`) reads
`analyze`'s `auto-tasks.jsonl` and (when present) `verify-report.jsonl`,
and writes **only the statically-verifiable subset** (tasks whose verify
`ok` is `True`, i.e. the 49/80 that create a relative file we can check)
into `eval/tasks/auto-verified.jsonl` in `bench.Task` format:

```bash
python -m src.cli bench-build \
  --tasks=/media/scott/data/finetune-staging/data/analysis/auto-tasks.jsonl \
  --verify-report=/media/scott/data/finetune-staging/data/analysis/verify-report.jsonl \
  --out=eval/tasks/auto-verified.jsonl
# -> [bench-build] wrote 49 verifiable tasks -> eval/tasks/auto-verified.jsonl
```

The bench harness drives a **model** through a real multi-turn tool loop in
a throwaway sandbox (root = temp dir, deleted after) and then verifies the
outcome with the same `file_contains` checks. Crucially, the sandbox's
`bash` tool now reuses the **same denylist as `verify-exec`** (refuses
`rm -rf`, `sudo`, `dd`, `ssh`, `curl`, `/media`, …) so grading a model's
*own* shell commands is guarded, not just the operator's history.

Run it across any model/runner (CPU or GPU) once a model is free:

```bash
# a local HF checkpoint (base / finetune / merged), CPU or ROCm
python -m src.cli bench --runner=self --model=/path/to/ckpt \
  --tasks=eval/tasks/auto-verified.jsonl

# or a fleet / lmstudio reference via the OpenAI-compatible api
python -m src.cli bench --runner=api --fleet \
  --tasks=eval/tasks/auto-verified.jsonl

# or a full matrix (local-refs | lmstudio | fleet | fast | all)
python -m src.cli bench-matrix --preset=fast \
  --tasks=eval/tasks/auto-verified.jsonl --report
```

**Why this is the real "done?" signal:** `verify`/`verify-exec` grade the
*recorded operator solution*. `bench` grades *any model you point at it* on
the same tasks — exactly the capability claim of the tool-caller LoRA. The
gold ceiling is 49/49 (these tasks are verifiable by construction); a 3B
finetune scoring well above a base model on this set is the deployment bar.

**Run when:** these are model-inference ops (GPU or a CPU model). Building
the set + the harness hardening here are CPU-only and safe to do while
training runs; only the actual `bench`/`bench-matrix` runs need a model.

### Mine failures into contrastive repair pairs (`mine-repairs`)

`analyze` also writes `failures.jsonl` (the 1338 sessions that ended in
error). Those are free training signal: a model that *avoids* the failing
move and *makes* the successful one is a more robust tool-caller.

`python -m src.cli mine-repairs` (`src.contrast.mine_repairs`) walks each
failure, reconstructs its (call -> result) timeline (the cleaned records
store only the call; the result is the next message's text), and looks for an
**in-session self-repair** — an error step on a target file followed, later
in the *same* session, by a successful step on that same target. That pair
is emitted as a DPO-style record:

```jsonc
{"session": "...", "bucket": "debug", "error_tool": "write_file",
 "target": "fleet.py", "error_marker": "error:",
 "prompt_messages": [ /* convo up to, but excluding, the erroneous call */ ],
 "rejected_call": {"name": "write_file", "arguments": {...}},
 "chosen_call":  {"name": "edit",       "arguments": {...}}}
```

```bash
python -m src.cli mine-repairs \
  --cleaned=/media/scott/data/finetune-staging/data/cleaned \
  --failures=/media/scott/data/finetune-staging/data/analysis/failures.jsonl \
  --out=/media/scott/data/finetune-staging/data/analysis/repairs.jsonl
# -> 26 contrastive repair pairs (file-targeted only, on a 1249-failure corpus)
python -m src.cli mine-repairs ... --include-commands
# -> 605 pairs (adds shell-tool self-repairs: 519 terminal + 107 bash + 75 execute_code)
```

It also prints a **failure taxonomy** (which markers / tools / buckets fail
most) so we know where the model needs the most help.

**Honest scope:** by default only **file-targeted** tools (write/edit/patch/…)
are matchable across steps. With `--include-commands`, shell-tool
self-repairs are also mined — an errored command followed (later, same
session) by a *successful* call to the same tool with different arguments
is a genuine command self-repair, and on the live corpus that lifts the
pair count from 26 to 605. Read/search/web calls are intentionally
excluded (they rarely self-correct into a better same-tool call). The 605
are a clean seed for a future DPO pass; `train` is currently SFT-only,
so wire a preference stage (or turn `chosen_call` into an SFT "repair"
example) before using them.

**Run when:** CPU-only and safe to run while training runs.

### Grow the repair pairs (`--all-sessions`)

`mine-repairs --all-sessions` scans **every** cleaned session (not just
final-failure ones) for an in-session error→success self-repair, which
adds genuine corrections that happened before the session ultimately
recovered. On the live corpus this lifts the pair count from **605 → 702**
(file-targeted + command self-repairs). `bucket` defaults to `` for
non-failure sessions. Rebuild the DPO mix afterward:

```bash
python -m src.cli mine-repairs --all-sessions --include-commands \
  --out=/media/scott/data/finetune-staging/data/analysis/repairs.jsonl
python -c "from src.repair_mix import build_dpo_mix; \
  build_dpo_mix('/media/scott/data/finetune-staging/data/analysis/repairs.jsonl', \
                 '/media/scott/data/finetune-staging/launch/repairs/repairs.dpo.jsonl')"
```

### Diagnose the benchmark ceiling (`verify-gap`)

`analyze` mines 80 auto-tasks; `verify` can only statically replay
file writes, so exactly 31 tasks can never pass (the 61.3% ceiling).
`verify-gap` (`src/verify_gap.py`) categorizes **why** each failing
task fails, from `verify-report.jsonl`:

```bash
python -m src.cli verify-gap
# -> 26 file_not_materialized (absolute/remote path not replayed)
# ->  5 snippet_missing       (file present, expected snippet absent)
```

`file_not_materialized` = the expected file was created at an absolute or
remote path the sandbox can't relocate; `snippet_missing` = a likely
check-extraction bug. **Extending past the ceiling** needs either a
container sandbox that bind-mounts the original dirs (so absolute writes
land inside it) or new safe check kinds — that verifier work is left for
a dedicated pass (it trades on execution safety).

### Leakage audit (`audit`)

Confirms no benchmark task's instruction appears as a substring of any
training example (content-level leakage beyond the session hold-out):

```bash
python -m src.cli audit
# -> [audit] train=10000 bench=49 hits=0  (no leakage)
```

### Near-duplicate detection (`dedup`)

`python -m src.cli dedup` (`src/dedup.py`) uses MinHash + LSH to find
near-duplicate sessions that differ only in whitespace, formatting, or
minor edits. This is CPU-only and safe to run while training.

```bash
python -m src.cli dedup --threshold=0.85
# [dedup] 2500 sessions, threshold=0.85
# [dedup] removed 180 exact session_id duplicates
# [dedup] kept 2320 sessions, removed 45 near-duplicates
```

The threshold controls similarity (0.0-1.0):
- `0.95` = very strict (only near-identical)
- `0.85` = default (good balance)
- `0.75` = loose (catches more variants)

### Dataset profiling (`profile`)

`python -m src.cli profile` (`src/profile.py`) computes token length
distributions, language detection, code complexity metrics, and topic
clustering. Fully CPU-safe.

```bash
python -m src.cli profile
# [profile] profiling 2320 sessions
# [profile] 2320 sessions, 4500000 tokens
# [profile] avg tokens: 1940, median: 1650
# [profile] languages: {'python': 1200, 'javascript': 450, 'text': 670}
```

### Auto-balancing (`auto-balance`)

`python -m src.cli auto-balance` (`src/auto_balance.py`) uses the bucket
analysis to create balanced training datasets by upsampling minority
classes and downsampling majority classes.

```bash
python -m src.cli auto-balance --cap=500
# [auto-balance] loaded 2320 sessions
# [auto-balance] sampled 1800 sessions across 10 buckets
#   file-edit: 200
#   multi-file-refactor: 150
#   shell: 180
#   debug: 250
```

Bucket weights:
- `multi-file-refactor`: 2.5x (highest priority)
- `file-edit`, `data-analysis`: 2.0x
- `shell`, `code-search`: 1.5x
- `debug`, `docs`: 1.0x
- `reasoning`: 0.5x (downsampled)

### Dataset versioning (`dataset-version`)

`python -m src.cli dataset-version-create` (`src/dataset_version.py`)
creates versioned snapshots of datasets with content hashing for
reproducibility.

```bash
python -m src.cli dataset-version-create --label=combined
# [dataset-version] created v20260720-220000
#   label: combined
#   examples: 10000
#   hash: e3b0c44298fc1c14...

python -m src.cli dataset-version-list
# [dataset-version] 3 versions:
#   v20260720-220000 [combined] 10000 examples 45.2MB
#   v20260719-180000 [combined] 9500 examples 42.1MB
#   v20260718-120000 [combined] 9000 examples 39.8MB
```

### Pre-tokenize the mix (`pretokenize`)

`python -m src.cli pretokenize` (`src/pretokenize.py`) renders the
dataset through the chat template and tokenizes it **once** with the
real base tokenizer, saving a pre-tokenized Arrow/Parquet dataset.

```bash
python -m src.cli pretokenize --label=combined --max-length=2048
# [pretokenize] loading tokenizer: Qwen/Qwen2.5-7B-Instruct
# [pretokenize] 10000 sessions, max_length=2048
# [pretokenize] tokenized 10000 sessions
# [pretokenize] 4500000 total tokens, 450 avg per session
# [pretokenize] output: /path/to/pretokenized/tokenized.parquet
```

### Pre-tokenize the mix (`binarize`)

`binarize` (`src/binarize.py`) renders the 10k mix through the chat
template and tokenizes it **once** with the real base tokenizer,
saving a pre-tokenized Arrow dataset (`input_ids`/`attention_mask`/
`labels`, full-sequence supervision) so the focused `train` step doesn't
re-tokenize every epoch. `train` consumes it via `TRAIN_TOKENIZED_DIR`
(or `main(tokenized_dir=...)`); the default JSONL path is unchanged.

```bash
python -m src.cli binarize \
  --src=/media/scott/data/finetune-staging/data/analysis/train.balanced.jsonl \
  --out=/media/scott/data/finetune-staging/data/analysis/train.balanced.arrow \
  --model=/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b
# then later:  TRAIN_TOKENIZED_DIR=.../train.balanced.arrow python -m src.cli train --label=...
```

### Base-vs-adapter benchmark diff (`bench-compare`)

`bench-compare` (`src/bench.py`) runs the held-out 80-task suite on a
**base model** and on a **trained adapter**, then prints the delta in
task / check pass-rate. Pure CPU until a driver loads a GPU model, so the
harness is ready the moment an adapter exists.

```bash
python -m src.cli bench-compare \
  --base=/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b \
  --adapter=/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-<label> \
  --tasks=eval/tasks/auto-verified.jsonl
```

