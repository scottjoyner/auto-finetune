# auto-finetune

A pipeline that harvests real agent-coding sessions and turns them into a
finetuning dataset for a coding-agent model — then **evaluates** and
**benchmarks** the result on a genuine agentic task-completion test.

It ingests **opencode** session databases (the SQLite stores that opencode.ai
writes) and **Hermes** agent session exports. Conversations are extracted
resiliently (the source DBs have scattered page corruption), cleaned, formatted
into standard templates, trained with [Unsloth](https://github.com/unslothai/unsloth)
(CUDA) or HuggingFace PEFT + bitsandbytes (CUDA **and** AMD/ROCm) via QLoRA,
and finally measured on a held-out loss + an "did the task actually get done?"
agentic benchmark.

## Layout

```
auto-finetune/
├── config.yaml            # all knobs (sources, cleaning, formatting, training)
├── launch-next.sh         # hands-off chained finetune runner
├── post-queue.sh          # post-training: eval -> best -> probe -> merge -> validate -> benchmark
├── src/
│   ├── config.py          # loads config.yaml
│   ├── db.py              # resilient read of corrupt SQLite DBs (row-by-row)
│   ├── extract_opencode.py# pull sessions/messages/parts -> data/raw
│   ├── extract_hermes.py  # pull Hermes session exports
│   ├── clean.py           # normalize, redact secrets, dedupe
│   ├── format_dataset.py  # rebuild conversations -> chatml/alpaca/sharegpt
│   ├── train.py           # Unsloth/PEFT QLoRA training
│   ├── eval.py            # held-out loss + tool-call scoring + curated probes
│   ├── merge.py           # fuse a LoRA adapter into a standalone model
│   ├── bench.py           # agentic task-completion harness (5 runners)
│   ├── drivers_localchat.py # standard HF chat-model runner (native tool format)
│   ├── subagent.py        # MCP/ACP-over-stdio server wrapping the optimized loop
│   ├── fleet.py           # read-only lan fleet-router helper (large references)
│   │
│   │  # CPU-heavy data processing
│   ├── dedup.py           # MinHash + LSH near-duplicate detection
│   ├── profile.py         # token stats, language detection, topic clustering
│   ├── pretokenize.py     # batch tokenize to Arrow/Parquet format
│   ├── auto_balance.py    # weighted balancing from bucket analysis
│   ├── dataset_version.py # version datasets with content hashing
│   │
│   │  # Analysis and verification
│   ├── analyze.py         # task bucket classification, difficulty, quality flags
│   ├── contrast.py        # mine contrastive repair pairs from failures
│   ├── verify.py          # static replay verification
│   ├── verify_exec.py     # guarded execution replay
│   ├── verify_gap.py      # diagnose benchmark ceiling
│   ├── audit.py           # training/benchmark leakage detection
│   ├── binarize.py        # pre-tokenize to Arrow format
│   │
│   │  # Auto-harvester pipeline
│   ├── harvest.py         # data drift detection, batch planning
│   ├── scheduler.py       # orchestrate harvest-train-deploy cycle
│   ├── deploy.py          # single/multi-node deployment with health checks
│   ├── registry.py        # model version registry with lineage
│   │
│   │  # Operations
│   ├── notify.py          # desktop/webhook/email alerts
│   ├── metrics.py         # training metrics tracking, regression detection
│   ├── quantize.py        # post-merge GPTQ/AWQ quantization
│   ├── cost.py            # GPU hours and resource tracking
│   │
│   └── cli.py             # `python -m src.cli <command>`
└── data/
    ├── raw/               # extracted JSON per session
    ├── cleaned/           # normalized conversations
    ├── datasets/          # final formatted train.jsonl
    └── analysis/          # buckets, tasks, metrics, registry
```

## Quick Start

```bash
# Full pipeline (extract -> clean -> train -> eval -> deploy)
python -m src.cli all

# Or run the continuous auto-harvester loop
python -m src.cli scheduler-loop --interval=3600

# Manual steps
python -m src.cli extract --label=ssd
python -m src.cli hermes
python -m src.cli clean
python -m src.cli analyze
python -m src.cli auto-balance --cap=500
python -m src.cli format
python -m src.cli train --label=combined
python -m src.cli eval-all
python -m src.cli merge --label=combined
python -m src.cli deploy --label=combined --nodes=local,nas5
```

## Stages

### Data Pipeline (CPU-only)

| Stage | Command | What it does |
|-------|---------|--------------|
| Extract | `python -m src.cli extract` | Reads opencode DBs row-by-row, writes `data/raw/<session>.json` |
| Clean | `python -m src.cli clean` | Redacts secrets, drops empty turns, dedupes |
| Dedup | `python -m src.cli dedup --threshold=0.85` | MinHash + LSH near-duplicate detection |
| Profile | `python -m src.cli profile` | Token stats, language detection, topic clustering |
| Analyze | `python -m src.cli analyze` | Task bucket classification, difficulty scoring |
| Balance | `python -m src.cli auto-balance --cap=500` | Weighted balancing across task buckets |
| Format | `python -m src.cli format` | Reconstructs conversations into a chat template |
| Pretokenize | `python -m src.cli pretokenize` | Batch tokenize to Arrow/Parquet format |
| Version | `python -m src.cli dataset-version-create` | Version dataset snapshots for reproducibility |

### Training

| Stage | Command | What it does |
|-------|---------|--------------|
| Train | `python -m src.cli train` | Unsloth/PEFT QLoRA finetune on the formatted dataset |
| Eval | `python -m src.cli eval-all` | Held-out loss + tool-call correctness table |
| Best | `python -m src.cli best --metric=loss` | Pick the winning adapter |
| Probe | `python -m src.cli probe --label=<x>` | Qualitative tool-call check (base vs adapter) |
| Merge | `python -m src.cli merge --label=<x>` | Fuse LoRA into a standalone model |
| Quantize | `python -m src.cli quantize --label=<x> --bits=4` | GPTQ/AWQ quantization for faster inference |

### Deployment

| Stage | Command | What it does |
|-------|---------|--------------|
| Deploy | `python -m src.cli deploy --label=<x> --nodes=n1,n2` | Deploy to multiple inference nodes |
| Status | `python -m src.cli multi-deploy-status` | Check deployment status across nodes |
| Rollback | `python -m src.cli rollback --label=<x>` | Revert to previous model version |

### Operations

| Stage | Command | What it does |
|-------|---------|--------------|
| Notify | `python -m src.cli notify --event=<x> --message=<y>` | Send alerts (desktop/webhook/email) |
| Metrics | `python -m src.cli metrics-record --label=<x>` | Track training metrics over time |
| Regression | `python -m src.cli metrics-regression --label=<x>` | Detect performance regressions |
| Cost | `python -m src.cli cost-record --label=<x> --hours=8` | Track GPU hours and resource usage |

### Auto-Harvester

| Stage | Command | What it does |
|-------|---------|--------------|
| Status | `python -m src.cli harvest-status` | Check data drift and new sessions |
| Plan | `python -m src.cli harvest-plan --min-new=50` | Decide if harvest/train should run |
| Scheduler | `python -m src.cli scheduler-run` | Run one complete harvest-train-deploy cycle |
| Loop | `python -m src.cli scheduler-loop --interval=3600` | Continuous monitoring and automation |

## Auto-Harvester Pipeline

The auto-harvester orchestrates the full lifecycle autonomously:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCHEDULER LOOP                               │
├─────────────────────────────────────────────────────────────────┤
│  harvest-status → harvest-plan → harvest → train → eval →      │
│  merge → quantize → multi-deploy → notify                      │
├─────────────────────────────────────────────────────────────────┤
│  metrics-track → cost-record → dataset-version-create          │
└─────────────────────────────────────────────────────────────────┘
```

### Data Drift Detection

```bash
python -m src.cli harvest-status
# [harvest-status]
#   opencode: 1250 sessions, 125 new since last harvest, 2.3 days ago
#   hermes: 890 sessions, 45 new since last harvest, 1.1 days ago

python -m src.cli harvest-plan --min-new=50
# [harvest-plan] should_harvest=True should_train=True
#   total_new=170, batch=['opencode', 'hermes']
#   reason: opencode: 125 new sessions >= 50; hermes: 45 new sessions >= 50
```

### Multi-Node Deployment

```bash
# Deploy to all discovered nodes
python -m src.cli deploy --label=combined --nodes=local,nas5,laptop

# With quorum requirement
python -m src.cli deploy --label=combined --nodes=local,nas5 --quorum=2

# Check status
python -m src.cli multi-deploy-status
# [multi-deploy-status]
#   local:
#     combined v3 [active] 1250MB
#   nas5:
#     combined v3 [active] 1250MB
```

### Metrics and Regression Detection

```bash
python -m src.cli metrics-record --label=combined --eval-loss=0.42 --tool-exact=0.85

python -m src.cli metrics-regression --label=combined
# [metrics-regression] OK: eval_loss=0.4200 (best=0.4150)

python -m src.cli metrics-compare --label=combined
# [metrics-compare] toolcall-v5-3b-combined-v2 vs toolcall-v5-3b-combined-v3
#   eval_loss: 0.4500 -> 0.4200 (-0.0300, -6.67%)
#   tool_exact_match: 0.8200 -> 0.8500 (+0.0300, +3.66%)
```

## The Agentic Benchmark (`src/bench.py`)

`eval`/`loss` only say *"does it look like the training data?"*. The benchmark
answers *"did the task actually get done?"* — it drives a **model** (not a
dataset row) through a real multi-turn tool-use loop in a throwaway sandbox and
verifies the outcome with concrete checkers (`file_exists`, `file_contains`,
`command_exit`, `command_output`, …).

### Five Runners

| runner | what it is | model source |
| --- | --- | --- |
| `self` | self-contained harness: parses `<tool_call>`, runs bash+file tools in a temp sandbox | a local HF dir (base / finetune / merged) |
| `subagent` | **optimized** loop built for RefinedToolCallV5 variants (per-variant result formatting, stop sequences, error recovery, variant autodetect) | a local HF dir |
| `local-chat` | **standard HF chat model** in its native function-call format (e.g. the cached Qwen2.5-7B) — a genuine local large reference | a local HF chat dir |
| `api` | OpenAI-compatible endpoint (lan fleet router, or a local lmstudio server) | `--base-url` + `--api-model` |
| `hermes` | delegates to the hermes-agent harness (`mini_swe_runner.py`) — Hermes runs its own model+tool loop | whatever Hermes is configured with |

Run it:

```bash
# one model, the local 3B base (needs GPU when not training)
python -m src.cli bench --runner=subagent \
  --model=/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b

# a standard HF large reference (native tool format, CPU by default)
python -m src.cli bench --runner=local-chat \
  --model=/home/scott/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct

# one suite, MANY models -> a single combined comparison table
python -m src.cli bench-matrix --preset=all --report
```

`bench-matrix` presets (no `--specs` needed):

- `local-refs` / `local` — Qwen2.5-7B (transformers) + any finished FT adapters
- `lmstudio` — the q8 `*.gguf` models under `~/.lmstudio/models` (needs the
  lmstudio OpenAI server on `:1234`)
- `fleet` — every model in `~/.config/opencode/endpoints.json`
- `fast` — ONE model per source (cheap smoke gate)
- `all` — **local + lmstudio + fleet combined** (what `post-queue.sh` runs)

## Data Model (opencode DB)

The opencode store is a SQLite DB with these relevant tables:

- `session` — one coding session (id, title, agent, model, project, timestamps).
- `message` — a top-level turn (`role`: user/assistant, `agent`, `model`,
  `parentID` for threading, `time`).
- `part` — content blocks attached to a message. Types seen in the wild:
  - `text` — assistant/user prose
  - `reasoning` — model CoT (often encrypted by the provider)
  - `tool` — a tool call with `state.input` / `state.output`
  - `patch` — a file diff applied
  - `step-start` / `step-finish` — agent step markers + token accounting
  - `compaction` — context-compaction event

### Dealing with Corruption

The migrated DBs (`opencode.db`, `opencode_old_broken.db`, …) report
`database disk image is malformed` on bulk scans because a handful of pages are
damaged. The extractor works around this by reading **one rowid at a time** with
a fresh statement, so a single bad page only drops a few rows instead of
aborting the whole export. In practice >99.99% of rows are recovered.

## Configuring Sources

Edit `config.yaml`:

- `sources.opencode.db_path` — the live DB.
- `sources.opencode.extra_dbs` — older/broken snapshots (toggle `enabled`).
- `sources.hermes.dir` — directory of Hermes session exports (when available).

## Requirements

```bash
pip install -r requirements.txt
```

`train` needs a GPU. The pipeline supports **both CUDA and AMD/ROCm**:

- **CUDA** — install Unsloth for the fastest path: `pip install unsloth`. The
  trainer auto-detects it.
- **AMD / ROCm** — Unsloth has no ROCm build, so the trainer falls back to the
  PEFT backend (HuggingFace PEFT + bitsandbytes, which *does* support ROCm).
  Install a ROCm torch build first:

  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
  pip install peft transformers trl datasets bitsandbytes
  export ROCM_PATH=/opt/rocm   # if not auto-detected
  ```

Set `train.backend` in `config.yaml` to `auto` (default), `unsloth`, or `peft`.
`auto` uses Unsloth on CUDA and PEFT everywhere else.

Extraction, cleaning, formatting, evaluation and the benchmark (on CPU/local
models) run without a GPU. The package `apsw` (bundled SQLite) is used for
resilient reads.

## This Machine (Ryzen AI 9 HX 370 / Radeon 890M, 24 cores, 91 GB RAM)

- No ROCm torch is installed by default. Install the ROCm wheel above, run
  `python -m src.smoke_rocm`, then `python -m src.cli train`.
- You can validate the whole data pipeline without a GPU:
  `python -m src.cli train --dry-run` (downloads only the tokenizer).
- Gated models (e.g. `Qwen/Qwen3-8B-Instruct`) need `hf auth login` first;
  the default `Qwen/Qwen2.5-7B-Instruct` is open and downloads directly.
- GPU recovery after a cold boot: the `amdgpu`/`amdkfd` modules are not
  auto-loaded. `sudo modprobe amdgpu` then verify `/dev/kfd` exists.

## Notes

- Secrets/tokens in tool inputs and outputs are redacted by default
  (`clean.redact_secrets`).
- Reasoning parts are dropped from training targets unless
  `clean.keep_reasoning_as_context` is set.
- Long sessions are split into sliding windows (`format.max_chars_per_example`,
  default 24k chars ≈ 8k tokens) so every example fits `train.max_seq_length`.
- The default base model is `Qwen/Qwen2.5-7B-Instruct`; swap for a smaller or
  larger instruct model depending on your GPU.
- The lmstudio q8 `*.gguf` models (e.g. `RefinedNeuro/RefinedToolCallV5-3b-Q8_0.gguf`)
  need llama.cpp / lmstudio's OpenAI server — use `--runner=api` or
  `--preset=lmstudio`, not the transformers `local-chat` runner.

## Next Steps

See [AUTOHARVEST.md](AUTOHARVEST.md) for the full auto-harvester architecture
and future work.
