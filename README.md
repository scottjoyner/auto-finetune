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
├── post-queue.sh          # post-training automation: eval -> best -> probe -> merge -> validate -> benchmark
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
│   └── cli.py             # `python -m src.cli <command>`
└── data/
    ├── raw/               # extracted JSON per session
    ├── cleaned/           # normalized conversations
    └── datasets/          # final formatted train.jsonl
```

## Stages

| Stage | Command | What it does |
|-------|---------|--------------|
| Extract | `python -m src.cli extract` | Reads opencode DBs row-by-row, writes `data/raw/<session>.json` |
| Clean | `python -m src.cli clean` | Redacts secrets, drops empty turns, dedupes |
| Format | `python -m src.cli format` | Reconstructs conversations into a chat template |
| Train | `python -m src.cli train` | Unsloth/PEFT QLoRA finetune on the formatted dataset |
| Eval | `python -m src.cli eval-all` | Held-out loss + tool-call correctness table |
| Best | `python -m src.cli best --metric=loss` | Pick the winning adapter |
| Probe | `python -m src.cli probe --label=<x>` | Qualitative tool-call check (base vs adapter) |
| Merge | `python -m src.cli merge --label=<x>` | Fuse LoRA into a standalone model |
| Benchmark | `python -m src.cli bench` / `bench-matrix` | Agentic "did-the-task-get-done?" benchmark |
| All | `./launch-next.sh --loop` then `./post-queue.sh` | Full train→eval→merge→benchmark automation |

> **Data harvesting (extract → clean → format) is documented separately in
> [`HARVEST.md`](HARVEST.md)** — including how to harvest on CPU *while* a GPU
> training run is live without disturbing it.

## The agentic benchmark (`src/bench.py`)

`eval`/`loss` only say *"does it look like the training data?"*. The benchmark
answers *"did the task actually get done?"* — it drives a **model** (not a
dataset row) through a real multi-turn tool-use loop in a throwaway sandbox and
verifies the outcome with concrete checkers (`file_exists`, `file_contains`,
`command_exit`, `command_output`, …).

### Five runners

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

### Subagent as an MCP/ACP server

`src/subagent.py` wraps the optimized loop as an MCP/ACP-over-stdio server
(stdlib-only JSON-RPC 2.0 — no `mcp` SDK needed). opencode (`opencode acp`) or
hermes can connect and call the `run_task` tool, delegating real tasks to the
RefinedToolCallV5 loop:

```bash
python -m src.subagent --model=/path/to/RefinedToolCallV5-3b [--variant base|finetune|auto]
```

## Data model (opencode DB)

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

### Dealing with corruption

The migrated DBs (`opencode.db`, `opencode_old_broken.db`, …) report
`database disk image is malformed` on bulk scans because a handful of pages are
damaged. The extractor works around this by reading **one rowid at a time** with
a fresh statement, so a single bad page only drops a few rows instead of
aborting the whole export. In practice >99.99% of rows are recovered.

## Configuring sources

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

## This machine (Ryzen AI 9 HX 370 / Radeon 890M, 24 cores, 91 GB RAM)

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
