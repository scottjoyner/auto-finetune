# auto-finetune

A pipeline that harvests real agent-coding sessions and turns them into a
finetuning dataset for a coding-agent model.

It currently ingests **opencode** session databases (the SQLite stores that
opencode.ai writes) and is structured to later add **Hermes** agent sessions.
Conversations are extracted resiliently (the source DBs have scattered page
corruption), cleaned, formatted into standard templates, and trained with
[Unsloth](https://github.com/unslothai/unsloth) (CUDA) or HuggingFace PEFT +
bitsandbytes (CUDA **and** AMD/ROCm) via QLoRA.

## Layout

```
auto-finetune/
├── config.yaml            # all knobs (sources, cleaning, formatting, training)
├── src/
│   ├── config.py          # loads config.yaml
│   ├── db.py              # resilient read of corrupt SQLite DBs (row-by-row)
│   ├── extract_opencode.py# pull sessions/messages/parts -> data/raw
│   ├── extract_hermes.py  # (stub) pull Hermes session exports
│   ├── clean.py           # normalize, redact secrets, dedupe
│   ├── format_dataset.py  # rebuild conversations -> chatml/alpaca/sharegpt
│   ├── train.py           # Unsloth QLoRA training
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
| Train | `python -m src.cli train` | Unsloth QLoRA finetune on the formatted dataset |

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

- **CUDA** — install Unsloth for the fastest path:
  `pip install unsloth`. The trainer auto-detects it.
- **AMD / ROCm** — Unsloth has no ROCm build, so the trainer falls back to the
  PEFT backend (HuggingFace PEFT + bitsandbytes, which *does* support ROCm).
  Install a ROCm torch build first:

  ```bash
  # example for ROCm 6.x on a Strix Point / 890M iGPU
  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
  pip install peft transformers trl datasets bitsandbytes
  # export if not auto-detected
  export ROCM_PATH=/opt/rocm
  ```

Set `train.backend` in `config.yaml` to `auto` (default), `unsloth`, or `peft`.
`auto` uses Unsloth on CUDA and PEFT everywhere else.

Extraction, cleaning and formatting run on CPU. The package `apsw` (bundled
SQLite) is used for resilient reads.

## This machine (Ryzen AI 9 HX 370 / Radeon 890M, 24 cores, 91 GB RAM)

- No ROCm torch is installed by default. Install the ROCm wheel above, run
  `python -m src.smoke_rocm`, then `python -m src.cli train`.
- You can validate the whole data pipeline without a GPU:
  `python -m src.cli train --dry-run` (downloads only the tokenizer).
- Gated models (e.g. `Qwen/Qwen3-8B-Instruct`) need `hf auth login` first;
  the default `Qwen/Qwen2.5-7B-Instruct` is open and downloads directly.

## Notes

- Secrets/tokens in tool inputs and outputs are redacted by default
  (`clean.redact_secrets`).
- Reasoning parts are dropped from training targets unless
  `clean.keep_reasoning_as_context` is set.
- Long sessions are split into sliding windows (`format.max_chars_per_example`,
  default 24k chars ≈ 8k tokens) so every example fits `train.max_seq_length`.
  The source 101 sessions expand to ~1.3k bounded examples this way.
- The default base model is `Qwen/Qwen2.5-7B-Instruct`; swap for a smaller or
  larger instruct model depending on your GPU.
