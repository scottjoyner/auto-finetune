---
name: auto-finetune-ops
description: Operate xwing auto-finetune trace harvesting, training, and multi-agent coordination safely. Use for Hermes/OpenCode trace ingest, dataset refresh, training status, or concurrent agent work in auto-finetune.
version: 1.0.0
metadata:
  hermes:
    tags: [finetune, opencode, traces, harvesting, multi-agent, rocm]
---

# Auto-Finetune Operations

Use this skill whenever work touches `/home/scott/git/auto-finetune`, live Hermes/OpenCode traces, `/media/scott/data/finetune-staging`, or xwing GPU training.

## Canonical Surfaces

- Main checkout (often owned by OpenCode): `/home/scott/git/auto-finetune`
- Per-agent worktrees: `/media/scott/data/worktrees/auto-finetune-*`
- Runtime staging: `/media/scott/data/finetune-staging`
- Runtime leases: `/media/scott/data/finetune-staging/locks`
- Live OpenCode DB: `/home/scott/.local/share/opencode/opencode.db`
- Live Hermes DB: `/home/scott/.hermes/state.db`
- SSD OpenCode archive: `/media/scott/SSD_4TB/opencode/opencode.db`
- SSD Hermes mirror: `/media/scott/SSD_4TB/hermes-home/.hermes/state.db`

The local databases are the active writers on xwing. SSD copies are durable mirrors/archives and can lag. Discover active writers dynamically before changing config; never infer liveness from filename alone.

## Multi-Agent Invariant

Git worktrees isolate source edits but do not isolate shared datasets, GPU, checkpoints, or deployments. Both are mandatory:

1. Use a dedicated branch/worktree when another Hermes or OpenCode process owns the main checkout.
2. Enter runtime operations through `python -m src.cli ...`; it acquires canonical `flock` leases.

Check ownership first:

```bash
cd /home/scott/git/auto-finetune
/media/scott/data/finetune-venv/bin/python -m src.cli coordination-status
```

Exit code `75` means a resource is busy. Do not bypass it. Owner JSON is diagnostic; the kernel lock is authoritative. A pre-lease trainer is detected as `legacy_trainers` and blocks dataset/GPU operations until it exits.

## Safe Worktree Workflow

```bash
cd /home/scott/git/auto-finetune
git status --short --branch
git worktree add -b fix/<topic> /media/scott/data/worktrees/auto-finetune-<agent>-<topic> HEAD
```

Checkpoint shared WIP only with Scott's permission. Commit granularly in the isolated worktree. Never reset, stash, or overwrite the main checkout while OpenCode owns it.

## Trace Lifecycle

```text
live SQLite (read-only) -> raw per-session JSON -> redacted cleaned JSON
-> immutable/versioned JSONL dataset -> held-out eval -> LoRA checkpoint
```

- OpenCode uses singular table `session` with `time_created/time_updated`; corruption-tolerant APSW reads are required.
- Hermes uses `sessions` + `messages` with `started_at/ended_at`; read live WAL using `mode=ro`, never `immutable=1`.
- Extraction is idempotent by session ID and outputs are atomically promoted.
- Redaction must remain enabled.
- Drift planning fails closed if any configured live source cannot be inspected.
- Harvest baselines store each source's total count atomically, never an aggregate delta.

## Commands

Read-only planning:

```bash
/media/scott/data/finetune-venv/bin/python -m src.cli harvest-status
/media/scott/data/finetune-venv/bin/python -m src.cli harvest-plan --min-new=50
```

Manual harvesting (CPU-only; serialized by `harvest.lock`):

```bash
/media/scott/data/finetune-venv/bin/python -m src.cli extract --label=ssd
/media/scott/data/finetune-venv/bin/python -m src.cli hermes
/media/scott/data/finetune-venv/bin/python -m src.cli clean --keep-reasoning
```

Dataset writers and GPU commands acquire `datasets`/`gpu`/checkpoint leases. Never directly invoke module internals to bypass them.

```bash
python -m src.cli format --label=ssd
python -m src.cli format --source=hermes
python -m src.cli train --label=ssd
python -m src.cli train --source=hermes
```

Do not format/combine/replace datasets while any legacy trainer is active. Training holds a shared dataset lease; dataset writers require exclusive access.

## Knowledge and Skill Synchronization

These are separate planes:

- SQLite session stores are the trace-training source.
- Neo4j is the durable second-brain/index surface.
- Markdown docs remain real source files; index them with `kg_index_docs`, then resolve through `kg_read_doc` before answering.
- Import/reconcile Hermes sessions into KG on demand; use the auto-finetune extractor separately for model data.

```bash
# Dry-run is the default. Add --write only after reviewing counts.
hermes knowledge-graph import-sessions --db /home/scott/.hermes/state.db --since-ts <unix-seconds>
hermes knowledge-graph import-sessions --db /home/scott/.hermes/state.db --since-ts <unix-seconds> --write --no-embed
```

`--no-embed` preserves graph structure when the embedding endpoint is unavailable. The knowledge-graph provider requires the `knowledge-graph` optional dependency (`neo4j`); on xwing it is installed in the Hermes venv. New installs should include `hermes-agent[knowledge-graph]`.
- Installed Hermes skills live on the SSD-backed shared Hermes home, so all agents see the same skill after `/reload-skills` or a new session.
- The repository copy of this skill documents the code version. When behavior changes, patch both the repo copy and the installed skill in one change.

## Verification

```bash
/media/scott/data/finetune-venv/bin/python -m pytest -q -o addopts='' \
  tests/test_locking.py tests/test_harvest_runtime.py tests/test_autoharvest.py
python -m src.cli coordination-status
python -m src.cli harvest-status
```

Required checks before enabling a scheduler loop:

- no active legacy trainer;
- source stats succeed;
- lock contention test returns 75 without mutation;
- scheduler dry-run has no state side effects;
- datasets/checkpoints use distinct labels and output directories;
- NAS sources are disabled or healthy (no stale handle);
- root and staging disks have sufficient headroom.

## Pitfalls

- `git worktree` alone does not protect shared runtime paths.
- Never place `"extract --label=ssd"` in one subprocess argv element; command and flag are separate elements.
- Counts printed by extract/clean/format are not Unix exit codes; successful CLI operations return 0.
- Do not treat source inspection errors as zero new sessions.
- Do not re-plan after advancing a harvest baseline; one immutable plan drives a scheduler cycle.
- Do not deploy an old checkpoint after a skipped or failed train phase.
