# Auto-Harvester Pipeline

This document describes the fully autonomous harvest-train-deploy pipeline
that monitors for new data, trains models, and deploys them to inference nodes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUTO-HARVESTER PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   Harvest    │───▶│    Train     │───▶│    Eval      │───▶│  Deploy  │ │
│  │  (CPU-only)  │    │   (GPU)      │    │  (GPU/CPU)   │    │ (multi)  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │  Drift      │    │   Metrics    │    │   Notify     │    │  Health  │ │
│  │  Detection  │    │   Tracking   │    │   Alerts     │    │  Check   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Drift Detection (`harvest.py`)

Monitors live session databases for new data and decides when to trigger
a training run.

```bash
# Check current status
python -m src.cli harvest-status
# [harvest-status]
#   opencode: 1250 sessions, 125 new since last harvest, 2.3 days ago
#   hermes: 890 sessions, 45 new since last harvest, 1.1 days ago

# Plan next action
python -m src.cli harvest-plan --min-new=50
# [harvest-plan] should_harvest=True should_train=True
#   total_new=170, batch=['opencode', 'hermes']
#   reason: opencode: 125 new sessions >= 50
```

**Decision Logic:**
- `should_harvest = True` if any source has >= `min_new_sessions` new sessions
- `should_train = True` if total new sessions >= `min_new_sessions`
- Estimates training time based on dataset size

### 2. CPU-Heavy Data Processing

All of these run on CPU while GPU training is active:

| Module | Command | Purpose |
|--------|---------|---------|
| `dedup.py` | `dedup --threshold=0.85` | MinHash + LSH near-duplicate detection |
| `profile.py` | `profile` | Token stats, language detection, topic clustering |
| `pretokenize.py` | `pretokenize --model=<path>` | Batch tokenize to Arrow/Parquet |
| `auto_balance.py` | `auto-balance --cap=500` | Weighted balancing across task buckets |
| `dataset_version.py` | `dataset-version-create` | Version datasets for reproducibility |

**MinHash Deduplication:**
```bash
python -m src.cli dedup --threshold=0.85
# [dedup] 2500 sessions, threshold=0.85
# [dedup] removed 180 exact session_id duplicates
# [dedup] kept 2320 sessions, removed 45 near-duplicates
```

**Dataset Profiling:**
```bash
python -m src.cli profile
# [profile] profiling 2320 sessions
# [profile] 2320 sessions, 4500000 tokens
# [profile] avg tokens: 1940, median: 1650
# [profile] languages: {'python': 1200, 'javascript': 450, 'text': 670}
```

**Auto-Balancing:**
```bash
python -m src.cli auto-balance --cap=500
# [auto-balance] loaded 2320 sessions
# [auto-balance] buckets: {'debug': 800, 'reasoning': 600, 'file-edit': 400, ...}
# [auto-balance] sampled 1800 sessions across 10 buckets
#   file-edit: 200
#   multi-file-refactor: 150
#   shell: 180
#   debug: 250
#   ...
```

### 3. Scheduler (`scheduler.py`)

Orchestrates the full harvest-train-deploy cycle.

```bash
# Run one complete cycle
python -m src.cli scheduler-run

# Or run continuously
python -m src.cli scheduler-loop --interval=3600
```

**Scheduler Flow:**
1. **Harvest Phase**: Extract + clean new sessions
2. **Train Phase**: Format datasets, train on queue
3. **Eval Phase**: Evaluate all adapters, pick best, merge
4. **Deploy Phase**: Deploy to inference nodes
5. **Record**: Update metrics, costs, dataset versions

**State Persistence:**
```json
{
  "last_run": 1784500000,
  "last_harvest": 1784500000,
  "last_train": 1784500000,
  "last_deploy": 1784500000,
  "runs_completed": 15,
  "runs_failed": 2,
  "current_phase": "idle",
  "last_error": null
}
```

### 4. Multi-Node Deployment (`deploy.py`)

Deploys models to multiple inference nodes in parallel with health checks.

```bash
# Deploy to specific nodes
python -m src.cli deploy --label=combined --nodes=local,nas5,laptop

# With quorum requirement
python -m src.cli deploy --label=combined --nodes=local,nas5 --quorum=2

# Auto-discover nodes
python -m src.cli discover-nodes
# [discover-nodes] 3 nodes: ['local', 'nas5', 'laptop']

# Check deployment status
python -m src.cli multi-deploy-status
# [multi-deploy-status]
#   local:
#     combined v3 [active] 1250MB
#   nas5:
#     combined v3 [active] 1250MB
#   laptop:
#     combined v2 [standby] 1250MB
```

**Health Checks:**
- Verifies `config.json` and `tokenizer.json` exist
- Tests tokenizer load and basic encoding
- Atomic symlink rotation for zero-downtime deploys

**Rollback:**
```bash
python -m src.cli rollback --label=combined --nodes=local,nas5
# [multi-rollback] 2/2 nodes rolled back
```

### 5. Model Registry (`registry.py`)

Tracks model versions, lineage, and metrics.

```bash
# List all models
python -m src.cli registry-list
# [registry] 12 models:
#   toolcall-v5-3b-combined-v3 [active] 1250MB loss=0.4200 tool=0.850
#   toolcall-v5-3b-combined-v2 [deployed] 1250MB loss=0.4500 tool=0.820
#   toolcall-v5-3b-ssd-v5 [deployed] 1250MB loss=0.3800 tool=0.880

# Register a model
python -m src.cli registry-add --label=combined --checkpoint=/path/to/model
```

**Lineage Tracking:**
```python
{
  "model_id": "toolcall-v5-3b-combined-v3",
  "label": "combined",
  "version": 3,
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "parent_model": "toolcall-v5-3b-combined-v2",
  "train_loss": 0.42,
  "eval_loss": 0.45,
  "tool_exact_match": 0.85
}
```

### 6. Metrics and Regression Detection (`metrics.py`)

Tracks training metrics across versions and detects regressions.

```bash
# Record metrics
python -m src.cli metrics-record --label=combined \
  --eval-loss=0.42 --tool-exact=0.85 --runtime=28800

# Check for regression
python -m src.cli metrics-regression --label=combined
# [metrics-regression] OK: eval_loss=0.4200 (best=0.4150)

# Compare versions
python -m src.cli metrics-compare --label=combined
# [metrics-compare] v2 vs v3
#   eval_loss: 0.4500 -> 0.4200 (-0.0300, -6.67%)
#   tool_exact_match: 0.8200 -> 0.8500 (+0.0300, +3.66%)
```

**Regression Detection:**
- Compares latest vs best version
- Threshold: 5% worse = regression
- Returns `(is_regression, message)` tuple

### 7. Notifications (`notify.py`)

Sends alerts when pipeline events occur.

```bash
# Send notification
python -m src.cli notify --event=training_complete --message="combined model trained"

# View history
python -m src.cli notify-history
# [notify-history] 5 recent notifications:
#   [info] 2026-07-20 22:00 training_complete: combined model trained
#   [info] 2026-07-20 20:30 deploy_complete: deployed to 3 nodes
#   [error] 2026-07-20 18:00 training_failed: CUDA out of memory
```

**Notification Channels:**
- Desktop (`notify-send` on Linux)
- Webhook (Slack, Discord, etc.)
- Email (SMTP)
- Log file (always)

**Configuration:**
```yaml
notify:
  desktop: true
  webhook_url: "https://hooks.slack.com/..."
  email_to: "admin@example.com"
  smtp_host: "localhost"
  smtp_port: 25
```

### 8. Cost Tracking (`cost.py`)

Tracks GPU hours and resource usage.

```bash
# Record cost
python -m src.cli cost-record --label=combined --hours=8.5 --gpu="Radeon 890M"

# View summary
python -m src.cli cost-summary
# [cost-summary]
#   runs: 15
#   total hours: 125.5
#   training hours: 110.0
#   eval hours: 15.5
#   avg hours/run: 8.4
#
#   by label:
#     combined: 5 runs, 42.5h
#     ssd: 3 runs, 25.0h
#     hermes: 2 runs, 18.0h
```

### 9. Quantization (`quantize.py`)

Post-merge quantization for faster inference.

```bash
# Quantize to 4-bit GPTQ
python -m src.cli quantize --label=combined --bits=4 --method=gptq
# [quantize] combined (4-bit GPTQ)
#   source: /path/to/merged
#   output: /path/to/merged-4bit
# [quantize] GPTQ 4-bit quantized
#   1250MB -> 320MB (3.9x compression)

# Check quantized models
python -m src.cli quantize-status
# [quantize-status] 2 quantized models:
#   toolcall-v5-3b-combined-merged-4bit: 320MB
#   toolcall-v5-3b-ssd-merged-4bit: 280MB
```

## Continuous Operation

### Running as a Service

```bash
# Start the scheduler loop
nohup python -m src.cli scheduler-loop --interval=3600 > /var/log/auto-finetune.log 2>&1 &

# Or use the provided script
./launch-next.sh --loop
```

### Monitoring

```bash
# Check pipeline status
python -m src.cli scheduler-status
# [scheduler-status]
#   last_run: 1784500000
#   runs_completed: 15
#   runs_failed: 2
#   current_phase: idle

# View recent notifications
python -m src.cli notify-history --limit=10

# Check costs
python -m src.cli cost-summary
```

### Recovery

```bash
# If training fails
python -m src.cli scheduler-run  # retries automatically

# If deployment fails
python -m src.cli rollback --label=combined --nodes=local,nas5

# Reset scheduler state
python -m src.cli scheduler-status  # check current state
rm -f /media/scott/data/finetune-staging/data/analysis/scheduler-state.json
```

## Safety Rules

1. **Never overwrite `datasets/` while training is running.**
   - `extract` and `clean` are safe during training
   - `format` and `combine` should only run when training is idle

2. **Read source DBs read-only.**
   - The extractor uses `?mode=ro` URI parameter
   - Never write to network mounts

3. **Verify GPU health before training.**
   - The scheduler runs a GPU liveness probe
   - Falls back to warning if probe fails

4. **Use atomic symlink rotation for deploys.**
   - Creates temp link, then `os.rename()` for atomic swap
   - Previous version preserved for rollback

5. **Health check after deploy.**
   - Verifies tokenizer loads correctly
   - Checks critical files exist
   - Fails deployment if health check fails

## File Locations

| Type | Path |
|------|------|
| Raw sessions | `/media/scott/data/finetune-staging/data/raw/` |
| Cleaned sessions | `/media/scott/data/finetune-staging/data/cleaned/` |
| Datasets | `/media/scott/data/finetune-staging/data/datasets/` |
| Analysis | `/media/scott/data/finetune-staging/data/analysis/` |
| Checkpoints | `/media/scott/data/finetune-staging/outputs/checkpoints/` |
| Logs | `/media/scott/data/finetune-staging/logs/` |
| Notifications | `/media/scott/data/finetune-staging/data/analysis/notifications/` |
| Metrics | `/media/scott/data/finetune-staging/data/analysis/metrics/` |
| Registry | `/media/scott/data/finetune-staging/data/analysis/registry/` |
| Costs | `/media/scott/data/finetune-staging/data/analysis/costs/` |
| Dataset versions | `/media/scott/data/finetune-staging/data/analysis/dataset-versions/` |
| Inference models | `/media/scott/SSD_4TB/inference/models/` |

## Next Steps

### Immediate (v1.1)
- [ ] Add webhook notification to Slack/Discord
- [ ] Implement canary deployments (partial rollout)
- [ ] Add GPU memory monitoring
- [ ] Create dashboard for pipeline visualization

### Medium-term (v1.2)
- [ ] Implement feedback loop (inference results → training data)
- [ ] Add A/B testing for model comparison
- [ ] Create automated dataset versioning on schedule
- [ ] Add cost optimization (spot instances, scheduling)

### Long-term (v2.0)
- [ ] Multi-machine training (distributed)
- [ ] Real-time learning from inference
- [ ] Automated hyperparameter tuning
- [ ] Model compression (distillation, pruning)
- [ ] Integration with cloud training (AWS/GCP/Azure)
