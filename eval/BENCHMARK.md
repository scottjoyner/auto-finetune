# Agentic Task-Completion Benchmark (`src/bench.py`)

This answers the question the loss / tool-format evals can't: **did the task
actually get done?** It drives a *model* (not a dataset row) through a real
multi-turn tool-use loop in a throwaway sandbox and verifies the outcome with
concrete checkers.

## Three runners (per the benchmark design)

| runner | what it is | model source |
| --- | --- | --- |
| `self` | self-contained harness in this repo: parses `<tool_call>`, runs bash+file tools in a temp sandbox | a local HF dir (base / finetune / merged) |
| `api` | OpenAI-compatible endpoint (e.g. the lan lm-fleet-router) | `--base-url` + `--api-model`, or `--fleet` to auto-pick a large model |
| `hermes` | delegates to the hermes-agent harness on this machine (`mini_swe_runner.py`) — Hermes runs its OWN model+tool loop | whatever Hermes is configured with |

The `self` and `api` runners share the same tool protocol, so a task suite is
directly comparable between a 3B finetune and a 35B reference model.

## Task spec (`eval/tasks/*.jsonl`)

Each line is a JSON object:
```json
{
  "id": "exec-hello-file",
  "kind": "exec",                       // "exec" | "replay"
  "prompt": "Create greeting.txt ...",
  "tools": ["bash","read","write","edit","list"],   // optional, default all
  "max_turns": 8,                                   // optional
  "checks": [
    {"kind": "file_exists", "path": "greeting.txt"},
    {"kind": "file_contains", "path": "greeting.txt", "expect": "Hello, agent."}
  ]
}
```
Check kinds: `file_exists`, `dir_exists`, `file_contains`, `file_regex`,
`command_exit` (with `expect_code`), `command_output` (with `expect`).

`kind: "replay"` = a gold session transcript: given the initial prompt + tools,
did the model reproduce the completed outcome? (Put the original session's prior
messages in `replay_context` if needed; the verifier checks the end-state.)

## Run it

```bash
# self-contained, local 3B base model (needs GPU when not training)
python -m src.cli bench --runner=self --model=/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b

# a finetune / merged variant
python -m src.cli bench --runner=self --model=/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-ssd-merged

# large reference via the lan fleet router (auto-pick a big model)
python -m src.cli bench --runner=api --fleet --fleet-hint=35b

# or pin explicitly
python -m src.cli bench --runner=api --base-url=http://100.78.106.121:1234/v1 --api-model=qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex

# hermes harness (uses Hermes's own configured model)
python -m src.cli bench --runner=hermes

# custom task file
python -m src.cli bench --runner=self --model=... --tasks=eval/tasks/mytasks.jsonl
```

Output is a markdown table: per-task success, checks passed, turns, completion
rate. Save it to compare base vs each finetune vs the large reference side by
side.

## Notes / caveats
- The local `self` harness uses the RefinedToolCallV5 `<tool_call name=..>json\u276E\u276E\u276E`
  format (parsed tolerantly for both the real char and the escaped form).
- `hermes` runner trusts Hermes's own `completed` flag; pass `--sandbox_root` to
  also run our verifiers against the dir Hermes wrote into.
- Subagent runner (optimized harness for this model, run inside opencode/hermes)
  is a future extension — register it via `bench.register_runner("subagent", ...)`.
- Keep all artifacts on the local data drive; the fleet endpoints are read-only.
