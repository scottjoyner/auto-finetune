# Agentic Task-Completion Benchmark (`src/bench.py`)

This answers the question the loss / tool-format evals can't: **did the task
actually get done?** It drives a *model* (not a dataset row) through a real
multi-turn tool-use loop in a throwaway sandbox and verifies the outcome with
concrete checkers.

## Five runners (per the benchmark design)

| runner | what it is | model source |
| --- | --- | --- |
| `self` | self-contained harness in this repo: parses `<tool_call>`, runs bash+file tools in a temp sandbox | a local HF dir (base / finetune / merged) |
| `api` | OpenAI-compatible endpoint (e.g. the lan lm-fleet-router) | `--base-url` + `--api-model`, or `--fleet` to auto-pick a large model |
| `hermes` | delegates to the hermes-agent harness on this machine (`mini_swe_runner.py`) — Hermes runs its OWN model+tool loop | whatever Hermes is configured with |
| `subagent` | **optimized** harness built specifically for RefinedToolCallV5 variants (see below) | a local HF dir (base / finetune / merged) |
| `local-chat` | **standard HF chat model** using its NATIVE function-call format (not the RefinedToolCallV5 dialect) — for genuine local large references like the cached Qwen2.5-7B | a local HF chat dir (e.g. Qwen2.5-7B-Instruct) |

The `self`, `api`, `subagent`, and `local-chat` runners share the same task
suite and verifiers, so they are directly comparable (3B finetune vs 35B
reference vs the optimized loop vs a native-format 7B — all on the same suite).

## `subagent` — the optimized harness

This is the model-specific loop (the "third runner" you asked for), built to
wring the best tool-calling out of RefinedToolCallV5. Key insight that drove it:
the model has **two** tool-call formats depending on variant —

  * **finetunes** were trained on the simpler dataset format:
    `<tool_call name="x" call_id="y">{json}\u276E\u276E\u276E` then `<tool_result>...</tool_result>`
  * **the base model** emits the chat-template format:
    `<tool_call>{"name":..,"arguments":..}</tool_call>` and expects results as
    `<tool_response>...</tool_response>`

The generic `self` runner only understood the finetune format. `subagent`
(`OptimizedDriver`) adds:

  * **per-variant result formatting** — feeds `<tool_result>` to finetunes (what
    they were trained on) but `<tool_response>` to the base model (what its chat
    template expects). This alone changes call fidelity.
  * **explicit stop sequences** — stops at `<|im_end|>` and the tool separator so
    the model doesn't ramble past a complete call.
  * **error recovery** — when args are unparseable, a structured recovery message
    is returned (mirrors the model's advertised 0.896 recovery rate) so it can
    self-correct instead of the loop dying.
  * **variant autodetect** — `base` if the checkpoint path is the bare
    `RefinedToolCallV5-3b` dir, else `finetune` (override with `--variant=`).

```bash
python -m src.cli bench --runner=subagent \
  --model=/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b
python -m src.cli bench --runner=subagent \
  --model=/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-ssd-merged
```

Intended to later run as a subagent inside opencode / hermes-agent; for now it
is a drop-in `bench` runner so you can measure the optimization's impact
(`self` vs `subagent` on the same checkpoint).

## `local-chat` — standard HF chat model (native tool format)

For models that are NOT RefinedToolCallV5 (e.g. the locally-cached
`Qwen/Qwen2.5-7B-Instruct` at
`/home/scott/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct`), which
emit tool calls as native function-call JSON
(`{"name":..,"arguments":{..}}` or `{"function":{..}}`) rather than the
`<tool_call>` dialect. `LocalChatDriver` (`src/drivers_localchat.py`):

  * builds the JSON-schema tool list and passes it to `apply_chat_template`
    (`tools=...`) so the model knows the tools,
  * parses tool calls with a balanced-brace tolerant extractor that handles
    **nested** argument objects,
  * feeds results back as a `tool` role message (HF convention),
  * defaults to **CPU** (`device_map="cpu"`) because a 7B model won't fit the
    12GB iGPU in fp16; override `rocm=True` if you have enough VRAM / offload.

The `q8` GGUF/quantized build is also available at
`/home/scott/.lmstudio/models` (lmstudio) if you want a faster local large
reference — point `local-chat` at it, or load the fp16 if higher fidelity is
needed. The 7B is a genuine *local* reference (no network, no LAN fleet).

```bash
python -m src.cli bench --runner=local-chat \
  --model=/home/scott/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
```

## `bench-matrix` — one suite, many models

Run the same task suite across several model/runner specs and emit a single
combined table (aggregate pass/complete rates *and* a per-task ✓/✗ grid). Two
ways to specify specs:

```bash
# explicit: --specs is a JSON list of {name, runner, ...make_driver kwargs}
python -m src.cli bench-matrix --specs='[
  {"name":"qwen2.5-7b","runner":"local-chat","model_path":"/home/scott/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"},
  {"name":"ssd-ft","runner":"subagent","model_path":"/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-ssd"},
  {"name":"fleet-35b","runner":"api","base_url":"http://...","model":"qwen3.6-35b"}
]'

# presets (no --specs needed)
python -m src.cli bench-matrix --preset=local-refs   # local qwen7b + any finished FT adapters
python -m src.cli bench-matrix --preset=fleet        # all models in endpoints.json
python -m src.cli bench-matrix --preset=local-refs --report   # also write eval/bench-matrix.md
```

`--tasks=` overrides the default `eval/tasks/*.jsonl`. Each spec runs in
isolation; a failing spec (e.g. bad runner) errors out without killing the
matrix.

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

# the optimized loop on the same base model
python -m src.cli bench --runner=subagent --model=/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b

# a finetune / merged variant
python -m src.cli bench --runner=self --model=/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-ssd-merged

# large reference via the lan fleet router (auto-pick a big model)
python -m src.cli bench --runner=api --fleet --fleet-hint=35b

# hermes harness (uses Hermes's own configured model)
python -m src.cli bench --runner=hermes

# custom task file
python -m src.cli bench --runner=self --model=... --tasks=eval/tasks/mytasks.jsonl

# one suite, many models (combined comparison table)
python -m src.cli bench-matrix --preset=local-refs --report
python -m src.cli bench-matrix --specs='[{"name":"qwen7b","runner":"local-chat","model_path":"/home/scott/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"},{"name":"ssd-ft","runner":"subagent","model_path":"/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-ssd"}]'
```

Output is a markdown table: per-task success, checks passed, turns, completion
rate. Save it to compare base vs each finetune vs the large reference side by
side.

## Notes / caveats
- The local `self` harness uses the RefinedToolCallV5 `<tool_call name=..>json\u276E\u276E\u276E`
  format (parsed tolerantly for both the real char and the escaped form) AND the
  base model's canonical `<tool_call>json</tool_call>` format.
- `subagent` feeds results back in the variant-correct wrapper
  (`<tool_result>` for finetunes, `<tool_response>` for base).
- `hermes` runner trusts Hermes's own `completed` flag; pass `--sandbox_root` to
  also run our verifiers against the dir Hermes wrote into.
- Keep all artifacts on the local data drive; the fleet endpoints are read-only.

## Running the subagent as a real MCP/ACP server

`src/subagent.py` wraps `OptimizedDriver` as an MCP/ACP-over-stdio server (no
`mcp` SDK needed — stdlib JSON-RPC 2.0). opencode (`opencode acp`/`serve`) or
hermes (`mcp serve`) can connect to its stdio and call the `run_task` tool,
delegating real tasks to the optimized RefinedToolCallV5 loop.

```bash
# launch with the model you want it to run
python -m src.subagent --model=/media/scott/SSD_4TB/models-fast/RefinedNeuro/RefinedToolCallV5-3b --variant=base
# or a finetune / merged
python -m src.subagent --model=/media/scott/data/finetune-staging/outputs/checkpoints/toolcall-v5-3b-ssd-merged
```

The server speaks `initialize` / `tools/list` / `tools/call` with MCP-shaped
responses. Tests: tests/test_subagent.py (7) cover the protocol handler and a
full stdio round-trip without loading a model.
