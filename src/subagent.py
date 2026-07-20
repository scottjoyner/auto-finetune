"""Subagent adapter: expose the optimized RefinedToolCallV5 tool loop as an
agent subagent (MCP / ACP-over-stdio compatible) so opencode or hermes-agent
can delegate tasks to it.

Why stdlib-only (no `mcp` package):
  The `mcp` SDK is a heavy dependency and isn't installed in the finetune venv.
  The MCP/ACP wire protocol is just JSON-RPC 2.0 over stdio, so we implement the
  *minimal* surface by hand: initialize / tools/list / tools/call. This keeps
  the subagent runnable with zero extra deps and, crucially, makes the protocol
  handler a pure, unit-testable function (no live socket, no opencode needed).

  To plug into a real MCP client later, point it at this server's stdio; the
  message shapes match the MCP spec (serverInfo, capabilities.tools,
  tools/list, tools/call with content blocks).

Wire it up:
  python -m src.subagent --model /path/to/RefinedToolCallV5-3b [--variant base|finetune|auto]
  # opencode/hermes connect to this process's stdio as an MCP server.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.bench import Task, run_task

SERVER_NAME = "refinedtoolcall-subagent"
SERVER_VERSION = "0.1.0"


@dataclass
class SubagentContext:
    """Injectable dependencies so the handler is testable without a model.

    `make_driver` builds the ModelDriver for a given (model_path, variant).
    `run_one` runs a single task and returns a TaskResult (defaults to the real
    bench run_task).
    """
    make_driver: Callable[[str, str], object]
    run_one: Callable = field(default=run_task)


def _tool_spec() -> dict:
    return {
        "name": "run_task",
        "description": (
            "Run a task through the optimized RefinedToolCallV5 agent loop. "
            "The agent uses bash/read/write/edit/list tools in a sandbox and "
            "returns whether it completed the task."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string",
                           "description": "The task/instruction for the agent."},
                "model_path": {"type": "string",
                               "description": "HF checkpoint dir for the model."},
                "variant": {"type": "string", "enum": ["base", "finetune", "auto"],
                            "description": "Tool-format variant (default auto)."},
                "max_turns": {"type": "integer", "description": "Max agent turns."},
            },
            "required": ["prompt", "model_path"],
        },
    }


def _result_block(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def handle_request(req: dict, ctx: SubagentContext) -> Optional[dict]:
    """Pure JSON-RPC handler. Returns a response dict, or None for notifications.

    `req` is a parsed JSON-RPC 2.0 request. The returned dict (if any) already
    carries jsonrpc/id and the result/error — callers wrap it on the wire.
    """
    method = req.get("method", "")
    msg_id = req.get("id")
    params = req.get("params", {}) or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        }
    if method == "notifications/initialized":
        return None  # notification: no response
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": [_tool_spec()]}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        if name != "run_task":
            return {"jsonrpc": "2.0", "id": msg_id, "error":
                    {"code": -32601, "message": f"unknown tool: {name}"}}
        prompt = args.get("prompt", "")
        model_path = args.get("model_path", "")
        variant = args.get("variant", "auto")
        if not prompt or not model_path:
            return {"jsonrpc": "2.0", "id": msg_id, "error":
                    {"code": -32602, "message": "prompt and model_path required"}}
        driver = ctx.make_driver(model_path, variant)
        task = Task(id="subagent-task", prompt=prompt, kind="exec",
                    max_turns=int(args.get("max_turns", 12)), checks=[])
        res = ctx.run_one(driver, task, SERVER_NAME, "subagent")
        summary = (f"completed={res.completed} turns={res.turns} "
                   f"error={res.error or 'none'}")
        return {"jsonrpc": "2.0", "id": msg_id, "result": _result_block(summary)}
    # unknown method
    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601, "message": f"method not found: {method}"}}
    return None


def serve_stdio(model_path: str, variant: str = "auto", rocm: bool = False,
                ctx: Optional[SubagentContext] = None) -> None:
    """Run the subagent as an MCP/ACP-over-stdio server (blocking).

    Reads newline-delimited JSON-RPC requests from stdin, writes responses to
    stdout. `model_path` is fixed at startup (the connected harness launches
    this server with the model it wants to delegate to). `ctx` is injectable for
    tests; when None, a real ctx is built (lazy-imports torch via bench).
    """
    if ctx is None:
        from src.bench import make_driver as _mk  # lazy: avoid torch in tests
        ctx = SubagentContext(
            make_driver=lambda mp, v: _mk("subagent", model_path=mp,
                                          rocm=rocm, variant=v),
            run_one=run_task,
        )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req, ctx)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="RefinedToolCallV5 subagent (MCP stdio)")
    p.add_argument("--model", required=True, help="HF checkpoint dir")
    p.add_argument("--variant", default="auto",
                   choices=["base", "finetune", "auto"])
    p.add_argument("--rocm", action="store_true", help="use GPU (ROCm)")
    args = p.parse_args(argv)
    serve_stdio(args.model, variant=args.variant, rocm=args.rocm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
