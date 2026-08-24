"""LFM2.5 agent harness: expose the finetuned edge model as an MCP subagent.

While src/subagent.py serves the RefinedToolCallV5 champions (loading HF
checkpoints onto this box), the LFM2.5 line is served by a llama.cpp
endpoint (CPU-friendly, GPU-free). This adapter exposes the SAME minimal
MCP-over-stdio surface so opencode / hermes-agent can delegate tasks to the
edge model exactly like they do to the champion:

    python -m src.lfm_harness --base-url http://127.0.0.1:8095

Tool:
    lfm_task {prompt, max_turns?, base_url?}
      -> runs the task through the LFM25Driver (pythonic-dialect parsing,
         stall/retry guards, tool clipping, autocompaction) in a sandbox
         and reports completion + a transcript digest.

The JSON-RPC handler is pure and injectable for tests (no live endpoint
needed): pass SubagentContext(make_driver=..., run_one=...).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.bench import Task, run_task

SERVER_NAME = "lfm25-subagent"
SERVER_VERSION = "0.1.0"
DEFAULT_BASE_URL = "http://127.0.0.1:8095"


@dataclass
class HarnessContext:
    """Injectable dependencies (mirrors subagent.SubagentContext)."""
    make_driver: Callable[..., object]
    run_one: Callable = field(default=run_task)


def _tool_spec() -> dict:
    return {
        "name": "lfm_task",
        "description": (
            "Delegate a sandbox task to the LFM2.5 1.2B edge agent. Uses "
            "bash/read/write/edit tools with dialect parsing, retry guards "
            "and autocompaction. Cheap and fast; best for small well-scoped "
            "file/command jobs."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string",
                           "description": "The task/instruction."},
                "max_turns": {"type": "integer",
                              "description": "Max agent turns (default 8)."},
                "temperature": {"type": "number",
                                "description": "Sampling temp (default 0.0)."},
            },
            "required": ["prompt"],
        },
    }


def _result_block(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle_request(req: dict, ctx: HarnessContext) -> Optional[dict]:
    method = req.get("method", "")
    msg_id = req.get("id")
    params = req.get("params", {}) or {}

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME,
                                   "version": SERVER_VERSION},
                }}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": [_tool_spec()]}}
    if method == "tools/call":
        if params.get("name") != "lfm_task":
            return {"jsonrpc": "2.0", "id": msg_id, "error":
                    {"code": -32601,
                     "message": f"unknown tool: {params.get('name')}"}}
        args = params.get("arguments", {}) or {}
        prompt = args.get("prompt", "")
        if not prompt:
            return {"jsonrpc": "2.0", "id": msg_id, "error":
                    {"code": -32602, "message": "prompt required"}}
        driver = ctx.make_driver(
            base_url=args.get("base_url") or DEFAULT_BASE_URL,
            temperature=float(args.get("temperature", 0.0)))
        task = Task(id="lfm-task", prompt=prompt, kind="exec",
                    max_turns=int(args.get("max_turns", 8)), checks=[])
        res = ctx.run_one(driver, task, "lfm2.5", "lfm_harness")
        digest_lines = [f"completed={res.completed} turns={res.turns}"]
        if res.error:
            digest_lines.append(f"error={res.error}")
        for entry in res.transcript[-6:]:
            role = entry.get("role", "?")
            body = str(entry.get("content") or "")[:300]
            digest_lines.append(f"[{role}] {body}")
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": _result_block("\n".join(digest_lines))}
    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id,
                "error": {"code": -32601,
                          "message": f"method not found: {method}"}}
    return None


def serve_stdio(ctx: Optional[HarnessContext] = None) -> None:
    if ctx is None:
        from src.drivers_lfm25 import LFM25Driver  # lazy import
        ctx = HarnessContext(
            make_driver=lambda **kw: LFM25Driver(**kw),
            run_one=run_task)
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
    ap = argparse.ArgumentParser(description="LFM2.5 subagent (MCP stdio)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = ap.parse_args(argv)

    from src.drivers_lfm25 import LFM25Driver
    ctx = HarnessContext(
        make_driver=lambda **kw: LFM25Driver(
            base_url=kw.get("base_url") or args.base_url,
            temperature=float(kw.get("temperature", 0.0))),
        run_one=run_task)
    serve_stdio(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
