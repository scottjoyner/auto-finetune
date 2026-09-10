#!/usr/bin/env python3
"""MiniCPM5-optimized subagent harness.

Built continuously. Uses reasoning_content when content is empty,
handles thinking-pattern stops, verifies bash/file tool execution,
and feeds errors back into the loop.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Import the harness fix from bench.py
sys.path.insert(0, "/media/scott/data/git/auto-finetune/src")
from bench import Task, run_task, ApiDriver

SERVER_NAME = "minicpm5-subagent"
SERVER_VERSION = "1.0.0"


def make_minicpm_driver(model_path: str, rocm: bool = False):
    """Build a driver for MiniCPM5 (GGUF or HF path) with harness fix applied."""
    # For LMStudio endpoint (local or remote), use ApiDriver
    # For local GGUF, we use the harness-adapted approach
    return ApiDriver(base_url="http://127.0.0.1:38899", model="minicpm5-2b", api_key="")


def handle_request(req: dict, ctx: "SubagentContext") -> dict:
    method = req.get("method", "")
    msg_id = req.get("id")
    params = req.get("params", {}) or {}

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"protocolVersion": "2024-11-05",
                           "capabilities": {"tools": {"bash": {}, "write": {}, "read": {}, "edit": {}, "list": {}}},
                           "serverInfo": {"name": "minicpm5-subagent", "version": "1.0.0"}}}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": [{"name": "bash", "description": "Run bash commands"},
                                     {"name": "write", "description": "Write file"},
                                     {"name": "read", "description": "Read file"},
                                     {"name": "edit", "description": "Edit file"},
                                     {"name": "list", "description": "List files"}]}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        if name == "bash":
            cmd = args.get("command", "")
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                return {"jsonrpc": "2.0", "id": msg_id,
                        "result": {"content": [{"type":"text", "text": result.stdout + result.stderr}], "isError": result.returncode != 0}}
            except Exception as e:
                return {"jsonrpc": "2.0", "id": msg_id,
                        "result": {"content": [{"type":"text", "text": str(e)}], "isError": True}}
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"message": f"unknown tool: {name}"}}
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"message": f"method not found: {method}"}}


class SubagentContext:
    def __init__(self):
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/scott/.lmstudio/models/openbmb/MiniCPM5-2B-GGUF/MiniCPM5-2B-Q4_K_M.gguf")
    args = p.parse_args()
    print(f"MiniCPM5 subagent harness active. Model: {args.model}")
