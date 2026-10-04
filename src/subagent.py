"""RefinedToolCallV5 local subagent adapter.

This module is the local-model profile for the shared MCP/ACP-over-stdio
subagent harness. The protocol and task loop live in ``src.subagent_harness``;
this file keeps the historical command-line API and test doubles compatible.
"""
from __future__ import annotations

import argparse
from typing import Optional

from src.subagent_harness import (
    LOCAL_PROFILE,
    HarnessContext,
    handle_request as shared_handle_request,
    make_default_context,
    result_block,
    serve_stdio as shared_serve_stdio,
)

SERVER_NAME = LOCAL_PROFILE.server_name
SERVER_VERSION = LOCAL_PROFILE.server_version
SubagentContext = HarnessContext


def _tool_spec() -> dict:
    return LOCAL_PROFILE.tool_spec()


def _result_block(text: str) -> dict:
    return result_block(text)


def handle_request(req: dict, ctx: SubagentContext) -> Optional[dict]:
    return shared_handle_request(req, ctx)


def serve_stdio(model_path: str, variant: str = "auto", rocm: bool = False,
                ctx: Optional[SubagentContext] = None) -> None:
    if ctx is None:
        ctx = make_default_context(
            "local", model_path=model_path, variant=variant, rocm=rocm)
    shared_serve_stdio(ctx)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="RefinedToolCallV5 subagent (MCP stdio)")
    parser.add_argument("--model", required=True, help="HF checkpoint dir")
    parser.add_argument("--variant", default="auto",
                        choices=["base", "finetune", "auto"])
    parser.add_argument("--rocm", action="store_true", help="use GPU (ROCm)")
    args = parser.parse_args(argv)
    serve_stdio(args.model, variant=args.variant, rocm=args.rocm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
