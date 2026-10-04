"""LFM2.5 subagent adapter backed by the shared harness framework."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

from src.subagent_harness import (
    LFM25_PROFILE,
    make_default_context,
    result_block,
)
from src.subagent_harness import (
    HarnessContext as _HarnessContext,
)
from src.subagent_harness import (
    handle_request as shared_handle_request,
)
from src.subagent_harness import (
    serve_stdio as shared_serve_stdio,
)


@dataclass
class HarnessContext(_HarnessContext):
    profile: object = field(default_factory=lambda: LFM25_PROFILE)


SERVER_NAME = LFM25_PROFILE.server_name
SERVER_VERSION = LFM25_PROFILE.server_version
DEFAULT_BASE_URL = "http://127.0.0.1:8095"
SubagentContext = HarnessContext


def _tool_spec() -> dict:
    return LFM25_PROFILE.tool_spec()


def _result_block(text: str, is_error: bool = False) -> dict:
    return result_block(text, is_error)


def handle_request(req: dict, ctx: HarnessContext) -> Optional[dict]:
    return shared_handle_request(req, ctx)


def serve_stdio(ctx: Optional[HarnessContext] = None) -> None:
    if ctx is None:
        ctx = make_default_context("lfm25", base_url=DEFAULT_BASE_URL)
    shared_serve_stdio(ctx)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LFM2.5 subagent (MCP stdio)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=LFM25_PROFILE.default_model)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args(argv)
    ctx = make_default_context(
        "lfm25", base_url=args.base_url, model=args.model,
        temperature=args.temperature)
    serve_stdio(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
