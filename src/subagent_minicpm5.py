"""MiniCPM5 endpoint driver and MCP tool adapter."""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.bench import ApiDriver, ModelDriver, register_runner, run_task
from src.parsers import parse_native_tool_calls
from src.subagent_harness import (
    MINICPM5_PROFILE,
    HarnessContext,
    direct_tool_specs,
)
from src.subagent_harness import (
    handle_request as shared_handle_request,
)
from src.subagent_harness import (
    serve_stdio as shared_serve_stdio,
)

DEFAULT_BASE_URL = "http://127.0.0.1:38899/v1"
DEFAULT_MODEL = "minicpm5-2b"
SERVER_NAME = MINICPM5_PROFILE.server_name
SERVER_VERSION = MINICPM5_PROFILE.server_version

SYSTEM_PROMPT = """You are a bash coding agent. You have these tools: bash, read, write, edit, list.
Use exactly one tool call per turn. For bash, emit this exact format:
<function name="bash"><param name="command">COMMAND</param>"""

_FUNCTION_RE = re.compile(
    r'<function\s+name=["\']bash["\'][^>]*>\s*<param\s+name=["\']command["\'][^>]*>(.*?)</param>',
    re.S | re.I,
)
_BAD_TAG_RE = re.compile(r'<bash\s+command=["\']([^"\']*)["\'][^>]*>', re.I)
_BAD_LINE_RE = re.compile(
    r'(?:^|\n)\s*bash:\s*(.+?)(?=\n(?:result|Result|Output|thinking|Thinking|bash|Bash)|$)',
    re.S,
)
_CODE_RE = re.compile(r'```(?:bash|sh|shell)?\s*\n(.*?)```', re.S)
_INLINE_RE = re.compile(r'`([^`\n]+)`')
_DOLLAR_RE = re.compile(
    r'\$\s*((?:find|ls|cat|echo|tar|wc|grep|df|ps|free|nproc|whoami|pwd|mkdir|chmod|cp|mv|rm)[^\n]+)'
)


def _strip_cdata(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<![CDATA[") and raw.endswith("]]>"):
        raw = raw[len("<![CDATA["):-len("]]>")]
    return raw.strip()


def _looks_like_command(cmd: str) -> bool:
    first = cmd.splitlines()[0].strip().lstrip("$ ").lower()
    keywords = (
        "echo", "printf", "cat", "find", "ls", "mkdir", "touch", "cp", "mv",
        "grep", "wc", "df", "free", "pwd", "date", "uptime", "chmod", "tar",
        "python", "python3", "bash", "sh", "rm", "whoami", "nproc",
    )
    return first.startswith(keywords) or bool(re.search(r"\b(?:find|ls|cat|echo)\b", first))


def parse_minicpm_tool_calls(text: str) -> list[dict]:
    calls: list[dict] = []
    strict: list[dict] = []
    for match in _FUNCTION_RE.finditer(text):
        cmd = _strip_cdata(match.group(1))
        strict.append({"name": "bash", "args": {"command": cmd} if cmd else None})
    if strict:
        return strict
    for match in _BAD_TAG_RE.finditer(text):
        strict.append({"name": "bash", "args": {"command": match.group(1)}})
    if strict:
        return strict
    for match in _BAD_LINE_RE.finditer(text):
        cmd = match.group(1).strip()
        if cmd:
            calls.append({"name": "bash", "args": {"command": cmd}})
    if calls:
        return calls
    for match in _CODE_RE.finditer(text):
        cmd = match.group(1).strip()
        if _looks_like_command(cmd):
            calls.append({"name": "bash", "args": {"command": cmd}})
    if calls:
        return calls
    for match in _INLINE_RE.finditer(text):
        cmd = match.group(1).strip()
        if _looks_like_command(cmd):
            calls.append({"name": "bash", "args": {"command": cmd}})
    if calls:
        return calls
    for match in _DOLLAR_RE.finditer(text):
        cmd = match.group(1).strip()
        if cmd:
            calls.append({"name": "bash", "args": {"command": cmd}})
    if calls:
        return calls
    for call in parse_native_tool_calls(text):
        if call.get("name") == "bash" and isinstance(call.get("args"), dict):
            calls.append(call)
    if calls:
        return calls
    generic = re.compile(r'<tool_call\s+name=["\']bash["\'][^>]*>(.*?)</tool_call>', re.S)
    for match in generic.finditer(text):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            args = obj.get("arguments", obj.get("args"))
            calls.append({"name": "bash", "args": args or {"command": raw}})
    return calls


class MiniCPM5Driver(ApiDriver):
    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, api_key: str = "",
                 temperature: float = 0.0):
        super().__init__(base_url, model, api_key)
        self.temperature = temperature
        self.parse_tool_calls = staticmethod(parse_minicpm_tool_calls)

    def generate(self, messages, max_new_tokens: int = 512) -> str:
        messages = [dict(message) for message in messages]
        if not any(message.get("role") == "system" for message in messages):
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        import urllib.request
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode())
        message = data["choices"][0]["message"]
        return message.get("content", "") or message.get("reasoning_content", "")

    def wrap_result(self, result: str) -> str:
        return f"Result:\n{result}\nContinue task."


def make_minicpm_driver(model_path: str = DEFAULT_MODEL, rocm: bool = False,
                        base_url: str = DEFAULT_BASE_URL,
                        model: Optional[str] = None, api_key: str = "",
                        temperature: float = 0.0) -> MiniCPM5Driver:
    return MiniCPM5Driver(
        base_url=base_url, model=model or model_path or DEFAULT_MODEL,
        api_key=api_key, temperature=temperature)


_TOOL_SPECS = [MINICPM5_PROFILE.tool_spec()] + direct_tool_specs(
    MINICPM5_PROFILE.direct_tools)
_TOOL_NAMES = {tool["name"] for tool in _TOOL_SPECS}


@dataclass
class SubagentContext(HarnessContext):
    root: Path | None = None
    driver: Optional[ModelDriver] = None
    run_one: Callable = run_task
    profile: object = field(default_factory=lambda: MINICPM5_PROFILE)
    root_factory: Callable[[], Path] = field(
        default=lambda: Path(tempfile.mkdtemp(prefix="minicpm5-")))

    def __post_init__(self):
        if self.make_driver is None:
            self.make_driver = make_minicpm_driver
        if self.root is None:
            self.root = self.root_factory()
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)


def handle_request(req: dict, ctx: SubagentContext) -> Optional[dict]:
    return shared_handle_request(req, ctx)


def serve_stdio(ctx: Optional[SubagentContext] = None) -> None:
    if ctx is None:
        ctx = SubagentContext(driver=make_minicpm_driver())
    shared_serve_stdio(ctx)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MiniCPM5 subagent (MCP stdio)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.environ.get("MINICPM5_API_KEY", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)
    ctx = SubagentContext(
        driver=make_minicpm_driver(
            base_url=args.base_url, model=args.model, api_key=args.api_key,
            temperature=args.temperature))
    serve_stdio(ctx)
    return 0


def _make_minicpm(runner="minicpm5", **kwargs) -> MiniCPM5Driver:
    return make_minicpm_driver(**kwargs)


register_runner("minicpm5", _make_minicpm)
