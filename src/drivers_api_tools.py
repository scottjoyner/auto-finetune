"""ApiToolsDriver: drive an OpenAI-compatible endpoint that supports the
native `tools` parameter, and translate structured tool_calls into the
bench loop's <tool_call> JSON dialect.

Why: llama.cpp / vLLM handle template injection + parsing server-side when
tools are passed natively, which is far more reliable than prompt-shaped
tool lists for models whose special dialect we don't parse (e.g. serving
RefinedToolCallV5 GGUFs through llama.cpp). Proven on RefinedToolCallV5-3b
Q8_0 via llama.cpp: emits clean write/bash calls with correct escaping.

Registered as runner "api-tools":
    python -m src.cli bench --runner=api-tools \
        --base-url=http://127.0.0.1:8097/v1 --model=<served-name>
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Optional

from src.bench import ModelDriver, register_runner  # noqa: E402

_TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": ("Run ANY shell command in the sandbox; use "
                        "redirection to write outputs to files"),
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read", "description": "Read a file's contents",
        "parameters": {"type": "object",
                       "properties": {"filePath": {"type": "string"}},
                       "required": ["filePath"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "Write literal content to a file "
                                        "(overwrites)",
        "parameters": {"type": "object",
                       "properties": {"filePath": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["filePath", "content"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Replace exact old text with new text",
        "parameters": {"type": "object",
                       "properties": {"filePath": {"type": "string"},
                                      "oldString": {"type": "string"},
                                      "newString": {"type": "string"}},
                       "required": ["filePath", "oldString", "newString"]}}},
]

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class ApiToolsDriver(ModelDriver):
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 temperature: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        body = {"model": self.model, "messages": messages, "tools": _TOOLS,
                "tool_choice": "auto", "temperature": self.temperature,
                "max_tokens": max_new_tokens, "stream": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=600) as resp:
            msg = json.loads(resp.read())["choices"][0]["message"]

        blocks = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            blocks.append("<tool_call>\n" + json.dumps(
                {"name": fn.get("name"), "arguments": args or {}}) +
                "\n</tool_call>")
        content = _THINK_RE.sub("", msg.get("content") or "").strip()
        return "\n".join(blocks) if blocks else content


def _make_api_tools(runner: str = "api-tools", **kw) -> ApiToolsDriver:
    if not kw.get("base_url"):
        raise ValueError("api-tools requires --base-url (e.g. "
                         "http://127.0.0.1:8097/v1)")
    return ApiToolsDriver(base_url=kw["base_url"],
                          model=kw.get("model") or "local-model",
                          api_key=kw.get("api_key", ""))


register_runner("api-tools", _make_api_tools)
