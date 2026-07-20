"""LocalChatDriver: drive a STANDARD HF chat model (e.g. the locally-cached
Qwen2.5-7B-Instruct) through the same sandbox tool loop, using its NATIVE
tool-calling format (not the RefinedToolCallV5 <tool_call> dialect).

Why this exists: the cached Qwen2.5-7B is a genuine *local* large-reference
model (no network, no LAN fleet needed). It emits tool calls via the HF
convention (function-call JSON in the assistant turn), which is different from
RefinedToolCallV5's format — so it can't reuse OptimizedDriver.

Design notes:
  * We build a JSON-schema tool list from the standard tool names and pass it to
    apply_chat_template so the model knows the tools.
  * We parse tool calls from the decoded text with a tolerant balanced-brace
    JSON extractor (handles both {"name":..,"arguments":..} and
    {"function": {"name":..,"parameters":..}} shapes, with nested braces).
  * Results are fed back as a `tool` role message (HF convention), which the
    model's template renders correctly on the next turn.
  * Defaults to CPU (device_map="cpu") because a 7B model won't fit the 12GB
    iGPU in fp16; generation is slow but correct. Override rocm=True if you have
    enough VRAM / offload.
  * The model/tokenizer are injectable (_load can be replaced in tests) so the
    driver is fully testable without loading a 7B model.
"""
from __future__ import annotations

import json
from typing import Optional

from src.bench import ModelDriver

_TOOL_SCHEMAS = {
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the sandbox and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    "read": {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file's contents.",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
                "required": ["filePath"],
            },
        },
    },
    "write": {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write content to a file (overwrites).",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"},
                               "content": {"type": "string"}},
                "required": ["filePath", "content"],
            },
        },
    },
    "edit": {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "Replace oldText with newText in a file.",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"},
                               "oldText": {"type": "string"},
                               "newText": {"type": "string"}},
                "required": ["filePath", "oldText", "newText"],
            },
        },
    },
    "list": {
        "type": "function",
        "function": {
            "name": "list",
            "description": "List a directory's contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
}


def _tool_schemas(names: list[str]) -> list[dict]:
    return [_TOOL_SCHEMAS[n] for n in names if n in _TOOL_SCHEMAS]


def _balanced_brace_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of every balanced {...} block in text."""
    spans = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, i + 1))
                    start = None
    return spans


def parse_native_tool_calls(text: str) -> list[dict]:
    """Extract {"name", "args"} from a model's native function-call JSON.

    Handles both shapes:
      {"name": "bash", "arguments": {...}}   and
      {"function": {"name": "bash", "parameters": {...}}}
    Returns [] if nothing parseable is found.
    """
    calls = []
    for s, e in _balanced_brace_spans(text):
        raw = text[s:e]
        if '"name"' not in raw and '"function"' not in raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        name = obj.get("name")
        if name is None and "function" in obj:
            fn = obj["function"]
            name = fn.get("name")
            args = fn.get("parameters") or fn.get("arguments")
        else:
            args = obj.get("arguments") or obj.get("parameters")
        if name:
            calls.append({"name": name, "args": args})
    return calls


class LocalChatDriver(ModelDriver):
    """Runner 'local-chat': a standard HF chat model (e.g. Qwen2.5-7B)."""

    def __init__(self, model_path: str, rocm: bool = False, max_seq: int = 8192,
                 tools: Optional[list] = None, _model=None, _tok=None):
        self.model_path = model_path
        self.rocm = rocm
        self.max_seq = max_seq
        self.tools = tools or list(_TOOL_SCHEMAS.keys())
        self._model = _model
        self._tok = _tok
        # run_task prefers a per-driver parser when present
        self.parse_tool_calls = staticmethod(parse_native_tool_calls)

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = {"device_map": "auto"} if self.rocm else {"device_map": "cpu"}
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                          torch_dtype="auto", **dev)
        self._tok = AutoTokenizer.from_pretrained(self.model_path)

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        self._load()
        schemas = _tool_schemas(self.tools)
        prompt = self._tok.apply_chat_template(
            messages, tools=schemas, tokenize=False, add_generation_prompt=True)
        ids = self._tok(prompt, return_tensors="pt").input_ids.to(self._model.device)
        out = self._model.generate(ids, max_new_tokens=max_new_tokens,
                                   do_sample=False,
                                   pad_token_id=self._tok.pad_token_id)
        return self._tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)


def _make_local_chat(runner: str = "local-chat", **kw) -> ModelDriver:
    return LocalChatDriver(kw["model_path"], rocm=kw.get("rocm", False),
                           tools=kw.get("tools"))


# self-register so `make_driver("local-chat", ...)` works without the caller
# importing this module explicitly.
from src.bench import register_runner  # noqa: E402
register_runner("local-chat", _make_local_chat)
