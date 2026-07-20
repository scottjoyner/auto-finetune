"""Shared tool-call parsers for the benchmark harness.

Two distinct model output formats are supported:

  * RefinedToolCallV5 dialect (`parse_tool_calls`):
      - finetune/dataset:  <tool_call name="x" call_id="y">{json}<sep>
      - base canonical:    <tool_call>\n{"name":..,"arguments":..}\n</tool_call>
  * native HF function-call JSON (`parse_native_tool_calls`):
      {"name":..,"arguments":{..}}   or   {"function":{"name":..,"parameters":{..}}}

Both return a list of ``{"name", "args"}`` dicts so the tool loop can stay
format-agnostic.
"""
from __future__ import annotations

import json
import re
from typing import Optional

# ── RefinedToolCallV5 dialect ────────────────────────────────────────────────
# tolerant of both the real U+276E separator and the literal backslash-escaped
# "\u276E\u276E\u276E" seen in the serialized datasets.
_SEP_CHARS = "\u276E\u276E\u276E"
_SEP_LITERAL = "\\u276E\\u276E\\u276E"
_TOOL_CALL_RE = re.compile(
    r"<tool_call\s+name=\"([^\"]+)\"\s*(?:call_id=\"[^\"]*\")?>(.*?)(?:"
    + re.escape(_SEP_CHARS) + "|" + re.escape(_SEP_LITERAL) + ")",
    re.DOTALL,
)
# Canonical format the BASE RefinedToolCallV5 model emits (per its chat
# template): <tool_call>\n{...}\n</tool_call>.
_CANON_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _safe_json(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(raw.strip().strip("`").strip())
        except Exception:
            return None


def parse_tool_calls(text: str) -> list[dict]:
    """RefinedToolCallV5 dialect: finetune + base canonical formats.

    Returns a list of {"name", "args"}; args is normalized to the "args" key.
    """
    calls: list[dict] = []
    # dataset / finetune format first
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(2).strip()
        calls.append({"name": m.group(1), "args": _safe_json(raw)})
    if calls:
        return calls
    # canonical base format
    for m in _CANON_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        name = obj.get("name")
        args = obj.get("arguments", obj.get("args"))
        if name:
            calls.append({"name": name, "args": args})
    return calls


# ── native HF function-call JSON ──────────────────────────────────────────────
def _balanced_brace_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of every balanced {...} block in text."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start: Optional[int] = None
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
    Returns [] if nothing parseable is found. An empty ``arguments: {}`` is
    preserved as ``{}`` (not collapsed to ``None``).
    """
    calls: list[dict] = []
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
            args = fn.get("parameters")
            if args is None:
                args = fn.get("arguments")
        else:
            args = obj.get("arguments")
            if args is None:
                args = obj.get("parameters")
        if name:
            calls.append({"name": name, "args": args})
    return calls
