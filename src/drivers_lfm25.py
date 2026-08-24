"""LFM25Driver: harness adapter for LiquidAI LFM2.5-* models served via an
OpenAI-compatible endpoint (llama.cpp / LM Studio / vLLM).

Why this exists: LFM2.5 has excellent native tool-calling *habits* but a
non-JSON dialect. Probed behaviour (2026-08-23, Q4_0 GGUF via llama.cpp):

  * Emits PYTHONIC calls: `<|tool_call_start|>[bash(command="ls")]<|tool_call_end|>`
    — special tokens arrive literally in the content; the JSON-dialect system
    override is IGNORED by the model.
  * Batches parallel independent calls into ONE list.
  * Escapes nested quotes correctly, then appends trailing prose after the
    end token.
  * Handles error tool-results gracefully (reports, doesn't retry-loop).
  * Never hallucinates results — it calls a tool instead of guessing.

This driver translates that dialect into whatever the bench loop expects,
so a 1.2B edge model can run the same agentic tasks as fleet models:

  * `parse_tool_calls` recovers calls from the pythonic spans (ast-based,
    execution-free) and returns bench-shaped dicts; unparseable calls yield
    args=None so bench's self-correction loop engages (the model recovers
    well from explicit argument errors).
  * `wrap_result` renders results as the trained JSON-array shape.
  * `generate` rewrites those wrapped user-role rows into proper `tool`
    role messages for the API, injects the "List of tools:" system doc if
    missing, and retries ONCE with a syntax nudge when it emits malformed
    call syntax.
"""
from __future__ import annotations

import ast
import json
import re
import urllib.request
from typing import Any, Optional

from src.bench import ModelDriver, register_runner  # noqa: E402

_TOOL_CALL_SPAN_RE = re.compile(
    r"<\|tool_call_start\|>(.*?)(?:<\|tool_call_end\|>|$)", re.DOTALL)

_TOOLS_DOC = (
    "You are an autonomous sandbox agent. You operate a real computer through "
    "tools; you CAN run any command, create directories, and write files "
    "yourself. Never say you cannot act and never ask the user to run "
    "commands - always act via tools immediately.\n"
    "Rules:\n"
    "1. To capture a command's output into a file, FIRST run it via bash "
    '(e.g. bash(command="uname -s > out.txt")). NEVER invent or guess what a '
    "command would output.\n"
    "2. Prefer one bash call that completes the whole task.\n"
    "3. After tools return results, give a short final answer with no tool "
    "call.\n"
    "4. If a tool result confirms your command already wrote the file, do "
    "NOT write or overwrite that file again - just give the final "
    "answer.\n"
    "5. Never claim to know what a command returns without running it; run "
    "the tool first, then speak.\n"
    'List of tools: [{"name": "bash", "description": "Run ANY shell command '
    'in the sandbox; use redirection to write outputs to files", '
    '"parameters": {"type": "object", "properties": {"command": {"type": '
    '"string"}}, "required": ["command"]}}, '
    '{"name": "read", "description": "Read a file\'s contents", "parameters": '
    '{"type": "object", "properties": {"filePath": {"type": "string"}}, '
    '"required": ["filePath"]}}, '
    '{"name": "write", "description": "Write literal content to a file '
    '(overwrites)", "parameters": {"type": "object", "properties": '
    '{"filePath": {"type": "string"}, "content": {"type": "string"}}, '
    '"required": ["filePath", "content"]}}, '
    '{"name": "edit", "description": "Replace exact old text with new text in '
    'a file", "parameters": {"type": "object", "properties": {"filePath": '
    '{"type": "string"}, "oldString": {"type": "string"}, "newString": '
    '{"type": "string"}}, "required": ["filePath", "oldString", '
    '"newString"]}}]'
)

# ── sm0l-derived guards ──────────────────────────────────────────────────
TOOL_OUTPUT_CLIP = 2000        # max chars of any single tool result fed back
COMPACT_RATIO = 0.62           # compact when est. tokens exceed this * ctx
CHARS_PER_TOKEN = 3.6

_COMPACT_SYS = (
    "You compress a chat log into a brief for your future self. "
    "Keep: user goals, file paths, commands, decisions, errors, unfinished "
    "work. Drop greetings and repeated tool dumps. Output plain prose, max "
    "280 words. No tools. No questions."
)


def estimate_tokens(messages: list[dict]) -> int:
    """Cheap char-based estimate (sm0l's heuristic)."""
    n = 0
    for m in messages:
        n += 8
        n += int(len(str(m.get("content") or "")) / CHARS_PER_TOKEN) + 1
    return n


def _clip(text: str, limit: int = TOOL_OUTPUT_CLIP) -> str:
    if len(text) <= limit:
        return text
    keep = limit // 2
    return text[:keep] + "\n…[clipped " + str(len(text) - limit) + " chars]…\n" + text[-keep:]


_MALFORMED_NUDGE = (
    "Your previous message contained a malformed tool call. Emit EXACTLY this "
    'syntax between the special tokens, one pythonic call per list element: '
    '<|tool_call_start|>[tool_name(arg="value", flag=True)]<|tool_call_end|>. '
    "Use double quotes for strings, True/False for booleans, no bare names."
)

# Probed habit: on abstract-sounding tasks the model sometimes slides into
# assistant-mode ("I don't have the ability...", "you can run ... locally",
# fenced instruction blocks) instead of acting. These are recoverable with
# one corrective turn.
_REFUSAL_RE = re.compile(
    r"i don't (?:see|have (?:the )?(?:ability|capability))|cannot directly|"
    r"not able to directly|i'm unable to|as an ai(?: language)? model|"
    r"you can run .*(?:locally|yourself)|would you like me to provide|"
    r"i cannot (?:create|make|run|write|access)|no (?:direct )?(?:file "
    r"system|terminal|shell) access|"
    r"here(?:'s| is) how you can|you could use the following|"
    r"i won't perform|since there are none|let me know if you'd like|"
    r"i have written|i've written|i have saved|i've saved|"
    r"i don't have [^.\n]{0,48}(?:way|ability|capability|access)|"
    r"without running a command|(?:run|do) .{0,30}for you[.?]|"
    r"would you like me to\b|i can run a command for you|"
    r"i will (?:now )?(?:write|create|save|run)\b|"
    r"(?:count|number) .{0,24}to the file you requested|"
    r"in your terminal", re.IGNORECASE)


def _literal(node: ast.AST) -> Any:
    """Evaluate a restricted expression tree: literals and containers only."""
    return ast.literal_eval(node)


def _call_from_ast(call_node: ast.Call) -> dict:
    name_node = call_node.func
    if not isinstance(name_node, ast.Name):
        raise ValueError("callee is not a plain name")
    name = name_node.id
    args: dict[str, Any] = {}
    # Positional args beyond zero are a habit violation -> force recovery.
    if call_node.args:
        raise ValueError("positional arguments not supported")
    for kw in call_node.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs not supported")
        args[kw.arg] = _literal(kw.value)
    return {"name": name, "args": args}


def parse_pythonic_tool_calls(text: str) -> list[dict]:
    """Extract LFM2.5-style calls from raw model text.

    Returns bench-shaped [{"name": str, "args": dict | None}, ...]. A span
    whose calls cannot be safely recovered yields a single entry with
    args=None so the caller's correction loop can engage.
    """
    out: list[dict] = []
    for span in _TOOL_CALL_SPAN_RE.finditer(text):
        body = span.group(1).strip()
        if not body:
            continue
        try:
            tree = ast.parse(body, mode="eval")
        except SyntaxError:
            out.append({"name": "", "args": None})
            continue
        expr = tree.body
        items: list[ast.AST]
        if isinstance(expr, ast.List):
            items = expr.elts
        elif isinstance(expr, ast.Call):
            items = [expr]
        else:
            out.append({"name": "", "args": None})
            continue
        got_any = False
        for item in items:
            try:
                if not isinstance(item, ast.Call):
                    raise ValueError("not a call")
                out.append(_call_from_ast(item))
                got_any = True
            except (ValueError, TypeError, MemoryError, RecursionError):
                # Name the tool when possible so error-recovery reads well.
                nm = ""
                if isinstance(item, ast.Call) and isinstance(item.func, ast.Name):
                    nm = item.func.id
                out.append({"name": nm, "args": None})
                got_any = True
        if not got_any:
            out.append({"name": "", "args": None})
    return out


def _is_wrapped_tool_result(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    s = content.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return False
    try:
        data = json.loads(s)
    except ValueError:
        return False
    return isinstance(data, list) and bool(data) and \
        all(isinstance(x, dict) and "output" in x for x in data)


class LFM25Driver(ModelDriver):
    """Drive an OpenAI-compatible endpoint hosting an LFM2.5 model."""

    def __init__(self, base_url: str = "http://127.0.0.1:8095",
                 model: str = "lfm2.5-1.2b-instruct",
                 api_key: str = "", temperature: float = 0.2,
                 advertise_tools: bool = True, num_ctx: int = 8192):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.advertise_tools = advertise_tools
        self.num_ctx = num_ctx

    # ── wire protocol ────────────────────────────────────────────────────
    def _chat(self, messages: list[dict], max_tokens: int) -> str:
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": max_tokens,
                "stream": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        msg = data["choices"][0]["message"]
        parts = []
        if msg.get("reasoning_content"):
            parts.append(str(msg["reasoning_content"]))
        parts.append(msg.get("content") or "")
        return "".join(parts)

    @staticmethod
    def _translate(messages: list[dict], advertise_tools: bool) -> list[dict]:
        """Map bench-loop history onto LFM2.5's trained conversation shape."""
        out: list[dict] = []
        sys_done = False
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role == "user" and _is_wrapped_tool_result(content):
                out.append({"role": "tool", "content": content})
                continue
            if role == "system":
                if advertise_tools and _TOOLS_DOC.split(":")[0] not in content:
                    content = _TOOLS_DOC + "\n" + (content or "")
                sys_done = True
            elif advertise_tools and not sys_done and role == "user":
                out.append({"role": "system", "content": _TOOLS_DOC})
                sys_done = True
            out.append({"role": role, "content": content})
        if advertise_tools and not sys_done:
            out.insert(0, {"role": "system", "content": _TOOLS_DOC})
        return out

    # ── ModelDriver contract ─────────────────────────────────────────────
    def _maybe_compact(self, payload_msgs: list[dict]) -> list[dict]:
        """sm0l-style autocompaction: summarise older turns at 62% ctx.

        Never splits a tool loop (cut points only at user turns); on summary
        failure the oldest non-system turns are hard-dropped instead.
        """
        num_ctx = int(self.num_ctx or 8192)
        thresh = max(int((num_ctx - 2048) * COMPACT_RATIO), 1024)
        if estimate_tokens(payload_msgs) < thresh:
            return payload_msgs
        user_idxs = [i for i, m in enumerate(payload_msgs)
                     if m.get("role") == "user"
                     and not _is_wrapped_tool_result(m.get("content"))]
        if len(user_idxs) < 2:
            return payload_msgs
        cut = user_idxs[-1]
        prefix, suffix = payload_msgs[:cut], payload_msgs[cut:]
        if not prefix:
            return payload_msgs
        blob_lines = []
        for m in prefix:
            if m.get("role") == "system":
                continue
            c = str(m.get("content") or "")[:600]
            if c:
                blob_lines.append(f"{m.get('role')}: {c}")
        try:
            summary = self._chat(
                [{"role": "system", "content": _COMPACT_SYS},
                 {"role": "user", "content": "Compress this log:\n\n"
                  + "\n".join(blob_lines)[-12000:]}], 400)
        except Exception:
            summary = ""
        memory = {"role": "system",
                  "content": "[compacted memory — earlier turns]\n" + summary}
        return [payload_msgs[0], memory] + suffix

    def generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        payload_msgs = self._translate(messages, self.advertise_tools)
        payload_msgs = self._maybe_compact(payload_msgs)
        text = self._chat(payload_msgs, max_new_tokens)

        # Habit-shaping retry: emitted call syntax we cannot recover?
        if _TOOL_CALL_SPAN_RE.search(text) and \
                not parse_pythonic_tool_calls(text):
            retry = payload_msgs + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": _MALFORMED_NUDGE},
            ]
            text2 = self._chat(retry, max_new_tokens)
            if parse_pythonic_tool_calls(text2):
                text = text2

        # Habit-shaping retry: assistant-mode refusal instead of acting.
        # Only triggers when the model produced NO tool call (a genuine final
        # answer never matches the refusal patterns).
        for _ in range(2):
            if _TOOL_CALL_SPAN_RE.search(text) or not _REFUSAL_RE.search(text):
                break
            payload_msgs = payload_msgs + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": (
                    "You DO have full tool access to this sandbox - the tools "
                    "are real and already connected. Do not ask me to run "
                    "anything. Act now: emit your next tool call.")},
            ]
            text = self._chat(payload_msgs, max_new_tokens)
        return text

    def parse_tool_calls(self, text: str) -> list[dict]:  # noqa: D102
        return parse_pythonic_tool_calls(text)

    @staticmethod
    def wrap_result(result: str) -> str:
        # sm0l guard: hard-clip huge tool dumps before they flood context.
        return json.dumps([{"output": _clip(str(result))}],
                          ensure_ascii=False)


def _make_lfm25(runner: str = "lfm25", **kw) -> LFM25Driver:
    return LFM25Driver(base_url=kw.get("base_url", "http://127.0.0.1:8095"),
                       model=kw.get("model", "lfm2.5-1.2b-instruct"),
                       api_key=kw.get("api_key", ""),
                       temperature=float(kw.get("temperature", 0.2)))


register_runner("lfm25", _make_lfm25)
