"""Reconstruct cleaned sessions into training examples.

Each cleaned session is turned into one (or several, if windowed) conversation
following a chat template: chatml | alpaca | sharegpt.

Tool calls are rendered as natural text so the model learns to emit tool-use
behaviour in a single stream. Patch parts are rendered as diff blocks.
"""
from __future__ import annotations

import json
import os
from typing import Any

from src.config import Config


def _render_part(p: dict) -> str:
    t = p.get("type")
    if t == "text":
        return p.get("text", "")
    if t == "tool":
        tool = p.get("tool", "tool")
        call_id = p.get("call_id", "")
        inp = p.get("input")
        out = p.get("output")
        lines = [f"<tool_call name=\"{tool}\" call_id=\"{call_id}\">"]
        lines.append(json.dumps(inp, indent=2) if isinstance(inp, (dict, list)) else str(inp or ""))
        lines.append("</tool_call>")
        if out:
            lines.append("<tool_result>")
            lines.append(out if isinstance(out, str) else json.dumps(out, indent=2))
            lines.append("</tool_result>")
        return "\n".join(lines)
    if t == "patch":
        files = p.get("files") or []
        return "<patch files=\"" + ", ".join(files) + "\">\n(diff applied)\n</patch>"
    if t == "reasoning":
        return p.get("text", "")
    # step markers / compaction: ignore in training text
    return ""


def _render_message(m: dict) -> str:
    return "\n".join(_render_part(p) for p in m.get("parts", []) if _render_part(p)).strip()


def to_chatml(messages: list[dict], system: str) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        content = _render_message(m)
        if not content:
            continue
        role = "assistant" if m.get("role") == "assistant" else "user"
        out.append({"role": role, "content": content})
    return out


def to_sharegpt(messages: list[dict], system: str) -> list[dict]:
    conv = []
    for m in messages:
        content = _render_message(m)
        if not content:
            continue
        role = "gpt" if m.get("role") == "assistant" else "human"
        conv.append({"from": role, "value": content})
    if system:
        conv.insert(0, {"from": "system", "value": system})
    return conv


def to_alpaca(messages: list[dict], system: str) -> dict:
    # Alpaca is instruction/input/output; collapse to last user->assistant pair.
    user_turns, asst_turns = [], []
    for m in messages:
        c = _render_message(m)
        if not c:
            continue
        if m.get("role") == "assistant":
            asst_turns.append(c)
        else:
            user_turns.append(c)
    instruction = (system + "\n\n" if system else "") + (user_turns[0] if user_turns else "")
    input_text = "\n".join(user_turns[1:]) if len(user_turns) > 1 else ""
    output_text = asst_turns[-1] if asst_turns else ""
    return {"instruction": instruction, "input": input_text, "output": output_text}


def main(cfg: Config) -> int:
    cleaned_dir = cfg.path("cleaned_dir")
    dataset_dir = cfg.path("dataset_dir")
    os.makedirs(dataset_dir, exist_ok=True)

    template = cfg.get("format", "template", default="chatml")
    system = cfg.get("format", "system_prompt", default="") or ""
    max_turns = cfg.get("format", "max_turns_per_example", default=0) or 0
    # Character budget per example. Conversations longer than this are split into
    # sliding windows so each fits the model's max_seq_length. 0 = no budget.
    max_chars = cfg.get("format", "max_chars_per_example", default=24000) or 0

    examples: list[Any] = []
    for fn in sorted(os.listdir(cleaned_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(cleaned_dir, fn)) as f:
            rec = json.load(f)
        msgs = rec.get("messages", [])
        windows = _window_messages(msgs, max_turns, max_chars)
        for w in windows:
            if len(w) < 2:
                continue
            examples.append(_format_window(w, template, system))

    out_file = os.path.join(dataset_dir, "train.jsonl")
    with open(out_file, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[format] wrote {len(examples)} examples to {out_file}")
    return len(examples)


def _window_messages(msgs: list[dict], max_turns: int, max_chars: int) -> list[list[dict]]:
    """Split a message list into training windows.

    - If max_turns set and exceeded: sliding windows of `max_turns` with 50%
      overlap.
    - Else if max_chars set: sliding windows that stay under the char budget
      (estimated from rendered text length), with 1-message overlap.
    - Else: the whole conversation as one window.
    """
    if max_turns and len(msgs) > max_turns:
        out = []
        step = max(1, max_turns // 2)
        for i in range(0, len(msgs), step):
            out.append(msgs[i:i + max_turns])
        return out
    if max_chars:
        out = []
        start = 0
        while start < len(msgs):
            end = start
            total = 0
            while end < len(msgs):
                est = sum(len(_render_part(p)) for p in msgs[end].get("parts", []))
                if total + est > max_chars and end > start:
                    break
                total += est
                end += 1
            out.append(msgs[start:end])
            if end >= len(msgs):
                break
            start = max(start + 1, end - 1)  # 1-message overlap
        return out
    return [msgs]


def _format_window(msgs: list[dict], template: str, system: str) -> Any:
    if template == "alpaca":
        return to_alpaca(msgs, system)
    if template == "sharegpt":
        return {"conversations": to_sharegpt(msgs, system)}
    return {"messages": to_chatml(msgs, system)}
