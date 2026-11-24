#!/usr/bin/env python3
"""Convert a harvested-corpus SFT dataset into LFM3-native tool-call format.

The harvested corpora embed tool invocations inside assistant content as
    <tool_call name="bash" call_id="...">{json args}</tool_call>
while LFM3's trained dialect is
    <tool_call>
    <function=bash>
    <parameter=command>
    ...
    </parameter>
    </function>
    </tool_call>

Rewriting the dialect means a finetuned LFM3 learns OUR tools and argument
shapes in its NATIVE syntax, which is exactly what the LFM3 bench harness
scores. Non-call text (reasoning, narration, observed output) is preserved.

Usage:
    python -m scripts.make_lfm3_sft \
        --src  /path/train.combined.jsonl \
        --dst  /path/train.lfm-combined.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys

OPEN_TAG_RE = re.compile(r'<tool_call\s+name="([^"]+)"[^>]*>', re.DOTALL)

_ALIASES = {"terminal": "bash", "read_file": "read", "write_file": "write"}


def _extract_json_object_span(body: str):
    start = body.find("{")
    if start < 0:
        return None, None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(body)):
        ch = body[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[start:i + 1]), (start, i + 1)
                except ValueError:
                    return None, None
    return None, None


def _render_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _rewrite_content(content: str) -> tuple[str, int]:
    converted = 0
    out = []
    last = 0
    for m in OPEN_TAG_RE.finditer(content):
        out.append(content[last:m.start()])
        last = m.end()
        close = content.find("</tool_call>", m.end())
        body = content[m.end():close if close >= 0 else len(content)]
        if close >= 0:
            last = close + len("</tool_call>")
        else:
            last = m.end()
        args, span = _extract_json_object_span(body)
        if args is None:
            continue
        if close < 0 and span is not None:
            last = m.end() + span[1]
        else:
            last = close + len("</tool_call>") if close >= 0 else m.end()
        name = _ALIASES.get(m.group(1), m.group(1))
        params = []
        for k, v in args.items():
            params.append(f"<parameter={k}>\n{_render_value(v)}\n</parameter>\n")
        block = "<tool_call>\n<function=" + name + ">\n" + "".join(params) + "</function>\n</tool_call>"
        out.append(block)
        converted += 1
    out.append(content[last:])
    return "".join(out), converted


def convert_example(example: dict) -> tuple[dict, int]:
    msgs = example.get("messages", [])
    out, pending_assistant = [], []
    total_calls = 0

    def flush_assistant():
        if pending_assistant:
            out.append({"role": "assistant",
                        "content": "\n".join(pending_assistant).strip()})
            pending_assistant.clear()

    for m in msgs:
        role, content = m.get("role"), m.get("content") or ""
        if role == "assistant":
            new_c, n = _rewrite_content(content)
            total_calls += n
            pending_assistant.append(new_c)
            continue
        flush_assistant()
        out.append({"role": role, "content": content})
    flush_assistant()

    new = dict(example)
    new["messages"] = out
    return new, total_calls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    n_in = n_out = n_calls = 0
    with open(args.src) as src, open(args.dst, "w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            n_in += 1
            conv, ncalls = convert_example(ex)
            n_calls += ncalls
            dst.write(json.dumps(conv) + "\n")
            n_out += 1
    print(f"[make_lfm3_sft] {n_in} examples, {n_calls} tool calls "
          f"converted -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
