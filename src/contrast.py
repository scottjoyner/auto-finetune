"""Error-mining -> contrastive repair pairs (CPU-only, safe).

Turns the mined failures (``failures.jsonl``) into training signal that
directly hardens the tool-caller. For each failing session we reconstruct
the (call -> result) step timeline and look for an **in-session
self-repair** -- an error step on a target file followed, later in the SAME
session, by a *successful* step on that same target. That is a genuine,
high-signal contrastive example:

    prompt    : the conversation up to (but excluding) the erroneous call
    rejected  : the erroneous tool call
    chosen    : the later successful call on the same target

Emitted as DPO-style records (prompt / chosen / rejected) ready for a
future preference pass, plus a ``failures-taxonomy`` so we know which
tools / error markers / buckets fail most and therefore where the model
needs the most help.

The cleaned records store only each tool *call* (``input``); the results
live in the following message's text. We pair call ``i`` with message
``i+1``'s text as its result, then flag error results with the exact same
markers ``analyze`` uses (``src.analyze._is_error``). Only file-targeted
tools (write/edit/patch/...) are matchable across steps, so terminal/debug
failures without a file target fall into the taxonomy's ``no_target``
bucket rather than producing a weak pair.

Reads only cleaned/ + failures.jsonl; never touches datasets/. Safe to run
while training occupies the GPU.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

from src.analyze import _ERROR_MARKERS, _is_error
from src.verify import load_session_map


def _target_file(tool: str, inp: dict) -> Optional[str]:
    """Basename of the file a tool acts on, or None if it has no file target."""
    tool = (tool or "").lower()
    if tool in ("write", "write_file", "create_file", "update_file", "write_text"):
        p = inp.get("filePath") or inp.get("path") or inp.get("file")
        return os.path.basename(p) if isinstance(p, str) else None
    if tool in ("edit", "str_replace", "str_replace_editor"):
        p = inp.get("filePath") or inp.get("path") or inp.get("file")
        return os.path.basename(p) if isinstance(p, str) else None
    if tool == "patch":
        for line in (inp.get("patch") or "").splitlines():
            if line.startswith("+++ "):
                p = line[4:].strip()
                if p.startswith("b/"):
                    p = p[2:]
                return os.path.basename(p) if p else None
    return None


# Tools whose self-repair is worth mining as a contrastive pair. Read /
# search / web calls rarely "self-correct" into a better same-tool call,
# so they stay in the `no_target` taxonomy (quality gate).
SHELL_TOOLS = {"bash", "terminal", "execute_code", "execute_command",
                "sh", "zsh", "cmd", "shell"}


def _match_key(step: dict, include_commands: bool) -> tuple | None:
    """Pairing key for an in-session self-repair.

    File-target steps pair on the target basename (the existing,
    high-quality signal). When ``include_commands`` is set, shell-tool
    steps also pair on the tool name -- an errored command followed
    (later, same session) by a *successful* call to the same tool
    with different arguments is a genuine command self-repair.
    """
    if step.get("target"):
        return ("file", step["target"])
    if include_commands and (step.get("tool") or "").lower() in SHELL_TOOLS:
        return ("tool", (step.get("tool") or "").lower())
    return None


def _iter_steps(rec: dict) -> list[dict]:
    """Reconstruct (call -> result) steps for a session.

    A step's ``result`` is the text of the message immediately following the
    assistant message that issued the call.
    """
    msgs = rec.get("messages", [])
    steps: list[dict] = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for p in m.get("parts", []):
            if not p.get("tool"):
                continue
            tool = p.get("tool")
            inp = p.get("input") or {}
            res = ""
            if i + 1 < len(msgs):
                for pp in msgs[i + 1].get("parts", []):
                    res += str(pp.get("content") or pp.get("text") or "")
            steps.append({
                "idx": i, "tool": tool, "input": inp,
                "target": _target_file(tool, inp), "result": res,
            })
    return steps


def mine_repairs(cleaned_dir: str, failures_path: str,
                  out_path: str, include_commands: bool = False) -> tuple[int, dict]:
    """Mine in-session self-repairs from failures into DPO-style pairs.

    Returns ``(n_pairs, taxonomy)``. Writes one JSON object per pair to
    ``out_path``.

    By default only file-target self-repairs are emitted (the highest
    signal). With ``include_commands=True`` shell-tool self-repairs are
    also mined: an errored command followed (later, same session) by a
    *successful* call to the same tool with different arguments.
    """
    sessions = load_session_map(cleaned_dir)
    pairs: list[dict] = []
    tax: dict = {"by_marker": Counter(), "by_tool": Counter(),
                 "by_bucket": Counter(), "no_target": 0, "repaired": 0}

    with open(failures_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("session_id")
            bucket = d.get("bucket", "")
            rec = sessions.get(sid)
            if rec is None:
                continue
            steps = _iter_steps(rec)
            # error steps that have a matchable key (file target, or a
            # shell tool when include_commands is set)
            err_idx = [k for k, s in enumerate(steps)
                       if _is_error(s["result"]) and _match_key(s, include_commands)]
            if not err_idx:
                tax["no_target"] += 1
                continue
            for k in err_idx:
                key_k = _match_key(steps[k], include_commands)
                marked = next((m for m in _ERROR_MARKERS
                               if m in (steps[k]["result"] or "").lower()), None)
                tax["by_marker"][marked] += 1
                tax["by_tool"][steps[k]["tool"]] += 1
                tax["by_bucket"][bucket] += 1
                # later step with the SAME key, different args, no error -> fix
                for j in range(k + 1, len(steps)):
                    if (_match_key(steps[j], include_commands) == key_k
                            and steps[j]["input"] != steps[k]["input"]
                            and not _is_error(steps[j]["result"])):
                        pairs.append({
                            "session": sid,
                            "bucket": bucket,
                            "error_tool": steps[k]["tool"],
                            "target": steps[k]["target"],
                            "error_marker": marked,
                            "prompt_messages": rec["messages"][:steps[k]["idx"]],
                            "rejected_call": {"name": steps[k]["tool"],
                                              "arguments": steps[k]["input"]},
                            "chosen_call": {"name": steps[j]["tool"],
                                            "arguments": steps[j]["input"]},
                        })
                        tax["repaired"] += 1
                        break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pr in pairs:
            f.write(json.dumps(pr) + "\n")
    tax_out = {k: (dict(v) if isinstance(v, Counter) else v)
               for k, v in tax.items()}
    return len(pairs), tax_out
