"""Build training mixes from mined analysis artifacts.

- ``build_dpo_mix`` turns in-session self-repair pairs
  (``data/analysis/repairs.jsonl``) into DPO-ready rows
  (prompt / chosen / rejected). These teach a future
  contrastive run to fix bad file writes -- the exact
  failure mode the 49-task benchmark rewards correcting.
- ``validate_messages_mix`` sanity-checks an SFT ``messages`` file
  (e.g. the staged ``train.focused.jsonl`` balanced set).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.format_dataset import to_hermes


def _asst_call(call: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments", {})),
                },
            }
        ],
    }


def _normalize_prompt(raw_messages: list[dict]) -> list[dict]:
    """Raw session messages -> OpenAI chat messages.

    The mined ``prompt_messages`` are the raw live-store schema
    (``parts``-based, ``role``/``agent``/``tool`` turns). The chat
    template and ``load_dpo_dataset`` need OpenAI ``messages``
    (``role`` + ``content``, with ``tool_calls`` for assistant and
    ``tool`` turns for results) -- the same shape ``to_hermes``
    produces for the SFT finetune data. Already-normalized
    (``content``/``tool_calls``, no ``parts``) prompts pass through.
    """
    if raw_messages and any("parts" in m for m in raw_messages):
        return to_hermes(raw_messages, system="")["messages"]
    return raw_messages


def build_dpo_mix(repairs_in: str | Path, dpo_out: str | Path) -> int:
    """Convert repair pairs -> DPO rows. Returns the number of pairs written."""
    repairs_in = Path(repairs_in)
    dpo_out = Path(dpo_out)
    n = 0
    with repairs_in.open() as fin, dpo_out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            row = {
                "prompt": _normalize_prompt(r["prompt_messages"]),
                "chosen": [_asst_call(r["chosen_call"])],
                "rejected": [_asst_call(r["rejected_call"])],
            }
            fout.write(json.dumps(row) + "\n")
            n += 1
    return n


def iter_messages_mix(path: str | Path) -> Iterable[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def validate_messages_mix(path: str | Path) -> dict:
    """Return ``{"total": n, "malformed": k}`` for a messages-format file."""
    total = 0
    bad = 0
    for row in iter_messages_mix(path):
        total += 1
        msgs = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(msgs, list) or not msgs:
            bad += 1
    return {"total": total, "malformed": bad}
