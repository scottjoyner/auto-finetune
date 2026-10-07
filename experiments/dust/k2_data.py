"""Privacy-preserving, deterministic selection for K2 research holdouts.

Scans existing JSONL in place, selects paired user/assistant examples by SHA-256
rank, refuses exact prompt overlap, and emits no user/assistant text in reports.
Selection must not be mistaken for an exhaustive semantic leakage audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def normalized(value: str) -> str:
    return " ".join(value.split()).strip().casefold()


def extract_pairs(row: object, max_chars: int = 4000):
    """Yield every adjacent plain user->assistant pair in a conversation."""
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        return
    messages = row["messages"]
    for user, assistant in zip(messages, messages[1:]):
        if not isinstance(user, dict) or not isinstance(assistant, dict):
            continue
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            continue
        if user.get("tool_calls") or assistant.get("tool_calls"):
            continue
        prompt, reply = user.get("content"), assistant.get("content")
        if not isinstance(prompt, str) or not isinstance(reply, str):
            continue
        prompt, reply = prompt.strip(), reply.strip()
        if not (8 <= len(prompt) <= max_chars and 8 <= len(reply) <= max_chars):
            continue
        if "\x00" in prompt or "\x00" in reply:
            continue
        yield prompt, reply


def read_pairs(file: Path) -> tuple[dict, dict]:
    """Text stays only in this process; evidence includes counts and digests."""
    entries = {}
    sha = hashlib.sha256()
    lines = 0
    eligible = 0
    with file.open("rb") as source:
        for blob in source:
            lines += 1
            sha.update(blob)
            try:
                pairs = list(extract_pairs(json.loads(blob)))
            except (ValueError, UnicodeError, TypeError):
                pairs = []
            for pair in pairs:
                eligible += 1
                prompt, response = pair
                user_hash = hashlib.sha256(normalized(prompt).encode()).hexdigest()
                pair_hash = hashlib.sha256(
                    (normalized(prompt) + "\x00" + normalized(response)).encode()
                ).hexdigest()
                if pair_hash not in entries:
                    entries[pair_hash] = (user_hash, pair)
    return entries, {"sha256": sha.hexdigest(), "lines": lines,
                     "eligible_pairs": eligible, "unique_pairs": len(entries),
                     "bytes": file.stat().st_size}


def tokenize_pair(tokenizer, pair, max_tokens: int):
    prompt, response = pair
    user = {"role": "user", "content": prompt}
    full = tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": response}],
        tokenize=True, add_generation_prompt=False,
    )["input_ids"]
    empty = tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": ""}],
        tokenize=True, add_generation_prompt=False,
    )["input_ids"]
    if not isinstance(full, list) or not isinstance(empty, list):
        raise TypeError("Unexpected tokenizer output; require token ID lists")
    common = 0
    while common < min(len(full), len(empty)) and full[common] == empty[common]:
        common += 1
    if not (4 <= common < len(full) - 1):
        return None
    if not (4 <= len(full) <= max_tokens):
        return None
    if len(full) - common < 3:
        return None
    return {"tokens": full, "labels": [-100] * common + full[common:],
            "assistant_tokens": len(full) - common}


def select_disjoint(train_file: Path, eval_file: Path, tokenizer, *,
                    train_count: int = 12, eval_count: int = 6,
                    max_tokens: int = 128, eval_max_tokens: int = 512
                    ) -> tuple[list, list, dict]:
    if train_file.resolve() == eval_file.resolve():
        raise ValueError("Train and eval files must differ")
    if not 1 <= train_count <= 64 or not 1 <= eval_count <= 32:
        raise ValueError("Dataset counts outside research bounds")
    if not 16 <= max_tokens <= 256 or not 16 <= eval_max_tokens <= 512:
        raise ValueError("Research train/eval token budgets exceeded")
    train_entries, train_stats = read_pairs(train_file)
    eval_entries, eval_stats = read_pairs(eval_file)
    if not train_entries or not eval_entries:
        raise ValueError("No eligible records")
    # Hold out every example sharing an exact normalized user prompt, not
    # merely the paired response, from the training source.
    train_prompts = {user_hash for user_hash, _ in train_entries.values()}
    train, heldout = [], []
    train_digests, eval_digests = [], []
    for digest, (_, pair) in sorted(train_entries.items()):
        if len(train) == train_count:
            break
        try:
            row = tokenize_pair(tokenizer, pair, max_tokens)
        except (ValueError, TypeError, KeyError, AttributeError):
            row = None
        if row:
            train.append(row)
            train_digests.append(digest)
    rejected_overlap = sum(
        user_hash in train_prompts for user_hash, _ in eval_entries.values())
    eval_candidates = []
    for digest, (user_hash, pair) in sorted(eval_entries.items()):
        if user_hash in train_prompts:
            continue
        try:
            row = tokenize_pair(tokenizer, pair, eval_max_tokens)
        except (ValueError, TypeError, KeyError, AttributeError):
            row = None
        if row:
            eval_candidates.append((digest, user_hash, row))
    prompt_counts: dict[str, int] = {}
    for _, user_hash, _ in eval_candidates:
        prompt_counts[user_hash] = prompt_counts.get(user_hash, 0) + 1
    rejected_duplicate_prompt = sum(
        count - 1 for count in prompt_counts.values() if count > 1)
    selected_eval_prompts: set[str] = set()
    for digest, user_hash, row in eval_candidates:
        if user_hash in selected_eval_prompts:
            continue
        heldout.append(row)
        eval_digests.append(digest)
        selected_eval_prompts.add(user_hash)
        if len(heldout) == eval_count:
            break
    if len(train) != train_count or len(heldout) != eval_count:
        raise ValueError("Not enough disjoint, tokenizable examples")
    meta = {
        "selection": "sha256-ranked-unique-normalized-prompt.v2",
        "train_source": train_stats, "eval_source": eval_stats,
        "train_rows": len(train), "eval_rows": len(heldout),
        "eval_unique_prompt_hashes": len(selected_eval_prompts),
        "excluded_eval_prompts_shared_with_train_source": rejected_overlap,
        "excluded_eval_pairs_repeating_selected_prompt": rejected_duplicate_prompt,
        "train_selected_pair_sha256": train_digests,
        "eval_selected_pair_sha256": eval_digests,
        "eval_selected_prompt_sha256": sorted(selected_eval_prompts),
        "train_assistant_tokens": sum(r["assistant_tokens"] for r in train),
        "eval_assistant_tokens": sum(r["assistant_tokens"] for r in heldout),
        "train_max_seq_tokens": max_tokens, "eval_max_seq_tokens": eval_max_tokens,
        "warning": (
            "Exact normalized prompt separation and within-eval prompt uniqueness "
            "only; semantic near-duplicates remain unproven"
        ),
    }
    return train, heldout, meta
