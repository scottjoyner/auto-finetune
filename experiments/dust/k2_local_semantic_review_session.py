"""On-host, interactive-only review of confidential K2 prompt pair candidates.

NO headless or SSH-batch mode. Human reviewer must open a REAL interactive
terminal on the machine holding the already-approved source JSONL. Private
prompt content is displayed only on that terminal and is NEVER persisted
in the receipt. Does not upload, log or print source content in non-TTY mode.

This tool DOES NOT make labels independent or cryptographically authenticate
the reviewer; self-attested reviewer identities are still an unresolved
custody threat. Data rights, semantic detector recall and split authorization
are NOT granted even when two reviews agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata

from .k2_auxiliary_prompt_clusters import digest_keyed
from .k2_data import normalized, read_pairs
from .k2_direction_witness import read_private_key
from .k2_private_semantic_review_queue import (
    SCHEMA as QUEUE_SCHEMA, validate_private_json)

SCHEMA = "auto-finetune.dust-k2-independent-human-review.v1"
CHOICES = {
    "s": "SAME_INTENT",
    "d": "DIFFERENT_INTENT",
    "u": "UNCERTAIN",
}


def display_safe_prompt(prompt: str) -> str:
    """Escape terminal instructions, invisible directional controls, line
    breaks and C0/C1 controls in untrusted prompt text before display.
    Refuse oversized text rather than silently truncate semantic context.
    """
    if (not isinstance(prompt, str) or not 1 <= len(prompt) <= 4096):
        raise ValueError("untrusted terminal prompt length")
    visible = []
    for ch in prompt:
        if unicodedata.category(ch).startswith("C"):
            visible.append("\\u%04x" % ord(ch) if ord(ch) <= 0xffff
                           else "\\U%08x" % ord(ch))
        else:
            visible.append(ch)
    return "".join(visible)


def refuse_remote_reviewer_session() -> None:
    """Defense in depth only. Absence of these flags does NOT certify
    physical console, human identity or absence of session recording.
    """
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise PermissionError("human TTY required; headless session denied")
    if any(os.environ.get(name) for name in (
            "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "MOSH_IP")):
        raise PermissionError("remote SSH/mosh reviewer session denied")


def make_lookup(*, prompts: list[str], key: bytes):
    mapping = {}
    for prompt in prompts:
        value = normalized(prompt)
        digest = digest_keyed(key, "exact-prompt-v1", value)
        if digest in mapping and mapping[digest] != value:
            raise ValueError("private prompt source identity collision")
        mapping[digest] = value
    return mapping


def validate_queue_and_source(*, queue: dict, queue_sha: str,
                              source_sha: str, prompt_lookup: dict):
    if (queue.get("schema") != QUEUE_SCHEMA
            or queue.get("source_sha256") != source_sha
            or queue.get("prior_aggregate_reconciled") is not True
            or queue.get("classifier_training_authorized") is not False):
        raise ValueError("unverified source / private review queue")
    items = queue.get("review_candidates")
    if not isinstance(items, list) or not 1 <= len(items) <= 500:
        raise ValueError("queue lacks bounded review candidates")
    ids = set()
    for item in items:
        if (not isinstance(item, dict)
                or item.get("human_label") != "UNREVIEWED"
                or item.get("independent_review_required") is not True
                or item.get("automatic_duplicate_certification") is not False
                or item["left_candidate_partition"] ==
                   item["right_candidate_partition"]):
            raise ValueError("untrusted review item or split")
        pair_id = item.get("pair_id_sha256")
        if pair_id in ids:
            raise ValueError("duplicate human review pair identifier")
        ids.add(pair_id)
        for label in ("left_prompt_hmac_sha256", "right_prompt_hmac_sha256"):
            if item[label] not in prompt_lookup:
                raise ValueError("review item not found in pinned private source")
    return len(items)


def record_decisions(queue, lookup, *,
                     reviewer_id: str, read=input, write=print):
    if (not isinstance(reviewer_id, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{3,64}", reviewer_id) is None):
        raise ValueError("reviewer ID must be a nonsecret human handle")
    decisions = []
    for index, item in enumerate(queue["review_candidates"], start=1):
        left = lookup[item["left_prompt_hmac_sha256"]]
        right = lookup[item["right_prompt_hmac_sha256"]]
        write(f"Private local semantic review {index}/{len(queue['review_candidates'])}")
        write(f"Prompt A: {display_safe_prompt(left)}")
        write(f"Prompt B: {display_safe_prompt(right)}")
        write("Labels: [s] same intent, [d] different intent, [u] uncertain")
        response = read("Select s/d/u: ").strip().lower()
        if response not in CHOICES:
            raise ValueError("missing/invalid human decision: no receipt written")
        decisions.append({
            "pair_id_sha256": item["pair_id_sha256"],
            "decision": CHOICES[response]})
    return {
        "schema": SCHEMA,
        "queue_sha256": queue["_source_sha256_from_file"],
        "reviewer_id": reviewer_id,
        "reviewer_attests_human_review": True,
        "model_generated_labels": False,
        "decisions": decisions,
    }


def run_interactive(*, queue_path: Path, queue_sha: str, source_path: Path,
                    source_sha: str, key_path: Path, reviewer_id: str,
                    receipt_path: Path):
    refuse_remote_reviewer_session()
    if (receipt_path.is_symlink() or receipt_path.exists()
            or not receipt_path.parent.is_dir()
            or receipt_path.parent.stat().st_mode & 0o077):
        raise PermissionError("new receipt in mode-700 private directory required")
    queue = validate_private_json(queue_path, queue_sha)
    src, stats = read_pairs(source_path)
    if stats["sha256"] != source_sha:
        raise ValueError("private corpus SHA drift")
    key = read_private_key(key_path)
    lookup = make_lookup(
        prompts=[pair[0] for _,(_,pair) in sorted(src.items())], key=key)
    validate_queue_and_source(
        queue=queue, queue_sha=queue_sha,
        source_sha=source_sha, prompt_lookup=lookup)
    queue["_source_sha256_from_file"] = queue_sha
    print("LOCAL HUMAN REVIEW ONLY: prompts will appear on this terminal.")
    print("Do not screen-share, record the console, or copy them into GitHub.")
    print("Decisions are not a licensed data-use or semantic recall approval.")
    receipt = record_decisions(queue, lookup, reviewer_id=reviewer_id)
    os.umask(0o077)
    with receipt_path.open("x", encoding="utf-8") as fd:
        fd.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        fd.flush()
        os.fsync(fd.fileno())
    print(f"Private review receipt saved; {len(receipt['decisions'])} decisions.")
    print("An independent human under a distinct real account must review separately.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactive-human-only", action="store_true")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--episode-key-file", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--private-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.interactive_human_only:
        parser.error("explicit interactive human-only mode required")
    run_interactive(
        queue_path=args.queue, queue_sha=args.queue_sha256,
        source_path=args.source_jsonl, source_sha=args.source_sha256,
        key_path=args.episode_key_file, reviewer_id=args.reviewer_id,
        receipt_path=args.private_receipt)


if __name__ == "__main__":
    main()
