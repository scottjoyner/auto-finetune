"""Independent, read-only replay of K2 causal-history PRE/POST evidence.

Recomputes each pre-probe history vector using ONLY preceding completed
POST losses, checks the producer event hash chain and exact derived-label
agreement. Never loads K2, uses network, trains, changes sample selection
or asserts independently isolated receiver credentials. No raw prompts,
tokens or direction vectors are accepted or emitted.

python -m experiments.dust.k2_causal_evidence_verify --verify-only \
    --events /private/events.jsonl --derived /private/derived.jsonl \
    --summary /private/summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .k2_causal_feature_history import CausalProbeHistory, SCHEMA as HIST_SCHEMA
from .k2_direction_witness import EVENT_SCHEMA, sha256_json
from .predictive_probe_contract import validate_record

MAX_FILE_BYTES = 1024 * 1024
BATCH_LIMIT = 4
# Whitelists also reject accidental sensitive payloads in producer logs.
PRE_FIELDS = frozenset({
    "schema", "phase", "episode_hmac_sha256", "candidate_index",
    "features_pre_probe", "history_feature_schema",
    "history_features_pre_probe", "clean_pre_probe", "sigma",
    "local_monotonic_ns", "previous_event_sha256",
})
POST_FIELDS = frozenset({
    "schema", "phase", "candidate_index", "pre_event_sha256",
    "loss_plus", "loss_minus", "local_monotonic_ns",
    "previous_event_sha256",
})
RECEIPT_FIELDS = frozenset({
    "schema", "phase", "batch_sha256", "receiver_hmac_sha256",
    "receiver_utc_ns", "batch_index", "previous_event_sha256",
})


def read_jsonl(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("untrusted evidence path")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("unbounded evidence file")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines) > 512:
        raise ValueError("empty/oversized witness")
    return [json.loads(s) for s in lines]


def _matches(values, expected):
    return (
        isinstance(values, list)
        and len(values) == len(expected)
        and all(type(a) in (int, float) and math.isfinite(a)
                and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
                for a, b in zip(values, expected, strict=True))
    )


def verify(events_path: Path, derived_path: Path, summary_path: Path) -> dict:
    events = read_jsonl(events_path)
    labels = read_jsonl(derived_path)
    if len(labels) not in (8, 16, 32, 64):
        raise ValueError("unexpected full K population")
    if (summary_path.is_symlink() or not summary_path.is_file()
            or summary_path.stat().st_size > 64 * 1024):
        raise ValueError("untrusted or oversized observation summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("population") != len(labels)
        or summary.get("causal_history_v2_observed") is not True
        or summary.get("causal_history_v2_training_authorized") is not False
        or summary.get("base_weights_unchanged") is not True
        or summary.get("adapter_weights_unchanged") is not True
        or summary.get("optimizer_updates") != 0
        or summary.get("backward_calls") != 0
    ):
        raise ValueError("model-read-only summary gates not met")
    episode = summary.get("source_episode_hmac_sha256")
    history = CausalProbeHistory(
        expected_population=len(labels), episode_hmac_sha256=episode)
    chain = "0" * 64
    pending_pre = []
    pending_post = []
    receipts = 0
    receipt_current = False
    post_started = False
    completed = 0
    beneficial = 0
    for event in events:
        event = dict(event)
        observed_sha = event.pop("event_sha256", None)
        if not isinstance(observed_sha, str) or observed_sha != sha256_json(event):
            raise ValueError("producer event hash mismatch")
        if (event.get("schema") != EVENT_SCHEMA
                or event.get("previous_event_sha256") != chain):
            raise ValueError("producer hash chain or schema mismatch")
        chain = observed_sha
        phase = event.get("phase")
        if phase == "PRE":
            if set(event) != PRE_FIELDS:
                raise ValueError("unsafe or incomplete PRE event fields")
            if post_started or receipt_current or len(pending_pre) >= BATCH_LIMIT:
                raise ValueError("bad PRE batch state")
            index = event.get("candidate_index")
            if index != completed + len(pending_pre):
                raise ValueError("PRE index skips or repeats candidate")
            if event.get("episode_hmac_sha256") != episode:
                raise ValueError("PRE source-group mismatch")
            if event.get("history_feature_schema") != HIST_SCHEMA:
                raise ValueError("unexpected history schema")
            expected = history.preview(before_candidate=index)
            if not _matches(event.get("history_features_pre_probe"), expected):
                raise ValueError("PRE history contains future, stale or forged values")
            if len(event.get("features_pre_probe", [])) != 8:
                raise ValueError("original direction features missing")
            if pending_pre and (
                event.get("clean_pre_probe") != pending_pre[0]["clean_pre_probe"]
                or event.get("sigma") != pending_pre[0]["sigma"]
            ):
                raise ValueError("clean/sigma changed inside batch")
            pending_pre.append({**event, "event_sha256": observed_sha})
        elif phase == "RECEIPT":
            if set(event) != RECEIPT_FIELDS:
                raise ValueError("unsafe or incomplete RECEIPT fields")
            if receipt_current or post_started or len(pending_pre) != BATCH_LIMIT:
                raise ValueError("receipt outside complete PRE batch")
            digest = hashlib.sha256(
                "|".join(p["event_sha256"] for p in pending_pre).encode()
            ).hexdigest()
            if event.get("batch_sha256") != digest:
                raise ValueError("PRE receipt digest inconsistent")
            receipt_current = True
            receipts += 1
        elif phase == "POST":
            if set(event) != POST_FIELDS:
                raise ValueError("unsafe or incomplete POST event fields")
            if not pending_pre or len(pending_post) >= len(pending_pre):
                raise ValueError("POST without complete PRE record")
            post_started = True
            offset = len(pending_post)
            corresponding = pending_pre[offset]
            index = completed + offset
            if (
                event.get("candidate_index") != index
                or event.get("pre_event_sha256")
                    != corresponding["event_sha256"]
                or event.get("local_monotonic_ns", 0)
                    <= corresponding["local_monotonic_ns"]
            ):
                raise ValueError("POST causal index or PRE reference mismatch")
            actual = validate_record(labels[index])
            if (
                actual["candidate"] != index
                or actual["episode"] != episode
                or actual["model_revision_sha256"]
                    != summary["model_revision_sha256"]
                or tuple(labels[index]["features_pre_probe"])
                    != tuple(corresponding["features_pre_probe"])
                or labels[index]["loss_clean"]
                    != corresponding["clean_pre_probe"]
                or labels[index]["loss_plus"] != event.get("loss_plus")
                or labels[index]["loss_minus"] != event.get("loss_minus")
                or labels[index]["sigma"] != corresponding["sigma"]
            ):
                raise ValueError("derived classifier label differs from POST/PRE")
            pending_post.append(event)
            beneficial += actual["label"]
            if len(pending_post) == len(pending_pre):
                if len(pending_pre) != BATCH_LIMIT:
                    raise ValueError("partial batch was not expected")
                history.commit_batch(
                    first_index=completed,
                    clean_loss=pending_pre[0]["clean_pre_probe"],
                    plus_losses=[v["loss_plus"] for v in pending_post],
                    minus_losses=[v["loss_minus"] for v in pending_post],
                    sigma=pending_pre[0]["sigma"])
                completed += len(pending_pre)
                pending_pre = []
                pending_post = []
                post_started = False
                receipt_current = False
        else:
            raise ValueError("unexpected witness event phase")
    if pending_pre or pending_post or completed != len(labels):
        raise ValueError("incomplete K direction evidence")
    if receipts not in (0, len(labels) // BATCH_LIMIT):
        raise ValueError("mixed local and receipt batches")
    return {
        "schema": "auto-finetune.dust-k2-causal-history-audit.v1",
        "result": "PASS_PRODUCER_CAUSAL_REPLAY_ONLY",
        "observed_directions": completed,
        "beneficial_plus_directions": beneficial,
        "history_schema": HIST_SCHEMA,
        "history_pre_only_recomputed": True,
        "original_label_schema_preserved": True,
        "producer_event_chain_verified": True,
        "receiver_receipt_count_not_independently_signed_here": receipts,
        "receiver_signing_key_isolation_accepted": False,
        "classifier_training_authorized": False,
        "optimizer_admission": False,
        "events_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(),
        "derived_sha256": hashlib.sha256(derived_path.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--derived", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.verify_only:
        parser.error("explicit --verify-only is required")
    print(json.dumps(verify(args.events, args.derived, args.summary),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
