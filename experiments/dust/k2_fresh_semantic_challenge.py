"""Pre-registered FRESH synthetic semantic challenge, independent of PR #33 examples.

Forty distinct anchors with (same-intent rewrite, related-topic different-intent)
cases. No historical PR #33 pair appears here, and no local No Robots
prompt is used. This is SYNTHETIC model diagnostic data, not human-reviewed
ground truth. Freezing the cases BEFORE running either model avoids adapting
them to the resulting scores; nevertheless independence from test-set author
bias or prior model-selection bias is NOT established.

This module never trains any model, selects training examples, or alters
K2 random directions. Both encoders are fixed to locally cached checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from .k2_offline_semantic_audit import encode, full_sha

SCHEMA = "auto-finetune.dust-k2-fresh-semantic-challenge.v1"
THRESHOLDS = (0.75, 0.85, 0.92)
# (anchor, same-intent rewrite, related-topic different-intent)
CASES = (
    ("List all filenames in a directory recursively.",
     "Enumerate every file path beneath a folder, including nested folders.",
     "Remove every file in a directory tree."),
    ("Detect duplicate rows in a CSV table.",
     "Find repeated records within comma-separated data.",
     "Remove columns that contain missing values from a table."),
    ("Convert UTC timestamps to a local timezone.",
     "Translate timestamps expressed in UTC into the target region's clock time.",
     "Determine elapsed seconds between two UTC timestamps."),
    ("Find the median of numeric observations.",
     "Calculate the middle value after ordering a list of numbers.",
     "Compute the arithmetic mean of a numeric list."),
    ("Check whether a TCP port is listening.",
     "Determine if a server has a process accepting TCP connections on a port.",
     "Close a TCP port in the host firewall."),
    ("Count unique identifiers in a log file.",
     "Determine the number of distinct IDs appearing in an event log.",
     "Sort log events in reverse chronological order."),
    ("Generate a checksum for a directory manifest.",
     "Produce digest values for the files described in a folder inventory.",
     "Verify whether directory permissions allow writes."),
    ("Explain when to use a read-only database transaction.",
     "Describe scenarios that benefit from querying a database without writes.",
     "Explain when to use a serializable read-write transaction."),
    ("Write a retry policy with exponential backoff.",
     "Implement retries whose delay doubles between successive failures.",
     "Implement a fixed-delay retry policy with constant sleep time."),
    ("Measure a service's ninety-ninth-percentile latency.",
     "Calculate the p99 duration for observed service requests.",
     "Compute the total request throughput per second."),
    ("Configure a process to restart after crashing.",
     "Set up automatic service recovery following unexpected process exit.",
     "Disable automatic restarts after a process failure."),
    ("Identify orphaned files in an object store.",
     "Locate stored objects that have no corresponding metadata references.",
     "List object-store files that were uploaded during the last hour."),
    ("Compare two JSON documents ignoring object key order.",
     "Test whether JSON objects have equal content despite reordered keys.",
     "Compare JSON documents preserving whitespace and property order."),
    ("Keep separate audit logs for two tenants.",
     "Ensure each tenant's audit events remain isolated from the other tenant.",
     "Merge every tenant's events into a single shared audit stream."),
    ("Find all graph nodes reachable from a root node.",
     "Traverse the graph to enumerate vertices accessible from a given start vertex.",
     "Find all nodes that cannot reach the designated root node."),
    ("Check whether a Kubernetes pod is ready.",
     "Determine if a pod currently satisfies its readiness condition.",
     "Determine whether a Kubernetes pod has been deleted."),
    ("Explain how to avoid integer overflow.",
     "Describe methods for keeping arithmetic operations within numeric bounds.",
     "Explain how to avoid a floating point loss of precision."),
    ("Compute the intersection of two sets.",
     "Find the items present in both of the supplied sets.",
     "Return the symmetric difference between two sets."),
    ("Rotate an API credential without downtime.",
     "Replace a service API secret while keeping requests continuously available.",
     "Revoke every service credential and force a complete outage."),
    ("Test a parser with malformed input.",
     "Check a parsing implementation against deliberately invalid data.",
     "Benchmark parser throughput on valid input."),
    ("Determine why an application exceeds its disk quota.",
     "Investigate which application files or directories are consuming allocated storage.",
     "Determine why an application exceeds its CPU allocation."),
    ("Explain optimistic concurrency control.",
     "Describe version-checking before committing concurrent updates.",
     "Explain pessimistic row locking for transactions."),
    ("Find a memory leak in a long-running daemon.",
     "Investigate steadily increasing allocations in a persistent background service.",
     "Detect whether a daemon repeatedly loses network connections."),
    ("Compress a text archive without losing content.",
     "Apply a lossless compression method to a bundle of text files.",
     "Reduce the archive by deleting infrequently used text documents."),
    ("Identify mismatched foreign keys between tables.",
     "Find child records referencing parent rows that do not exist.",
     "Identify primary key collisions inside a single table."),
    ("Sort database results by creation time.",
     "Order query rows using the timestamp when each record was created.",
     "Filter query rows to those created on the current day."),
    ("Convert degrees Celsius into Fahrenheit.",
     "Translate a temperature from the Celsius scale to the Fahrenheit scale.",
     "Convert a distance in kilometers into miles."),
    ("Explain the purpose of a circuit breaker.",
     "Describe how a service stops calling a failing dependency temporarily.",
     "Explain how to automatically retry a failed request forever."),
    ("Detect race conditions in asynchronous code.",
     "Find concurrency bugs caused by unsynchronized asynchronous operations.",
     "Find syntax errors that prevent asynchronous code from compiling."),
    ("Verify a backup can be restored successfully.",
     "Test recovery from a saved backup by restoring the stored data.",
     "Check that a backup job reports success without restoring anything."),
    ("List available CPU instruction extensions.",
     "Enumerate supported processor instruction-set features.",
     "Display how many physical disks are connected to the computer."),
    ("Detect expired TLS certificates on endpoints.",
     "Identify HTTPS servers presenting certificates past their validity date.",
     "Find HTTPS servers using certificates issued by a particular authority."),
    ("Calculate a moving average over sensor readings.",
     "Compute a rolling mean across sequential sensor values.",
     "Compute the maximum observed sensor reading."),
    ("Separate training data by source document.",
     "Partition machine learning examples so one source document belongs to a single split.",
     "Randomly distribute sentences from every document among all splits."),
    ("Explain what a database migration rollback does.",
     "Describe how to reverse a previously applied database schema change.",
     "Describe how to apply an additional forward-only schema migration."),
    ("Test a command-line tool with no arguments.",
     "Check how a CLI behaves when invoked without positional or flag options.",
     "Measure execution speed when a CLI receives many valid arguments."),
    ("Find repeated messages in a distributed queue.",
     "Detect messages that were delivered more than once through the queue.",
     "Determine the number of messages currently awaiting delivery."),
    ("Verify that a Linux mount is read-only.",
     "Check whether a mounted filesystem disallows file modifications.",
     "Check whether a mounted filesystem is encrypted."),
    ("Explain why a schema needs a version number.",
     "Describe how explicit schema revisions support compatibility across changes.",
     "Explain why database records need unique identifiers."),
    ("Audit which processes have open file descriptors.",
     "Inspect processes to determine the files and sockets they currently hold open.",
     "List processes ordered by percentage of CPU utilization."),
)
assert len(CASES) == 40


def validate_frozen_challenge():
    if len(CASES) != 40 or any(
        len(entry) != 3 or any(not isinstance(x, str) or len(x) < 20 for x in entry)
        for entry in CASES
    ):
        raise ValueError("unexpected 40-topic synthetic challenge shape")
    texts = [x for case in CASES for x in case]
    if len(set(texts)) != 120:
        raise ValueError("synthetic challenge repeats a prompt")
    from .k2_semantic_challenge_contract import PAIRS
    prior = {x for entry in PAIRS for x in entry[:2]}
    if prior.intersection(texts):
        raise ValueError("independent challenge repeats original PR33 examples")
    return hashlib.sha256(
        json.dumps(CASES, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def score_frozen_cosines(scores: list[tuple[float, float]]) -> dict:
    if len(scores) != 40 or any(
        len(pair) != 2 or any(
            type(x) not in (float, int) or not math.isfinite(x)
            or not -1.00001 <= x <= 1.00001 for x in pair)
        for pair in scores
    ):
        raise ValueError("must report 40 valid positive/negative cosine pairs")
    pos = [float(x) for x, _ in scores]
    neg = [float(x) for _, x in scores]
    rank_wins = sum(a > b for a, b in scores)
    ties = sum(a == b for a, b in scores)
    auc = sum(1 if x > y else .5 if x == y else 0
              for x in pos for y in neg) / 1600
    outcomes = {}
    for t in THRESHOLDS:
        tp = sum(x >= t for x in pos)
        fp = sum(x >= t for x in neg)
        outcomes[str(t)] = {
            "true_positives": tp,
            "false_negatives": 40 - tp,
            "false_positives": fp,
            "true_negatives": 40 - fp,
            "positive_recall": tp / 40,
            "hard_negative_false_positive_rate": fp / 40,
            "prespecified_acceptance": tp >= 32 and fp <= 8,
        }
    return {
        "schema": SCHEMA,
        "challenge_cases_sha256": validate_frozen_challenge(),
        "synthetic_positive_count": 40,
        "synthetic_topic_matched_hard_negative_count": 40,
        "matched_topic_rank_wins": rank_wins,
        "matched_topic_rank_ties": ties,
        "positive_vs_negative_pairwise_auc": auc,
        "mean_matched_topic_margin": sum(a-b for a,b in scores) / 40,
        "frozen_threshold_results": outcomes,
        "case_labels_independently_human_adjudicated": False,
        "challenge_not_trained_on": True,
        "source_corpus_independence_certified": False,
        "classifier_training_authorized": False,
        "optimizer_or_production_authorized": False,
    }


def run_challenge(model_dir: Path, *, input_prefix: str) -> dict:
    if input_prefix not in ("", "query: "):
        raise ValueError("unsupported encoder prefix")
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise ValueError("encoder must be pre-existing local snapshot")
    config = model_dir / "config.json"
    weights = model_dir / "model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise ValueError("offline checkpoint absent")
    # Freeze both data and prespecified cutoffs before running either encoder.
    frozen_sha = validate_frozen_challenge()
    texts = [input_prefix + text for case in CASES for text in case]
    vectors = encode(model_dir, texts)
    results = [
        (float(vectors[i] @ vectors[i+1]),
         float(vectors[i] @ vectors[i+2]))
        for i in range(0, len(vectors), 3)
    ]
    scores = score_frozen_cosines(results)
    if scores["challenge_cases_sha256"] != frozen_sha:
        raise RuntimeError("frozen cases mutated during evaluation")
    return {
        **scores,
        "mode": "OFFLINE_UNTRAINED_ENCODER_CHALLENGE_RESEARCH_ONLY",
        "encoder_config_sha256": full_sha(config),
        "encoder_weights_sha256": full_sha(weights),
        "encoder_input_prefix": input_prefix,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-new-challenge-only", action="store_true")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--input-prefix", choices=("", "query: "), default="")
    p.add_argument("--private-output", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.frozen_new_challenge_only:
        p.error("explicit frozen new-challenge mode required")
    output = args.private_output
    if (output.exists() or output.is_symlink() or
            not output.parent.is_dir() or output.parent.stat().st_mode & 0o077):
        p.error("must exclusive-create in existing private mode-700 directory")
    report = run_challenge(args.model_dir, input_prefix=args.input_prefix)
    os.umask(0o077)
    with output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
