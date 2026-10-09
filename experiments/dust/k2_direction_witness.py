"""Local, observational pre/post witness for *actual* K2 forward-only probes.

Invoked only by the opt-in read-only research runner. All K directions are
evaluated irrespective of features; no adapter update or model mutation.
No raw prompt, token, label, direction vector, hidden state or key is written.
This is a producer-local fsync/hash-chain witness, NOT an independent timing
attestation nor authorization to train a classifier.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import subprocess
from pathlib import Path
import time

from .predictive_probe_contract import SCHEMA, validate_jsonl
from .k2_causal_feature_history import CausalProbeHistory, FEATURES as HISTORY_FEATURE_NAMES, SCHEMA as HISTORY_SCHEMA

EVENT_SCHEMA = "auto-finetune.dust-k2-direction-events.v1"


def stable_json(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def sha256_json(record):
    return hashlib.sha256(stable_json(record).encode("utf-8")).hexdigest()


def pseudonym(key: bytes, sample: dict) -> str:
    if len(key) < 32:
        raise ValueError("episode HMAC key must be at least 32 bytes")
    tokens = sample["tokens"]
    labels = sample["labels"]
    if (not isinstance(tokens, list) or not isinstance(labels, list)
            or len(tokens) != len(labels)):
        raise ValueError("invalid source shape for keyed grouping")
    # Only the non-supervised prompt prefix determines the episode group.
    # Different target responses to the SAME prompt must never be partitioned
    # into different train/eval groups.
    first_scored = next((i for i, label in enumerate(labels)
                         if label != -100), None)
    if first_scored is None or first_scored < 1:
        raise ValueError("source lacks a masked prompt prefix")
    prefix = tokens[:first_scored]
    if any(type(x) is not int for x in prefix):
        raise ValueError("noninteger prompt token")
    return hmac.new(
        key, stable_json({"source_group_schema": "masked-prompt-prefix-v2",
                          "prefix_tokens": prefix}).encode(),
        hashlib.sha256,
    ).hexdigest()


def read_private_key(path: Path) -> bytes:
    st = path.stat()
    if st.st_mode & 0o077 or not path.is_file():
        raise PermissionError("episode HMAC key must be a mode-600 regular file")
    key = path.read_bytes()
    if len(key) < 32 or len(key) > 256:
        raise ValueError("invalid episode key length")
    return key


class SSHReceiver:
    """Request an x1-owned HMAC receipt; block until independently fsynced."""

    def __init__(self, host: str, run_id: str):
        if host != "x1-370" or len(run_id) != 32 or any(
            ch not in "0123456789abcdef" for ch in run_id
        ):
            raise ValueError("receiver host or run ID not allowlisted")
        self.host = host
        self.run_id = run_id

    def __call__(self, batch_sha256: str):
        if len(batch_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in batch_sha256
        ):
            raise ValueError("invalid PRE-batch digest")
        path = ("/home/scott/git/wt-dust-k2-direction-witness-20261009/"
                "experiments/dust/k2_receipt_receiver.py")
        command = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                   "-o", "ConnectTimeout=5", self.host, "/usr/bin/python3", path,
                   "--root", "/home/scott/git/dust-k2-receiver-custody-20261009",
                   "--receive", "--run-id", self.run_id,
                   "--batch-sha256", batch_sha256]
        result = subprocess.run(command, check=True, capture_output=True,
                                text=True, timeout=18)
        receipt = json.loads(result.stdout)
        if (receipt["run_id"] != self.run_id
                or receipt["batch_sha256"] != batch_sha256
                or len(receipt["receiver_hmac_sha256"]) != 64):
            raise RuntimeError("receiver receipt inconsistent")
        return receipt


class LocalProbeWitness:
    """Two separate, exclusively created mode-600 ledgers.

    The prepare record and its digest are flushed+fsynced before the
    actual perturbed forward. Complete records are flushed after that forward.
    Per-batch writing minimizes overhead and preserves the exact K population.
    The private episode grouping key never leaves the process.
    """

    def __init__(self, event_path: Path, completed_path: Path, *,
                 episode_hmac_sha256: str, model_revision_sha256: str,
                 sigma: float, receiver=None,
                 causal_history_population: int | None = None):
        from .predictive_probe_contract import required_hex
        if event_path.resolve() == completed_path.resolve():
            raise ValueError("event and completed files must be separate")
        self.episode = required_hex(episode_hmac_sha256, "episode")
        self.model = required_hex(model_revision_sha256, "model")
        if not 0 < sigma <= .5:
            raise ValueError("invalid probe sigma")
        self.sigma = sigma
        self.receiver = receiver
        self.remote_receipt_count = 0
        self.causal_history = (
            CausalProbeHistory(expected_population=causal_history_population,
                               episode_hmac_sha256=self.episode)
            if causal_history_population is not None else None
        )
        self.events = None
        self.completed = None
        self.completed_path = Path(completed_path)
        self.started = False
        self.pending = None
        self.chain = "0" * 64
        self.count = 0
        paths = (Path(event_path), Path(completed_path))
        for path in paths:
            if path.is_symlink() or not path.parent.is_dir():
                raise ValueError("refuse symlink/missing evidence directory")
        created = []
        try:
            for path in paths:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | getattr(os, "O_NOFOLLOW", 0), 0o600)
                created.append(os.fdopen(fd, "w", encoding="utf-8"))
            self.events, self.completed = created
        except BaseException:
            for file in created:
                file.close()
            # Never remove or overwrite partially provisioned custody files.
            raise

    def _write(self, target, record):
        target.write(stable_json(record) + "\n")
        target.flush()
        os.fsync(target.fileno())

    @staticmethod
    def _features(directions, cache, positions, sigma):
        """Eight primitive pre-probe geometric/historical-proxy signals.

        Historical gradients are intentionally UNAVAILABLE at first sight.
        This is an eight-field collection draft, NOT feature parity with the
        toy quadratic classifier or a quality-certified predictor.
        """
        import torch

        if directions.ndim != 2:
            raise ValueError("invalid direction batch")
        hidden = directions.shape[1]
        with torch.no_grad():
            reference = cache.oproj_output[:, positions, :].float()
            residual = cache.layer_input[:, positions, :].float()
            anchor = reference.mean(dim=(0, 1))
            prior = residual.mean(dim=(0, 1))
            anchor = anchor / anchor.norm().clamp_min(1e-8)
            prior = prior / prior.norm().clamp_min(1e-8)
            spread = reference.var(dim=(0, 1), unbiased=False).sqrt()
            for d in directions:
                norm = d.float().norm().clamp_min(1e-8)
                d_unit = d.float() / norm
                features = [
                    0.0,  # no earlier momentum until independent prior probes
                    float((d_unit * anchor).sum().item()),
                    float((d_unit * prior).sum().item()),
                    float((d_unit.square() * spread).sum().item()),
                    float(sigma * (d_unit.square() * spread).sum().item()),
                    float((d_unit * spread).abs().mean().item()),
                    float(sigma),
                    0.0,  # no historical momentum
                ]
                if len(features) != 8 or not all(math.isfinite(v)
                                                  and abs(v) <= 1e6
                                                  for v in features):
                    raise ValueError("invalid pre-probe features")
                yield features

    def before_batch(self, *, directions, clean, cache, positions, first):
        if self.pending is not None:
            raise RuntimeError("previous probe batch lacks post witness")
        if not (clean.ndim == 2 and clean.shape[0] == 1):
            raise ValueError("unexpected clean scored loss")
        clean_mean = float(clean.mean().item())
        rows = []
        for relative, features in enumerate(
            self._features(directions, cache, positions, self.sigma)
        ):
            index = first + relative
            base = {
                "schema": EVENT_SCHEMA, "phase": "PRE",
                "episode_hmac_sha256": self.episode,
                "candidate_index": index,
                "features_pre_probe": features,
                "clean_pre_probe": clean_mean,
                "sigma": self.sigma,
                "local_monotonic_ns": time.monotonic_ns(),
                "previous_event_sha256": self.chain,
            }
            if self.causal_history is not None:
                # PRE only: snapshot references PRIOR completed batches, never
                # current candidate forward scores. Included in PRE hash.
                base["history_feature_schema"] = HISTORY_SCHEMA
                base["history_features_pre_probe"] = list(
                    self.causal_history.preview(before_candidate=index))
            base["event_sha256"] = sha256_json(base)
            self._write(self.events, base)  # fsync BEFORE perturbed forward
            self.chain = base["event_sha256"]
            rows.append(base)
        if not rows:
            raise RuntimeError("empty direction batch")
        self.pending = rows
        if self.receiver is not None:
            # Submit only a hash of fsynced PRE events to receiver x1. The
            # receiver HMAC key stays on x1; blocking receipt arrives BEFORE
            # the caller enters the antithetic forward evaluation.
            digest = hashlib.sha256("|".join(
                row["event_sha256"] for row in rows).encode()
            ).hexdigest()
            receipt = self.receiver(digest)
            marker = {
                "schema": EVENT_SCHEMA, "phase": "RECEIPT",
                "batch_sha256": digest,
                "receiver_hmac_sha256": receipt["receiver_hmac_sha256"],
                "receiver_utc_ns": receipt["receiver_utc_ns"],
                "batch_index": receipt["batch_index"],
                "previous_event_sha256": self.chain,
            }
            marker["event_sha256"] = sha256_json(marker)
            self._write(self.events, marker)
            self.chain = marker["event_sha256"]
            self.remote_receipt_count += 1

    def after_batch(self, *, plus, minus, first):
        if self.pending is None:
            raise RuntimeError("post witness without pre witness")
        rows = self.pending
        if rows[0]["candidate_index"] != first or len(rows) != plus.shape[0]:
            raise RuntimeError("witness batch index/count mismatch")
        if plus.shape != minus.shape or plus.ndim != 2:
            raise ValueError("mismatched antithetic loss arrays")
        for offset, before in enumerate(rows):
            p = float(plus[offset].mean().item())
            m = float(minus[offset].mean().item())
            if not all(math.isfinite(v) and 0 <= v <= 100
                       for v in (p, m)):
                raise ValueError("nonfinite/unbounded forward scores")
            after = {
                "schema": EVENT_SCHEMA, "phase": "POST",
                "candidate_index": before["candidate_index"],
                "pre_event_sha256": before["event_sha256"],
                "loss_plus": p, "loss_minus": m,
                "local_monotonic_ns": time.monotonic_ns(),
                "previous_event_sha256": self.chain,
            }
            after["event_sha256"] = sha256_json(after)
            self._write(self.events, after)
            self.chain = after["event_sha256"]
            row = {
                "schema": SCHEMA,
                "episode_hmac_sha256": self.episode,
                "model_revision_sha256": self.model,
                "candidate_index": before["candidate_index"],
                "features_pre_probe": before["features_pre_probe"],
                "sigma": self.sigma,
                "loss_clean": before["clean_pre_probe"],
                "loss_plus": p, "loss_minus": m,
                "probe_time_order_attested": True,
            }
            self._write(self.completed, row)
            self.count += 1
        if self.causal_history is not None:
            # Must happen strictly after all this batch's POST loss records
            # have been fsynced, so future PRE cannot read future labels.
            self.causal_history.commit_batch(
                first_index=first,
                clean_loss=rows[0]["clean_pre_probe"],
                plus_losses=[float(plus[i].mean().item())
                             for i in range(len(rows))],
                minus_losses=[float(minus[i].mean().item())
                              for i in range(len(rows))],
                sigma=self.sigma)
        self.pending = None

    def finish(self, expected_count: int) -> dict:
        if (self.pending is not None or self.count != expected_count
                or (self.causal_history is not None
                    and self.causal_history.completed_count != expected_count)):
            raise RuntimeError("incomplete witness; do not admit derived labels")
        self.events.flush()
        self.completed.flush()
        self.events.close()
        self.completed.close()
        report = validate_jsonl(self.completed_path)
        if report["rows"] != expected_count:
            raise AssertionError("validated count does not equal population")
        report["local_pre_forward_fsync_only"] = True
        report["independent_timing_witness"] = False
        report["last_hash_chain_event_sha256"] = self.chain
        report["direction_population_unchanged"] = True
        report["receiver_precommit_receipts"] = self.remote_receipt_count
        report["independent_receiver_key_not_exported"] = (
            self.receiver is not None)
        report["receiver_hmac_verified_by_receiver"] = False
        report["causal_history_schema"] = (
            HISTORY_SCHEMA if self.causal_history is not None else None)
        report["causal_history_feature_names"] = (
            list(HISTORY_FEATURE_NAMES) if self.causal_history is not None else [])
        report["causal_history_pre_events_only"] = self.causal_history is not None
        report["causal_history_classifier_training_authorized"] = False
        return report

    def close(self):
        for file in (self.events, self.completed):
            if file and not file.closed:
                file.close()
