"""Near-duplicate detection using MinHash + LSH.

CPU-only heavy lifting for cleaning training data. Finds near-duplicate
sessions that differ only in whitespace, formatting, or minor edits.

Usage:
    python -m src.cli dedup [--threshold=0.85] [--label=<name>]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import Config


def _normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace, strip."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _shingle(text: str, k: int = 5) -> list[str]:
    """Generate k-character shingles from text."""
    if len(text) < k:
        return [text] if text else []
    return [text[i:i + k] for i in range(len(text) - k + 1)]


def _hash_shingle(shingle: str) -> int:
    """Hash a shingle to a 32-bit integer."""
    return int(hashlib.md5(shingle.encode()).hexdigest()[:8], 16)


class MinHash:
    """MinHash signature for estimating Jaccard similarity."""

    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        self.hashvalues: list[int] = []
        self._filled = False

    @classmethod
    def from_text(cls, text: str, num_perm: int = 128) -> MinHash:
        """Create MinHash from text."""
        m = cls(num_perm)
        normalized = _normalize(text)
        shingles = _shingle(normalized)
        if not shingles:
            return m

        # Use different hash seeds for each permutation
        hashvals = []
        for i in range(num_perm):
            seed = i * 0x9E3779B9  # golden ratio hash
            min_val = float("inf")
            for s in shingles:
                h = _hash_shingle(s) ^ int(seed)
                min_val = min(min_val, h)
            hashvals.append(min_val)

        m.hashvalues = hashvals
        m._filled = True
        return m

    def jaccard(self, other: MinHash) -> float:
        """Estimate Jaccard similarity between two MinHash signatures."""
        if not self._filled or not other._filled:
            return 0.0
        if len(self.hashvalues) != len(other.hashvalues):
            return 0.0
        matches = sum(1 for a, b in zip(self.hashvalues, other.hashvalues) if a == b)
        return matches / len(self.hashvalues)


class LSHIndex:
    """Locality-Sensitive Hashing index for fast near-duplicate lookup."""

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        # Band size: more bands = higher recall, fewer = higher precision
        self.bands = 16
        self.rows = num_perm // self.bands
        self.buckets: dict[tuple, list[str]] = defaultdict(list)

    def _hash_band(self, band: list[int]) -> tuple:
        """Hash a band of hash values."""
        return tuple(band)

    def add(self, doc_id: str, sig: MinHash):
        """Add a document's MinHash signature to the index."""
        if not sig._filled:
            return
        for i in range(self.bands):
            start = i * self.rows
            end = start + self.rows
            band = sig.hashvalues[start:end]
            key = self._hash_band(band)
            self.buckets[key].append(doc_id)

    def query(self, sig: MinHash) -> list[str]:
        """Find candidate near-duplicates for a signature."""
        candidates = set()
        for i in range(self.bands):
            start = i * self.rows
            end = start + self.rows
            band = sig.hashvalues[start:end]
            key = self._hash_band(band)
            candidates.update(self.buckets.get(key, []))
        return list(candidates)


def _session_text(rec: dict) -> str:
    """Extract text content from a session for comparison."""
    parts = []
    for msg in rec.get("messages", []):
        for p in msg.get("parts", []):
            if p.get("type") == "text":
                text = p.get("text", "")
                if text:
                    parts.append(text)
            elif p.get("type") == "tool":
                inp = p.get("input", {})
                if isinstance(inp, dict):
                    for v in inp.values():
                        if isinstance(v, str):
                            parts.append(v)
                out = p.get("output", "")
                if isinstance(out, str):
                    parts.append(out)
    return "\n".join(parts)


def find_duplicates(
    sessions: list[dict],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> list[tuple[str, str, float]]:
    """Find near-duplicate session pairs.

    Args:
        sessions: List of session records
        threshold: Jaccard similarity threshold (0-1)
        num_perm: Number of MinHash permutations

    Returns:
        List of (session_id_a, session_id_b, similarity) tuples
    """
    # Build MinHash signatures
    sigs: dict[str, MinHash] = {}
    for rec in sessions:
        sid = rec.get("session_id", "")
        if not sid:
            continue
        text = _session_text(rec)
        if not text.strip():
            continue
        sigs[sid] = MinHash.from_text(text, num_perm)

    # Build LSH index
    index = LSHIndex(threshold=threshold, num_perm=num_perm)
    for sid, sig in sigs.items():
        index.add(sid, sig)

    # Find duplicates
    duplicates = []
    seen_pairs: set[tuple[str, str]] = set()

    for sid, sig in sigs.items():
        candidates = index.query(sig)
        for cand in candidates:
            if cand == sid:
                continue
            pair = tuple(sorted([sid, cand]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Verify with actual Jaccard similarity
            cand_sig = sigs.get(cand)
            if cand_sig:
                sim = sig.jaccard(cand_sig)
                if sim >= threshold:
                    duplicates.append((pair[0], pair[1], sim))

    return sorted(duplicates, key=lambda x: -x[2])


def dedup_sessions(
    sessions: list[dict],
    threshold: float = 0.85,
    keep: str = "longest",
) -> tuple[list[dict], list[dict]]:
    """Remove near-duplicate sessions.

    Args:
        sessions: List of session records
        threshold: Jaccard similarity threshold
        keep: Which duplicate to keep ("longest", "shortest", "first")

    Returns:
        (kept_sessions, removed_sessions)
    """
    duplicates = find_duplicates(sessions, threshold)

    # Build removal set
    to_remove: set[str] = set()
    for sid_a, sid_b, sim in duplicates:
        # Find the sessions
        rec_a = next((s for s in sessions if s.get("session_id") == sid_a), None)
        rec_b = next((s for s in sessions if s.get("session_id") == sid_b), None)

        if not rec_a or not rec_b:
            continue

        len_a = len(json.dumps(rec_a))
        len_b = len(json.dumps(rec_b))

        if keep == "longest":
            if len_a >= len_b:
                to_remove.add(sid_b)
            else:
                to_remove.add(sid_a)
        elif keep == "shortest":
            if len_a <= len_b:
                to_remove.add(sid_b)
            else:
                to_remove.add(sid_a)
        else:  # first
            idx_a = sessions.index(rec_a)
            idx_b = sessions.index(rec_b)
            if idx_a < idx_b:
                to_remove.add(sid_b)
            else:
                to_remove.add(sid_a)

    kept = [s for s in sessions if s.get("session_id") not in to_remove]
    removed = [s for s in sessions if s.get("session_id") in to_remove]

    return kept, removed


def main(cfg: Config, label: str | None = None, threshold: float = 0.85) -> int:
    """Run near-duplicate detection and removal."""
    from src.clean import _dedup_by_session

    cleaned_dir = cfg.path("cleaned_dir")
    out_dir = cfg.path("cleaned_dir")  # overwrite in place

    # Load sessions
    all_sessions = []
    if label:
        src = os.path.join(cleaned_dir, label)
        if os.path.isdir(src):
            for fn in sorted(os.listdir(src)):
                if fn.endswith(".json"):
                    with open(os.path.join(src, fn)) as f:
                        all_sessions.append(json.load(f))
    else:
        for path in sorted(Path(cleaned_dir).rglob("*.json")):
            try:
                all_sessions.append(json.loads(path.read_text()))
            except Exception:
                continue

    if not all_sessions:
        print("[dedup] no sessions found")
        return 0

    print(f"[dedup] {len(all_sessions)} sessions, threshold={threshold}")

    # First pass: exact session_id dedup
    deduped = list(_dedup_by_session(all_sessions).values())
    exact_removed = len(all_sessions) - len(deduped)
    if exact_removed:
        print(f"[dedup] removed {exact_removed} exact session_id duplicates")

    # Second pass: near-duplicate detection
    kept, removed = dedup_sessions(deduped, threshold=threshold)

    # Write results
    written = 0
    for rec in kept:
        sid = rec.get("session_id", "unknown")
        # Determine output path
        if label:
            out_path = os.path.join(out_dir, label, f"{sid}.json")
        else:
            source = rec.get("source", "unknown")
            out_path = os.path.join(out_dir, source, f"{sid}.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(rec, f)
        written += 1

    # Write removed sessions to a separate file for inspection
    removed_path = os.path.join(cfg.path("analysis_dir"), "near-duplicates.jsonl")
    os.makedirs(os.path.dirname(removed_path), exist_ok=True)
    with open(removed_path, "w") as f:
        for rec in removed:
            f.write(json.dumps({
                "session_id": rec.get("session_id"),
                "source": rec.get("source"),
                "reason": "near-duplicate",
            }) + "\n")

    print(f"[dedup] kept {written} sessions, removed {len(removed)} near-duplicates")
    print(f"[dedup] removed sessions written to {removed_path}")
    return written
