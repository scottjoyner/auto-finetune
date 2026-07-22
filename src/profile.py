"""Dataset profiling and analysis with token statistics and clustering.

CPU-only heavy lifting for understanding training data composition.
Computes token length distributions, language breakdown, code complexity,
and topic clustering.

Usage:
    python -m src.cli profile [--out=<dir>] [--label=<name>]
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.config import Config


def _estimate_tokens(text: str) -> int:
    """Rough token count estimate (words * 1.3 for English code)."""
    words = len(text.split())
    return int(words * 1.3)


def _code_complexity(text: str) -> dict:
    """Estimate code complexity metrics."""
    lines = text.split("\n")
    n_lines = len(lines)
    n_blank = sum(1 for l in lines if not l.strip())
    n_comments = sum(1 for l in lines if l.strip().startswith(("#", "//", "/*")))
    n_code = n_lines - n_blank - n_comments

    # Cyclomatic complexity approximation
    control_flow = (
        len(re.findall(r"\bif\b", text)) +
        len(re.findall(r"\belif\b", text)) +
        len(re.findall(r"\belse\b", text)) +
        len(re.findall(r"\bfor\b", text)) +
        len(re.findall(r"\bwhile\b", text)) +
        len(re.findall(r"\bexcept\b", text)) +
        len(re.findall(r"\band\b", text)) +
        len(re.findall(r"\bor\b", text))
    )

    # Nesting depth
    max_depth = 0
    current_depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "if ", "for ", "while ", "try:", "except")):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif stripped == "" or stripped.startswith("return"):
            current_depth = max(0, current_depth - 1)

    return {
        "n_lines": n_lines,
        "n_code_lines": n_code,
        "n_blank_lines": n_blank,
        "n_comment_lines": n_comments,
        "cyclomatic_complexity": control_flow + 1,
        "max_nesting_depth": max_depth,
    }


def _language_detect(text: str) -> str:
    """Simple language detection based on file extensions and patterns."""
    # Check for common language patterns
    if re.search(r"def \w+\(.*\):|import \w+|from \w+ import", text):
        return "python"
    if re.search(r"function \w+\(|const \w+ =|let \w+ =|import \{.*\} from", text):
        return "javascript"
    if re.search(r"func \w+\(|package \w+|import \(", text):
        return "go"
    if re.search(r"fn \w+\(|let mut |impl \w+", text):
        return "rust"
    if re.search(r"public class |private |@Override", text):
        return "java"
    if re.search(r"#include|#define|void \w+\(", text):
        return "c"
    if re.search(r"<html|<div|<script", text):
        return "html"
    if re.search(r"SELECT |INSERT |UPDATE |DELETE ", text, re.IGNORECASE):
        return "sql"
    if re.search(r"---|\*\*|```", text):
        return "markdown"
    return "text"


def _file_type_stats(sessions: list[dict]) -> dict:
    """Analyze file types across sessions."""
    ext_counter: Counter = Counter()
    lang_counter: Counter = Counter()

    for rec in sessions:
        for msg in rec.get("messages", []):
            for p in msg.get("parts", []):
                # Tool inputs with file paths
                inp = p.get("input", {})
                if isinstance(inp, dict):
                    for key in ("filePath", "path", "filename", "file"):
                        val = inp.get(key)
                        if isinstance(val, str):
                            ext = os.path.splitext(val)[1].lower()
                            if ext:
                                ext_counter[ext] += 1

                # Code content
                if p.get("type") == "text":
                    text = p.get("text", "")
                    lang = _language_detect(text)
                    lang_counter[lang] += 1

    return {
        "extensions": dict(ext_counter.most_common(20)),
        "languages": dict(lang_counter.most_common(15)),
    }


def _session_length_stats(sessions: list[dict]) -> dict:
    """Compute token length statistics."""
    lengths = []
    msg_counts = []
    tool_counts = []

    for rec in sessions:
        total_tokens = 0
        n_msgs = 0
        n_tools = 0

        for msg in rec.get("messages", []):
            n_msgs += 1
            for p in msg.get("parts", []):
                if p.get("type") == "text":
                    total_tokens += _estimate_tokens(p.get("text", ""))
                elif p.get("type") == "tool":
                    n_tools += 1
                    inp = p.get("input", {})
                    if isinstance(inp, dict):
                        for v in inp.values():
                            if isinstance(v, str):
                                total_tokens += _estimate_tokens(v)
                    out = p.get("output", "")
                    if isinstance(out, str):
                        total_tokens += _estimate_tokens(out)

        lengths.append(total_tokens)
        msg_counts.append(n_msgs)
        tool_counts.append(n_tools)

    if not lengths:
        return {}

    return {
        "n_sessions": len(sessions),
        "total_tokens": sum(lengths),
        "avg_tokens": round(sum(lengths) / len(lengths)),
        "median_tokens": sorted(lengths)[len(lengths) // 2],
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "p90_tokens": sorted(lengths)[int(len(lengths) * 0.9)],
        "p95_tokens": sorted(lengths)[int(len(lengths) * 0.95)],
        "avg_messages": round(sum(msg_counts) / len(msg_counts), 1),
        "avg_tools": round(sum(tool_counts) / len(tool_counts), 1),
        "token_histogram": _histogram(lengths, bins=10),
    }


def _histogram(values: list[int], bins: int = 10) -> dict:
    """Create a histogram of values."""
    if not values:
        return {}
    min_val = min(values)
    max_val = max(values)
    if min_val == max_val:
        return {f"{min_val}": len(values)}

    bin_size = (max_val - min_val) / bins
    hist: dict[str, int] = {}
    for v in values:
        bin_idx = min(int((v - min_val) / bin_size), bins - 1)
        bin_start = int(min_val + bin_idx * bin_size)
        bin_end = int(min_val + (bin_idx + 1) * bin_size)
        key = f"{bin_start}-{bin_end}"
        hist[key] = hist.get(key, 0) + 1
    return hist


def _topic_clusters(sessions: list[dict], max_clusters: int = 15) -> dict:
    """Simple topic clustering using keyword extraction."""
    # Extract keywords from user messages
    keywords: Counter = Counter()
    session_keywords: list[set[str]] = []

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "don", "now", "and", "but", "or", "if", "while", "that",
        "this", "it", "its", "what", "which", "who", "whom", "these", "those",
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
        "she", "her", "they", "them", "their", "about", "up", "also",
    }

    for rec in sessions:
        texts = []
        for msg in rec.get("messages", []):
            if msg.get("role") == "user":
                for p in msg.get("parts", []):
                    if p.get("type") == "text":
                        texts.append(p.get("text", ""))

        words = set()
        for text in texts:
            tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
            words.update(t for t in tokens if t not in stop_words)

        session_keywords.append(words)
        keywords.update(words)

    # Simple clustering: group by most common keywords
    top_keywords = [kw for kw, _ in keywords.most_common(max_clusters)]
    clusters: dict[str, list[int]] = defaultdict(list)

    for idx, words in enumerate(session_keywords):
        best_cluster = "other"
        best_overlap = 0
        for kw in top_keywords:
            if kw in words:
                overlap = len(words & {kw})
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_cluster = kw
        clusters[best_cluster].append(idx)

    return {
        "n_clusters": len(clusters),
        "top_keywords": dict(keywords.most_common(30)),
        "cluster_sizes": {k: len(v) for k, v in clusters.items()},
        "cluster_examples": {
            k: [sessions[i].get("session_id", "") for i in v[:3]]
            for k, v in clusters.items()
        },
    }


def profile_sessions(sessions: list[dict]) -> dict:
    """Generate comprehensive profile of sessions."""
    stats = _session_length_stats(sessions)
    file_stats = _file_type_stats(sessions)
    topics = _topic_clusters(sessions)

    return {
        "length_stats": stats,
        "file_stats": file_stats,
        "topics": topics,
    }


def main(cfg: Config, label: str | None = None, out_dir: str | None = None) -> int:
    """Run dataset profiling."""
    from src.clean import _dedup_by_session

    cleaned_dir = cfg.path("cleaned_dir")

    # Load sessions
    sessions = []
    if label:
        src = os.path.join(cleaned_dir, label)
        if os.path.isdir(src):
            for fn in sorted(os.listdir(src)):
                if fn.endswith(".json"):
                    try:
                        sessions.append(json.loads(Path(os.path.join(src, fn)).read_text()))
                    except Exception:
                        continue
    else:
        for path in sorted(Path(cleaned_dir).rglob("*.json")):
            try:
                sessions.append(json.loads(path.read_text()))
            except Exception:
                continue

    if not sessions:
        print("[profile] no sessions found")
        return 0

    # Deduplicate by session_id
    sessions = list(_dedup_by_session(sessions).values())
    print(f"[profile] profiling {len(sessions)} sessions")

    # Run profiling
    profile = profile_sessions(sessions)

    # Write output
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(str(cleaned_dir)), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "dataset-profile.json")
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    # Print summary
    stats = profile["length_stats"]
    print(f"[profile] {stats['n_sessions']} sessions, {stats['total_tokens']} tokens")
    print(f"[profile] avg tokens: {stats['avg_tokens']}, median: {stats['median_tokens']}")
    print(f"[profile] token range: {stats['min_tokens']} - {stats['max_tokens']}")
    print(f"[profile] avg messages: {stats['avg_messages']}, avg tools: {stats['avg_tools']}")

    langs = profile["file_stats"]["languages"]
    if langs:
        print(f"[profile] languages: {dict(list(langs.items())[:5])}")

    topics = profile["topics"]
    print(f"[profile] {topics['n_clusters']} topic clusters")

    print(f"[profile] profile written to {out_path}")
    return 0
