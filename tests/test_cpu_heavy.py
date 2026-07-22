"""Tests for CPU-heavy modules: dedup, profile, pretokenize."""
import json
import os
import tempfile

import pytest


# Sample test sessions
SAMPLE_SESSION = {
    "session_id": "test-123",
    "source": "opencode",
    "messages": [
        {
            "role": "user",
            "parts": [{"type": "text", "text": "Write a Python function to calculate fibonacci numbers"}]
        },
        {
            "role": "assistant",
            "parts": [{"type": "tool", "tool": "write", "input": {"filePath": "/tmp/fib.py", "content": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"}},]
        },
    ],
}

SAMPLE_SESSION_2 = {
    "session_id": "test-456",
    "source": "hermes",
    "messages": [
        {
            "role": "user",
            "parts": [{"type": "text", "text": "Write a Python function to calculate fibonacci numbers"}]
        },
        {
            "role": "assistant",
            "parts": [{"type": "tool", "tool": "write", "input": {"filePath": "/tmp/fib.py", "content": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"}},]
        },
    ],
}

SAMPLE_SESSION_3 = {
    "session_id": "test-789",
    "source": "opencode",
    "messages": [
        {
            "role": "user",
            "parts": [{"type": "text", "text": "Explain quantum computing in simple terms"}]
        },
        {
            "role": "assistant",
            "parts": [{"type": "text", "text": "Quantum computing uses qubits that can be 0 and 1 at the same time."}]
        },
    ],
}


class TestMinHash:
    """Tests for MinHash near-duplicate detection."""

    def test_minhash_from_text(self):
        from src.dedup import MinHash

        m1 = MinHash.from_text("hello world")
        m2 = MinHash.from_text("hello world")
        m3 = MinHash.from_text("goodbye world")

        assert m1.jaccard(m2) == 1.0
        assert m1.jaccard(m3) < 0.5

    def test_minhash_similarity(self):
        from src.dedup import MinHash

        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumps over the lazy cat"
        text3 = "A completely different sentence about something else"

        m1 = MinHash.from_text(text1)
        m2 = MinHash.from_text(text2)
        m3 = MinHash.from_text(text3)

        sim_12 = m1.jaccard(m2)
        sim_13 = m1.jaccard(m3)

        assert sim_12 > sim_13

    def test_find_duplicates(self):
        from src.dedup import find_duplicates

        sessions = [SAMPLE_SESSION, SAMPLE_SESSION_2, SAMPLE_SESSION_3]
        dupes = find_duplicates(sessions, threshold=0.5)

        assert len(dupes) >= 1
        session_ids = {d[0] for d in dupes} | {d[1] for d in dupes}
        assert "test-123" in session_ids
        assert "test-456" in session_ids

    def test_dedup_sessions(self):
        from src.dedup import dedup_sessions

        sessions = [SAMPLE_SESSION, SAMPLE_SESSION_2, SAMPLE_SESSION_3]
        kept, removed = dedup_sessions(sessions, threshold=0.5)

        assert len(kept) >= 2
        assert len(removed) >= 1


class TestProfile:
    """Tests for dataset profiling."""

    def test_session_length_stats(self):
        from src.profile import _session_length_stats

        sessions = [SAMPLE_SESSION, SAMPLE_SESSION_2, SAMPLE_SESSION_3]
        stats = _session_length_stats(sessions)

        assert "total_tokens" in stats
        assert "avg_tokens" in stats
        assert stats["n_sessions"] == 3

    def test_language_detect(self):
        from src.profile import _language_detect

        assert _language_detect("def foo(): pass") == "python"
        assert _language_detect("function bar() {}") == "javascript"
        assert _language_detect("fn main() {}") == "rust"

    def test_file_type_stats(self):
        from src.profile import _file_type_stats

        sessions = [SAMPLE_SESSION, SAMPLE_SESSION_2]
        stats = _file_type_stats(sessions)

        assert "extensions" in stats
        assert "languages" in stats

    def test_topic_clusters(self):
        from src.profile import _topic_clusters

        sessions = [SAMPLE_SESSION, SAMPLE_SESSION_2, SAMPLE_SESSION_3]
        clusters = _topic_clusters(sessions, max_clusters=5)

        assert "n_clusters" in clusters
        assert "cluster_sizes" in clusters

    def test_profile_sessions(self):
        from src.profile import profile_sessions

        sessions = [SAMPLE_SESSION, SAMPLE_SESSION_2, SAMPLE_SESSION_3]
        profile = profile_sessions(sessions)

        assert "length_stats" in profile
        assert "file_stats" in profile
        assert "topics" in profile


class TestPretokenize:
    """Tests for pre-tokenization."""

    def test_format_conversation(self):
        from src.pretokenize import _format_conversation

        text = _format_conversation(SAMPLE_SESSION)
        assert "fibonacci" in text.lower()
        assert "<|user|>" in text
        assert "<|assistant|>" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
