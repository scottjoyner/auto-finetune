"""Tests for cross-language Rust exact categorical search custody."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.lfm3_category_bridge import (
    DOMAIN, UntrustedRetrievalRefused, read_ranked_packet,
    render_category_context, validate_ranked,
)
from src.lfm3_retrieval_bridge import validate_untrusted_packet


def fixture():
    root = Path(__file__).resolve().parents[1] / "research" / "fixtures"
    source = root / "lfm3_category_source_packet_20261027.json"
    ranked = root / "lfm3_category_ranked_packet_20261027.json"
    return source, ranked


class CategoryBridgeTests(unittest.TestCase):
    def test_actual_rust_categorical_index_fixture_cross_language(self):
        src, ranked = fixture()
        packet, base = read_ranked_packet(ranked, src)
        self.assertEqual(packet["candidate_window"], 16)
        self.assertEqual(packet["hits"][0]["source_seq"], 14)
        self.assertEqual(packet["hits"][0]["status"], "error")
        self.assertFalse(packet["semantic_retrieval_verified"])
        self.assertFalse(packet["may_generate_gate_plan"])
        context = render_category_context(packet, base, synthetic_lab_only=True)
        self.assertIn("status=error", context)
        self.assertNotIn(packet["session_hash"], context)
        self.assertNotIn(packet["hits"][0]["event_hash"], context)
        with self.assertRaises(UntrustedRetrievalRefused):
            render_category_context(packet, base, synthetic_lab_only=False)

    def test_tampering_source_session_status_and_permissions_rejected(self):
        src, ranked = fixture()
        packet, base = read_ranked_packet(ranked, src)
        for field, value in (
            ("query_status", "ok"),
            ("source_packet_sha256", "0"*64),
            ("session_hash", "1"*64),
            ("may_train_controller", True),
            ("semantic_retrieval_verified", True),
            ("candidate_window", 5),
        ):
            modified = copy.deepcopy(packet)
            modified[field] = value
            with self.subTest(field=field), self.assertRaises(UntrustedRetrievalRefused):
                validate_ranked(modified, base)
        corrupted = copy.deepcopy(packet)
        corrupted["hits"][0]["status"] = "ok"
        with self.assertRaises(UntrustedRetrievalRefused):
            validate_ranked(corrupted, base)

    def test_replay_from_other_session_is_rejected(self):
        src, ranked = fixture()
        packet, base = read_ranked_packet(ranked, src)
        changed = copy.deepcopy(base)
        changed["session_hash"] = "f"*64
        with self.assertRaises(UntrustedRetrievalRefused):
            validate_ranked(packet, changed)

    def test_unknown_packet_fields_and_oversize_refused(self):
        src, ranked = fixture()
        packet, base = read_ranked_packet(ranked, src)
        invalid = copy.deepcopy(packet)
        invalid["unknown_untrusted_elevation"] = "yes"
        with self.assertRaises(UntrustedRetrievalRefused):
            validate_ranked(invalid, base)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ranked.json"
            path.write_bytes(b" " * 16385)
            with self.assertRaises(UntrustedRetrievalRefused):
                read_ranked_packet(path, src)


if __name__ == "__main__":
    unittest.main()
