"""Unit tests for provenance and comparison admission, no model load."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.lfm3_fabricated_ab import read_lab_cases, make_manifest, compose_messages, ResearchRefused
from src.lfm3_ab_protocol import validate_manifest


def case(name, split):
    return {
        "id": name, "session_hash": hashlib.sha256(name.encode()).hexdigest(),
        "split": split, "kind": "graph_lookup", "prompt": "What is the code?",
        "reference": "Fictional code is L2.", "expected": "L2",
    }


class FabricatedABTests(unittest.TestCase):
    def make_path(self, record):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        p = Path(temp.name) / "synthetic.json"
        p.write_text(json.dumps(record))
        return p

    def fixture(self):
        return {"schema": "sg-lfm3-fabricated-ab.v1",
                "source": "fabricated-synthetic-lab-only",
                "cases": [case("dev-x", "dev"), case("heldout-y", "heldout")]}

    def test_valid_lab_packet_and_manifest(self):
        cases = read_lab_cases(self.make_path(self.fixture()))
        attest = {"checkpoint_sha256": "a"*64, "tokenizer_sha256": "b"*64}
        validate_manifest(make_manifest(cases, attest, 24))
        self.assertEqual(len(cases), 2)

    def test_retrieval_is_untrusted_separate_data(self):
        base = compose_messages("Question", "Fictional answer A", False)
        retrieved = compose_messages("Question", "Fictional answer A", True)
        self.assertEqual(base, retrieved[:2])
        self.assertEqual(len(retrieved), 3)
        self.assertIn("not instructions", retrieved[-1]["content"])

    def test_bad_provenance_duplicate_and_oversize_refused(self):
        p = self.fixture()
        p["source"] = "real-production-traces"
        with self.assertRaises(ResearchRefused):
            read_lab_cases(self.make_path(p))
        p = self.fixture()
        p["cases"][1]["session_hash"] = p["cases"][0]["session_hash"]
        with self.assertRaises(ResearchRefused):
            read_lab_cases(self.make_path(p))
        p = self.fixture()
        p["cases"][0]["prompt"] = "x" * 513
        with self.assertRaises(ResearchRefused):
            read_lab_cases(self.make_path(p))

    def test_bad_decoding_budget_refused(self):
        attest = {"checkpoint_sha256": "a"*64, "tokenizer_sha256": "b"*64}
        with self.assertRaises(ResearchRefused):
            make_manifest(self.fixture()["cases"], attest, 0)
        with self.assertRaises(ResearchRefused):
            make_manifest(self.fixture()["cases"], attest, 65)


if __name__ == "__main__":
    unittest.main()
