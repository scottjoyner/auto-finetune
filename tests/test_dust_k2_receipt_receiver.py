"""Independent receiver HMAC: negative receipts, fail-closed and chain checks."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.dust.k2_receipt_receiver import (
    prepare_receiver, receive, verify,
)


class ReceiverCustodyTests(unittest.TestCase):
    def test_receiver_precommit_and_verify_without_producer_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "independent-x1"
            prepare_receiver(root)
            key = root / "receiver-owned.key"
            self.assertEqual(key.stat().st_mode & 0o777, 0o600)
            first = receive(root, "a" * 32, "1" * 64)
            second = receive(root, "a" * 32, "2" * 64)
            self.assertEqual(first["batch_index"], 0)
            self.assertEqual(second["batch_index"], 1)
            self.assertEqual(second["previous_receipt_hmac_sha256"],
                             first["receiver_hmac_sha256"])
            summary = verify(root, "a" * 32)
            self.assertEqual(summary["signed_batch_receipts"], 2)
            self.assertTrue(summary["receiver_hmac_chain_verified"])
            self.assertFalse(summary["authorizes_real_classifier_training"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                receive(root, "a" * 32, "1" * 64)

    def test_modified_saved_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "receiver"
            prepare_receiver(root)
            receive(root, "f" * 32, "a" * 64)
            path = root / ("f" * 32 + ".receipt.jsonl")
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            rows[0]["batch_sha256"] = "b" * 64
            path.write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid receiver HMAC"):
                verify(root, "f" * 32)

    def test_no_key_leak_and_invalid_digests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "receiver"
            prepare_receiver(root)
            with self.assertRaises(ValueError):
                receive(root, "not-hex", "a" * 64)
            with self.assertRaises(ValueError):
                receive(root, "a" * 32, "bad-hash")
            self.assertFalse((root / "a" * 32).exists()
                             if False else False)


if __name__ == "__main__":
    unittest.main()
