"""No real weights required; exercise the fail-closed BF16 byte contract."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import lfm3_checkpoint_attestor as mod


class CheckpointAttestationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / mod.MODEL_CACHE_DIR
        self.snapshot = self.root / "snapshots" / mod.REVISION
        self.snapshot.mkdir(parents=True)
        (self.root / "blobs").mkdir()
        self.items = {"model.safetensors": b"synthetic-weights",
                      "config.json": b"synthetic-config",
                      "tokenizer.json": b"synthetic-tokenizer"}
        expected = {}
        for name, data in self.items.items():
            digest = hashlib.sha256(data).hexdigest()
            (self.root / "blobs" / digest).write_bytes(data)
            (self.snapshot / name).symlink_to(Path("../../blobs") / digest)
            expected[name] = (len(data), digest)
        self.patch = patch.object(mod, "ASSETS", expected)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_fixture_snapshot_receipt_does_not_claim_model_execution(self):
        r = mod.verify_snapshot(self.snapshot)
        self.assertTrue(r["pinned_files_verified_from_local_bytes"])
        self.assertFalse(r["remote_code_executed"])
        self.assertFalse(r["production_ready"])

    def test_mismatched_weights_and_replaced_file_refused(self):
        f = self.root / "blobs" / mod.ASSETS["model.safetensors"][1]
        f.write_bytes(b"synthetic-WeightS")
        with self.assertRaises(mod.CheckpointRefused):
            mod.verify_snapshot(self.snapshot)
        f.write_bytes(self.items["model.safetensors"])
        (self.snapshot / "config.json").unlink()
        (self.snapshot / "config.json").symlink_to("/etc/hosts")
        with self.assertRaises(mod.CheckpointRefused):
            mod.verify_snapshot(self.snapshot)

    def test_wrong_revision_and_missing_asset_refused(self):
        with self.assertRaises(mod.CheckpointRefused):
            mod.verify_snapshot(self.snapshot.parent / "wrong-revision")
        (self.snapshot / "tokenizer.json").unlink()
        with self.assertRaises(FileNotFoundError):
            mod.verify_snapshot(self.snapshot)


if __name__ == "__main__":
    unittest.main()
