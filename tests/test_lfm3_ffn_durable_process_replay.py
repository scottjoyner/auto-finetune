"""Scratch-only durable checkpoint path and manifest behavior, without loading model."""
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from src.lfm3_ffn_durable_process_replay import (
    scratch_directory,strict_run_args,_read_regular,_write_exclusive,
    STATE_NAME,MANIFEST_NAME,
)
from src.lfm3_real_checkpoint_smoke import ResearchRefused


class DurableProcessReplayBoundTests(unittest.TestCase):
    def setUp(self):
        d=tempfile.TemporaryDirectory(dir="/tmp",prefix="lfm3-durable-")
        self.addCleanup(d.cleanup)
        self.root=Path(d.name)
    def test_only_private_scratch_with_known_optimizer_mode(self):
        self.assertEqual(scratch_directory(self.root),self.root)
        self.assertTrue(strict_run_args("save","adamw",self.root))
        self.assertTrue(strict_run_args("resume","muon_match_rms_adamw",self.root))
        for mode,optimizer in (("deploy","adamw"),("save","muon_original"),("resume","auto")):
            with self.subTest(mode=mode,optimizer=optimizer),self.assertRaises(ResearchRefused):
                strict_run_args(mode,optimizer,self.root)
        with self.assertRaises(ResearchRefused):
            scratch_directory(Path("/nas/CLAIM"))
    def test_exclusive_nonfollow_write_and_bounded_read(self):
        payload=b"fictional lab state"
        _write_exclusive(self.root,STATE_NAME,payload)
        self.assertEqual(_read_regular(self.root/STATE_NAME,100),payload)
        with self.assertRaises(FileExistsError):
            _write_exclusive(self.root,STATE_NAME,payload)
        with self.assertRaises(ResearchRefused):
            _read_regular(self.root/STATE_NAME,4)
    def test_symlink_and_hardlink_inputs_refused(self):
        _write_exclusive(self.root,MANIFEST_NAME,b'{"fake":true}')
        os.symlink(self.root/MANIFEST_NAME,self.root/STATE_NAME)
        with self.assertRaises(ResearchRefused):
            _read_regular(self.root/STATE_NAME,1000)
        (self.root/STATE_NAME).unlink()
        os.link(self.root/MANIFEST_NAME,self.root/STATE_NAME)
        with self.assertRaises(ResearchRefused):
            _read_regular(self.root/STATE_NAME,1000)


if __name__=="__main__":
    unittest.main()
