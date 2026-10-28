"""Scratch-only native Torch durable checkpoint fault injection.

No real model/network. On platforms without Torch these optional native-state
tests skip; core path/manifest unit tests and actual isolated xwing trials
remain separate.
"""
import importlib.util
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from src.lfm3_ffn_durable_process_replay import (
    SCHEMA,MODEL_SHA,SNAPSHOT_REV,TARGET,RANK,SEED,RESUME_STEP,
    STATE_NAME,MANIFEST_NAME,save_durable,load_durable,
)
from src.lfm3_real_checkpoint_smoke import ResearchRefused

HAS_TORCH=importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH,"native Torch not available for isolated checkpoint roundtrip")
class NativeCheckpointFaultTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        scratch=tempfile.TemporaryDirectory(prefix="lfm3-durable-",dir="/tmp")
        self.addCleanup(scratch.cleanup)
        self.root=Path(scratch.name)
    def create(self):
        torch=self.torch
        return {
            "schema":SCHEMA,"optimizer":"adamw","seed":SEED,"step":RESUME_STEP,
            "model_sha256":MODEL_SHA,"model_revision":SNAPSHOT_REV,
            "target":TARGET,"rank":RANK,
            "A":OrderedDict(weight=torch.zeros((4,7680),dtype=torch.float32)),
            "B":OrderedDict(weight=torch.zeros((2048,4),dtype=torch.float32)),
            "optimizer_state":{"state":{},"param_groups":[]},
        }
    def test_fsynced_checkpoint_roundtrips_only_scratch(self):
        manifest=save_durable(self.root,self.create(),optimizer="adamw")
        state,loaded=load_durable(self.root,"adamw")
        self.assertEqual(manifest,loaded)
        self.assertEqual(state["step"],RESUME_STEP)
        self.assertEqual(tuple(state["A"]["weight"].shape),(4,7680))
        self.assertEqual(tuple(state["B"]["weight"].shape),(2048,4))
        self.assertTrue((self.root/STATE_NAME).is_file())
        self.assertTrue((self.root/MANIFEST_NAME).is_file())
        with self.assertRaises(ResearchRefused):
            save_durable(self.root,self.create(),optimizer="adamw")
    def test_tampered_payload_fails_closed_before_load(self):
        save_durable(self.root,self.create(),optimizer="adamw")
        p=self.root/STATE_NAME
        b=bytearray(p.read_bytes())
        b[len(b)//2]^=1
        p.write_bytes(b)
        with self.assertRaises(ResearchRefused):
            load_durable(self.root,"adamw")
    def test_missing_manifest_and_mismatched_optimizer_refused(self):
        save_durable(self.root,self.create(),optimizer="adamw")
        with self.assertRaises(ResearchRefused):
            load_durable(self.root,"muon_match_rms_adamw")
        (self.root/MANIFEST_NAME).unlink()
        with self.assertRaises(FileNotFoundError):
            load_durable(self.root,"adamw")
    def test_changed_manifest_metadata_fails_closed(self):
        save_durable(self.root,self.create(),optimizer="adamw")
        p=self.root/MANIFEST_NAME
        j=json.loads(p.read_text())
        j["production_ready"]=True
        p.write_text(json.dumps(j))
        with self.assertRaises(ResearchRefused):
            load_durable(self.root,"adamw")

if __name__=="__main__":
    unittest.main()
