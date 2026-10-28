"""Parity transfer evidence must remain once-only and non-authoritative."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_parity_transfer_gate import assess_parity_transfer
from src.lfm3_real_checkpoint_smoke import ResearchRefused
R=Path(__file__).resolve().parents[1]/"research"/"results"

def data():
    hold=json.loads((R/"lfm3_real_ffn_adamw_lr002_vs_004_parity_heldout_xwing_20261028.json").read_text())
    grid=json.loads((R/"lfm3_real_ffn_optimizer_lr_calibration_15_trials_xwing_20261028.json").read_text())
    dev=json.loads((R/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json").read_text())
    return hold,grid,dev

class ParityTransferEvidenceTests(unittest.TestCase):
    def test_actual_different_fictional_family_adamw_lr004_better(self):
        r=assess_parity_transfer(*data())
        self.assertEqual(r["lr_004_better_each_seed"],[True]*3)
        self.assertTrue(all(g>0.2 for g in r["lr_004_relative_gain_vs_002"]))
        self.assertFalse(r["may_auto_tune"])
        self.assertFalse(r["independent_curated_benchmark"])
        self.assertTrue(r["holdout_permanently_spent_for_lr_selection"])
    def test_holdout_tamper_dev_drift_and_wrong_seed_denied(self):
        for k,v in (("parity_holdout_loss",float("nan")),
                    ("seed",20261031),("dev_loss_reproduced",99.)):
            h,g,d=data()
            h["trials"][1][k]=v
            with self.subTest(k=k),self.assertRaises(ResearchRefused):
                assess_parity_transfer(h,g,d)
    def test_fake_training_or_holdout_authority_denied(self):
        for k in ("may_promote","auto_optimizer_update_authorized",
                  "producer_independently_attested","heldout_used_for_training_or_lr_selection"):
            h,g,d=data();h[k]=True
            with self.subTest(k=k),self.assertRaises(ResearchRefused):
                assess_parity_transfer(h,g,d)
        h,g,d=data();h["fictional_parity_holdout_sha256"]="0"*64
        with self.assertRaises(ResearchRefused):
            assess_parity_transfer(h,g,d)

if __name__=="__main__":unittest.main()
