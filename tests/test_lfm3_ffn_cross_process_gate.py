"""Cross-process state reports must match pinned original FFN development."""
import json
import unittest
from pathlib import Path
from src.lfm3_ffn_cross_process_gate import assess_cross_process
from src.lfm3_real_checkpoint_smoke import ResearchRefused

RESULTS=Path(__file__).resolve().parents[1]/"research"/"results"
def fixture():
    reports=[json.loads((RESULTS/f"lfm3_real_ffn_{k}_cross_process_scratch_restart_xwing_20261028.json").read_text()) for k in ("adamw","muon")]
    dev=json.loads((RESULTS/"lfm3_real_final_ffn_lora_three_seeds_six_trials_xwing_20261028.json").read_text())
    return reports,dev

class CrossProcessGateTests(unittest.TestCase):
    def test_real_lfm3_both_optimizer_processes_have_exact_replay(self):
        gate=assess_cross_process(*fixture())
        self.assertTrue(gate["restarted_processes_exact_both"])
        self.assertFalse(gate["can_auto_adjust_optimizer"])
        self.assertFalse(gate["power_loss_verified"])
        self.assertFalse(gate["production_ready"])
    def test_altered_state_and_wrong_seed_rejected(self):
        packets,dev=fixture()
        packets[0]["resume"]["adapter_optimizer_canonical_sha256"]="a"*64
        with self.assertRaises(ResearchRefused):assess_cross_process(packets,dev)
        packets,dev=fixture()
        packets[1]["save"]["seed"]=20261029
        with self.assertRaises(ResearchRefused):assess_cross_process(packets,dev)
    def test_forged_power_loss_or_model_promotion_refused(self):
        for k,v in (
            ("power_loss_tested",True),
            ("exact_cross_process_state",False),
            ("production_ready",True),
        ):
            reports,dev=fixture()
            reports[1][k]=v
            with self.subTest(key=k),self.assertRaises(ResearchRefused):
                assess_cross_process(reports,dev)
        reports,dev=fixture()
        reports[0]["baseline"]["source_authenticated"]=True
        with self.assertRaises(ResearchRefused):
            assess_cross_process(reports,dev)

if __name__=="__main__":
    unittest.main()
