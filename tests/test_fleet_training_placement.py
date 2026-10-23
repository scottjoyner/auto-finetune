"""No SSH, GPU model load, or mutating action in placement-policy tests."""
from __future__ import annotations
from copy import deepcopy
from io import StringIO
import contextlib
import json
import sys
import unittest
from unittest.mock import patch
from scripts.fleet_training_placement import decide, main

def healthy():
    return {
        "host": "x1-370",
        "mem_available_kib": str(42 * 1024 * 1024),
        "swap_free_kib": str(4 * 1024 * 1024),
        "pressure": {"some": 0.0, "full": 0.0},
        "oculink_identity": {"root_port_present": True, "r9700_behind_root": True,
                             "switch_1478": True, "switch_1479": True},
        "rocminfo_gfx1201": True,
        "pci_amd_displays": 2,
        "lease": {"device": "gfx1201", "owner": "predictive-finetuning", "exclusive": True},
        "observer": {"ledger_verified": True, "last_observation_state": "healthy", "age_seconds": 60, "state": "healthy"},
        "data_free_bytes": 500 * 1024 ** 3,
    }

class PlacementSafetyTests(unittest.TestCase):
    def test_missing_remote_deny(self):
        self.assertEqual(decide(None), ("NO_GO", ["remote_evidence_unavailable"]))

    def test_all_evidence_passes_preflight_not_launch(self):
        self.assertEqual(decide(healthy()), ("PREFLIGHT_PASS", []))

    def test_usb4_gpu_would_not_satisfy_original_oculink_hardware_identity(self):
        # Even if gfx1201 happens to reappear through another PCIe root,
        # this x1-specific training placement requires the accepted OCuLink
        # wiring and the historical AMD bridge switch chain.
        for missing in ("root_port_present", "r9700_behind_root",
                        "switch_1478", "switch_1479"):
            with self.subTest(missing=missing):
                evidence = healthy()
                evidence["oculink_identity"][missing] = False
                status, reasons = decide(evidence)
                self.assertEqual(status, "NO_GO")
                self.assertIn("r9700_oculink_path_unverified", reasons)

    def test_hardware_missing(self):
        d = healthy()
        d.update(rocminfo_gfx1201=False, pci_amd_displays=1)
        self.assertIn("r9700_physical_gpu_missing", decide(d)[1])

    def test_critical_observer_blocks_even_with_good_ledger(self):
        d = healthy()
        d["observer"]["last_observation_state"] = "critical"
        self.assertIn("gpu_observer_not_healthy", decide(d)[1])

    def test_stale_observer_must_deny_even_when_health_label_is_green(self):
        d = healthy()
        d["observer"]["age_seconds"] = 3000
        self.assertIn("gpu_observer_stale", decide(d)[1])

    def test_wrong_gpu_owner_or_nonexclusive(self):
        for overrides in ({"owner": "other"}, {"exclusive": False}, {"device": "gfx1150"}):
            d = healthy()
            d["lease"].update(overrides)
            self.assertIn("r9700_lease_not_verified", decide(d)[1])

    def test_unknown_or_bad_evidence_always_blocks(self):
        variants = [
            ("host", "xwing", "host_identity_mismatch"),
            ("mem_available_kib", None, "memory_headroom_unknown"),
            ("mem_available_kib", "1000", "memory_headroom_below_32GiB"),
            ("swap_free_kib", "0", "swap_headroom_below_2GiB"),
            ("swap_free_kib", None, "swap_headroom_unknown"),
            ("pressure", {}, "memory_pressure_unknown"),
            ("pressure", {"some": 9, "full": 0}, "memory_pressure_above_gate"),
            ("pressure", {"some": 0, "full": 2}, "memory_pressure_above_gate"),
            ("data_free_bytes", 5, "local_data_space_below_200GiB"),
            ("data_free_bytes", None, "local_data_space_unknown"),
            ("pci_amd_displays", None, "r9700_physical_gpu_missing"),
            ("oculink_identity", None, "r9700_oculink_path_unverified"),
            ("oculink_identity", {}, "r9700_oculink_path_unverified"),
            ("lease", "malformed", "r9700_lease_not_verified"),
            ("observer", "malformed", "gpu_observer_not_healthy"),
            ("observer", {}, "gpu_observer_stale"),
        ]
        for key, value, expected in variants:
            with self.subTest(key=key, value=value):
                d = healthy()
                d[key] = value
                status, reasons = decide(d)
                self.assertEqual(status, "NO_GO")
                self.assertIn(expected, reasons)

    def test_cli_never_grants_launch_authority_even_on_pass(self):
        stream = StringIO()
        with patch("scripts.fleet_training_placement.collect", return_value=healthy()):
            with patch.object(sys, "argv", ["placement"]):
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(main(), 0)
        result = json.loads(stream.getvalue())
        self.assertFalse(result["may_launch_training"])
        self.assertTrue(result["read_only"])
        self.assertEqual(result["status"], "PREFLIGHT_PASS")

if __name__ == "__main__":
    unittest.main()
