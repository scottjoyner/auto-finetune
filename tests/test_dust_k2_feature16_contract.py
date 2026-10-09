"""Feature-v1 aggregation tests using a complete synthetic causal ledger."""
import json
from pathlib import Path
import stat
import tempfile
import unittest

from test_dust_k2_causal_evidence_verify import fixture
from experiments.dust.k2_feature16_contract import (
    FEATURE_NAMES, SCHEMA, export_private, extract_verified_episode,
)


class Feature16ContractTests(unittest.TestCase):
    def test_complete_verifiable_episode_yields_private_16d_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            sources, _, _, _ = fixture(temp)
            items, report = extract_verified_episode(*sources)
            self.assertEqual(len(FEATURE_NAMES), 16)
            self.assertEqual(report["direction_rows"], 8)
            self.assertFalse(report["training_data_authorized"])
            self.assertFalse(report["classifier_training_authorized"])
            self.assertEqual({r["schema"] for r in items}, {SCHEMA})
            self.assertEqual(len({r["source_group_hmac_sha256"] for r in items}), 1)
            self.assertEqual([x["direction_index"] for x in items], list(range(8)))
            self.assertEqual([r["beneficial_plus_direction"] for r in items],
                             [1, 0, 1, 0, 1, 0, 1, 0])
            self.assertEqual(
                [row["features16_pre_probe"][8:] for row in items[:4]],
                [[0.0] * 8] * 4,
            )
            self.assertEqual(
                [row["features16_pre_probe"][8:] for row in items[4:]],
                [items[4]["features16_pre_probe"][8:]] * 4)
            self.assertEqual(items[4]["features16_pre_probe"][8], 0.5)
            self.assertTrue(all(len(x["features16_pre_probe"]) == 16
                                for x in items))
            out = Path(temp) / "private-features16.jsonl"
            manifest = export_private(*sources, out)
            self.assertEqual(stat.S_IMODE(out.stat().st_mode), 0o600)
            self.assertEqual(len(out.read_text().splitlines()), 8)
            self.assertEqual(manifest["private_export_mode"], "0600")
            self.assertFalse(manifest["production_authorized"])
            self.assertNotIn("tokens", out.read_text())
            self.assertNotIn("prompt", out.read_text())

    def test_existing_output_fails_closed_and_retains_original(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, *_ = fixture(temp)
            dest = Path(temp) / "private.jsonl"
            dest.write_text("already exists")
            with self.assertRaises(PermissionError):
                export_private(*paths, dest)
            self.assertEqual(dest.read_text(), "already exists")

    def test_modified_label_fails_before_any_data_export(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, _, rows, _ = fixture(temp)
            rows[0]["loss_plus"] = 0.5
            paths[1].write_text(
                "".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaises(ValueError):
                export_private(*paths, Path(temp) / "not-created.jsonl")
            self.assertFalse((Path(temp) / "not-created.jsonl").exists())

    def test_readonly_output_requires_private_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            paths, *_ = fixture(temp)
            public = Path(temp) / "public"
            public.mkdir(mode=0o755)
            with self.assertRaises(PermissionError):
                export_private(*paths, public / "dataset.jsonl")


if __name__ == "__main__":
    unittest.main()
