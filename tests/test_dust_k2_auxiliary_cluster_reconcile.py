"""Private cluster manifest integrity; synthetic only, no real prompts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dust.k2_auxiliary_prompt_clusters import build_private_manifest
from experiments.dust.k2_auxiliary_cluster_reconcile import verify_manifest


def private_sample(root):
    rows=[
        {"sample_index":0,"normalized_prompt":"how to collect private model prompt datasets",
         "prompt_token_count":20,"assistant_token_count":8},
        {"sample_index":1,"normalized_prompt":"how to collect private model prompt datasets",
         "prompt_token_count":20,"assistant_token_count":7},
        {"sample_index":2,"normalized_prompt":"why does my graphics card work slowly",
         "prompt_token_count":11,"assistant_token_count":10},
    ]
    report=build_private_manifest(
        rows,b"x"*32,source_sha="a"*64,config_sha="b"*64)
    path=root/"clusters.json"
    path.write_text(json.dumps(report,indent=2))
    path.chmod(0o600)
    return path,report,hashlib.sha256(path.read_bytes()).hexdigest()


class PrivateClusterReconciliationTests(unittest.TestCase):
    def test_source_counts_and_sha_verified_no_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,report,sha=private_sample(Path(tmp))
            result=verify_manifest(
                path,exact_sha=sha,source_sha="a"*64,config_sha="b"*64)
            self.assertEqual(result["result"],"PASS_PRIVATE_LEXICAL_ACCOUNTING_ONLY")
            self.assertEqual(result["tokenizable_paired_rows"],3)
            self.assertEqual(result["unique_exact_prompts"],2)
            self.assertEqual(result["lexical_components"],2)
            self.assertEqual(sum(result["source_candidate_component_counts_by_split"].values()),2)
            self.assertFalse(result["dataset_reuse_rights_approved"])
            self.assertFalse(result["classifier_training_authorized"])
            self.assertNotIn("how to collect",str(result))
            self.assertTrue(all("normalized_prompt" not in row
                for row in report["candidate_entries"]))

    def test_refuse_rehashed_forged_promotions_or_duplicate_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,base,_=private_sample(Path(tmp))
            edits=[
                ("classifier_training_authorized",True),
                ("dataset_reuse_rights_approved",True),
                ("source_partition_authorized",True),
                ("lexical_source_components",100),
            ]
            for field,value in edits:
                altered=json.loads(json.dumps(base))
                altered[field]=value
                path.write_text(json.dumps(altered))
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(field=field):
                    with self.assertRaises(ValueError):
                        verify_manifest(path,exact_sha=sha,source_sha="a"*64,
                                        config_sha="b"*64)
            copied=json.loads(json.dumps(base))
            copied["candidate_entries"][1]["prompt_hmac_sha256"] = (
                copied["candidate_entries"][0]["prompt_hmac_sha256"])
            path.write_text(json.dumps(copied))
            with self.assertRaisesRegex(ValueError,"replayed prompt"):
                verify_manifest(path,exact_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
                                source_sha="a"*64,config_sha="b"*64)

    def test_refuse_mode_sha_drift_and_split_forgery(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,base,sha=private_sample(Path(tmp))
            with self.assertRaisesRegex(ValueError,"SHA256 changed"):
                verify_manifest(path,exact_sha="0"*64,
                                source_sha="a"*64,config_sha="b"*64)
            path.chmod(0o644)
            with self.assertRaises(PermissionError):
                verify_manifest(path,exact_sha=sha,
                                source_sha="a"*64,config_sha="b"*64)
            path.chmod(0o600)
            record=base["candidate_entries"][0]
            record["partition_candidate_only"] = (
                "test" if record["partition_candidate_only"]!="test" else "train")
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError,"family split"):
                verify_manifest(path,exact_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
                                source_sha="a"*64,config_sha="b"*64)

if __name__=="__main__":
    unittest.main()
