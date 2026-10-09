"""Bounded group-balanced collector selection: no GPU/model required."""
from __future__ import annotations
import unittest
from experiments.dust.k2_cohort_collect import select_source_indices
from experiments.dust.k2_cohort_preflight import SCHEMA


def cohort():
    rows=[]
    for i,(split,cluster) in enumerate(
        [("train","a"),("train","a"),("train","b"),
         ("train","c"),("validation","d"),
         ("validation","e"),("test","f"),
         ("test","g"),("test","h")]
    ):
        rows.append({
            "sample_index": i,
            "group_split": split,
            "disposition": "ELIGIBLE",
            "near_duplicate_cluster_sha256": cluster * 64,
            "episode_hmac_sha256": chr(120 - i) * 64,
        })
    return {"schema": SCHEMA, "max_tokens": 128, "candidates":rows}


class CollectorTests(unittest.TestCase):
    def test_select_unique_clusters_across_stratified_splits(self):
        rows=select_source_indices(
            cohort(), {"train":2,"validation":2,"test":2})
        self.assertEqual([r["sample_index"] for r in rows],[0,2,4,5,6,7])
        self.assertEqual(len({r["near_duplicate_cluster_sha256"]
                              for r in rows}),6)

    def test_quarantine_and_ineligible_are_not_selected(self):
        data=cohort()
        data["candidates"][0]["disposition"] = "QUARANTINE"
        data["candidates"][4]["disposition"] = "QUARANTINE"
        with self.assertRaisesRegex(ValueError,"not enough unique eligible validation"):
            select_source_indices(
                data,{"train":2,"validation":2,"test":1})

    def test_rejects_invalid_quotas_or_unpinned_manifest(self):
        for bad in (
            {"train":5,"validation":1,"test":1},
            {"train":2,"validation":3,"test":4},
            {"train":-1,"validation":0,"test":2},
        ):
            with self.assertRaises(ValueError):
                select_source_indices(cohort(),bad)
        with self.assertRaisesRegex(ValueError,"schema"):
            select_source_indices(
                {"schema":"untrusted","max_tokens":128,"candidates":[]},
                {"train":1,"validation":0,"test":0})

    def test_missing_cluster_fails_closed(self):
        data=cohort()
        data["candidates"][0].pop("near_duplicate_cluster_sha256")
        with self.assertRaisesRegex(ValueError,"cluster"):
            select_source_indices(
                data,{"train":1,"validation":0,"test":0})


if __name__ == "__main__":
    unittest.main()
