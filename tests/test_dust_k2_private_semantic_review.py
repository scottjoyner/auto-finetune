"""Synthetic-only HMAC review queue and deny-only human adjudication tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.dust.k2_private_semantic_review_queue import make_review_queue
from experiments.dust.k2_human_semantic_adjudication import adjudicate_private_queue


def synthetic_queue():
    report = make_review_queue(
        pairs=[(0, 1, .91), (0, 2, .86), (1, 2, .89)],
        ids=["1" * 64, "2" * 64, "3" * 64],
        groups=["a" * 64, "b" * 64, "b" * 64],
        splits=["train", "test", "test"],
        source_sha="c"*64, cluster_sha="d"*64,
        semantic_sha="e"*64, model_sha="f"*64)
    report["prior_aggregate_reconciled"]=True
    return report


def write_private(path: Path, data):
    path.write_text(json.dumps(data, sort_keys=True))
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(queue, sha, reviewer, decisions):
    return {
        "schema": "auto-finetune.dust-k2-independent-human-review.v1",
        "queue_sha256": sha,
        "reviewer_id": reviewer,
        "reviewer_attests_human_review": True,
        "model_generated_labels": False,
        "decisions": [
            {"pair_id_sha256": item["pair_id_sha256"], "decision": decisions[i]}
            for i, item in enumerate(queue["review_candidates"])
        ],
    }


class SemanticHumanReviewTests(unittest.TestCase):
    def test_split_crossing_queue_drops_same_lexical_family(self):
        queue=synthetic_queue()
        self.assertEqual(queue["all_candidate_cross_family_edges"],2)
        self.assertEqual(queue["cross_partition_review_candidates"],2)
        self.assertEqual(len(queue["review_candidates"]),2)
        self.assertTrue(all(x["human_label"]=="UNREVIEWED"
                            for x in queue["review_candidates"]))
        self.assertTrue(all(
            x["left_candidate_partition"] != x["right_candidate_partition"]
            for x in queue["review_candidates"]))
        self.assertNotIn("prompt text",str(queue))
        self.assertFalse(queue["classifier_training_authorized"])

    def test_two_independent_reviews_aggregate_disagreement_never_promotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            queue=synthetic_queue()
            q=root/"queue.json"
            qsha=write_private(q,queue)
            ra=receipt(queue,qsha,"reviewer-A",
                       ["SAME_INTENT","DIFFERENT_INTENT"])
            rb=receipt(queue,qsha,"reviewer-B",
                       ["SAME_INTENT","UNCERTAIN"])
            a,b=root/"a.json",root/"b.json"
            asha=write_private(a,ra)
            bsha=write_private(b,rb)
            outcome=adjudicate_private_queue(q,qsha,a,asha,b,bsha)
            self.assertEqual(outcome["pairs_reviewed_by_both"],2)
            self.assertEqual(outcome["agreed_same_intent_candidates"],1)
            self.assertEqual(outcome["unresolved_or_uncertain_candidates"],1)
            self.assertFalse(outcome["automatic_source_reassignment"])
            self.assertFalse(outcome["classifier_training_authorized"])
            self.assertNotIn("reviewer-A",str(outcome))
            self.assertNotIn("1"*64,str(outcome))

    def test_fake_shared_reviewer_and_model_labels_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            queue=synthetic_queue()
            q=root/"queue.json"
            sha=write_private(q,queue)
            a,b=root/"a.json",root/"b.json"
            r=receipt(queue,sha,"same-reviewer",["SAME_INTENT"]*2)
            ah=write_private(a,r)
            bh=write_private(b,r)
            with self.assertRaisesRegex(ValueError,"same human"):
                adjudicate_private_queue(q,sha,a,ah,b,bh)
            r["reviewer_id"]="other-person"
            r["model_generated_labels"]=True
            bh=write_private(b,r)
            with self.assertRaisesRegex(ValueError,"not independently attested"):
                adjudicate_private_queue(q,sha,a,ah,b,bh)

    def test_forged_queue_and_missing_decision_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            queue=synthetic_queue()
            q=root/"queue.json"
            sha=write_private(q,queue)
            a,b=root/"a.json",root/"b.json"
            r1=receipt(queue,sha,"first",["SAME_INTENT"]*2)
            r2=receipt(queue,sha,"second",["DIFFERENT_INTENT"]*2)
            r2["decisions"].pop()
            ah=write_private(a,r1); bh=write_private(b,r2)
            with self.assertRaisesRegex(ValueError,"complete pair coverage"):
                adjudicate_private_queue(q,sha,a,ah,b,bh)
            queue["classifier_training_authorized"]=True
            write_private(q,queue)
            with self.assertRaisesRegex(ValueError,"SHA256 changed"):
                adjudicate_private_queue(q,sha,a,ah,b,bh)

    def test_invalid_or_replayed_cosine_edges_denied(self):
        kwargs=dict(
            ids=["1"*64,"2"*64,"3"*64],
            groups=["a"*64,"b"*64,"b"*64],
            splits=["train","test","test"],
            source_sha="c"*64,cluster_sha="d"*64,
            semantic_sha="e"*64,model_sha="f"*64)
        for pairs in ([ (1,0,.9) ],
                      [ (0,1,float("nan")) ],
                      [ (0,1,.90),(0,1,.91) ],
                      [ (0,1,.83) ]):
            with self.subTest(pairs=pairs):
                with self.assertRaises(ValueError):
                    make_review_queue(pairs=pairs,**kwargs)


if __name__=="__main__":
    unittest.main()
