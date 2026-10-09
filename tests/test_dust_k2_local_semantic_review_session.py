"""No private corpus; local-review TTY denial and synthetic reviewer receipt."""
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

from experiments.dust.k2_local_semantic_review_session import (
    validate_queue_and_source, record_decisions, run_interactive,
)
from test_dust_k2_private_semantic_review import synthetic_queue


class InteractiveReviewTests(unittest.TestCase):
    def test_noninteractive_never_exposes_prompt_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sys.stdin.isatty", return_value=False):
                with self.assertRaisesRegex(PermissionError, "human TTY"):
                    run_interactive(
                        queue_path=Path(tmp)/"nonexistent-queue.json",
                        queue_sha="0"*64,
                        source_path=Path(tmp)/"nonexistent-source.jsonl",
                        source_sha="a"*64,
                        key_path=Path(tmp)/"nonexistent-key",
                        reviewer_id="test-reviewer",
                        receipt_path=Path(tmp)/"review.json")
            self.assertFalse((Path(tmp)/"review.json").exists())

    def test_local_hmac_lookup_and_review_receipt_excludes_prompt(self):
        report=synthetic_queue()
        lookups={
            "1"*64:"synthetic question one",
            "2"*64:"synthetic paraphrase example two",
            "3"*64:"unrelated synthetic sample",
        }
        num=validate_queue_and_source(
            queue=report,queue_sha="e"*64,source_sha="c"*64,
            prompt_lookup=lookups)
        self.assertEqual(num,2)
        report["_source_sha256_from_file"]="e"*64
        outputs=[]
        choices=iter(["s","u"])
        result=record_decisions(
            report,lookups,reviewer_id="test-human-A",
            read=lambda _:next(choices),write=outputs.append)
        self.assertEqual(
            [x["decision"] for x in result["decisions"]],
            ["SAME_INTENT","UNCERTAIN"])
        self.assertNotIn("synthetic question one",str(result))
        self.assertEqual(result["queue_sha256"],"e"*64)
        self.assertTrue(any("Prompt A:" in s for s in outputs))

    def test_invalid_or_incomplete_review_fails_without_default(self):
        report=synthetic_queue()
        report["_source_sha256_from_file"]="e"*64
        lookups={"1"*64:"prompt 1","2"*64:"prompt 2",
                 "3"*64:"prompt 3"}
        with self.assertRaisesRegex(ValueError,"invalid human decision"):
            record_decisions(report,lookups,reviewer_id="human-A",
                             read=lambda _:"",write=lambda _:None)
        with self.assertRaisesRegex(ValueError,"human handle"):
            record_decisions(report,lookups,reviewer_id="../other-human",
                             read=lambda _:"s",write=lambda _:None)

    def test_private_prompt_mapping_must_cover_all_queue_items(self):
        report=synthetic_queue()
        with self.assertRaisesRegex(ValueError,"not found"):
            validate_queue_and_source(
                queue=report,queue_sha="e"*64,source_sha="c"*64,
                prompt_lookup={"1"*64:"only first source prompt"})


if __name__=="__main__":
    unittest.main()
