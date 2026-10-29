"""Run all registry safety checks on isolated temporary files."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.model_registry import ModelRegistry

class ScratchConfig:
    def __init__(self,p):self.path_value=Path(p)
    def path(self,k):
        if k!="analysis_dir":raise ValueError("unexpected key")
        return str(self.path_value)

class RegistryInvariantTests(unittest.TestCase):
    def setUp(self):
        d=tempfile.TemporaryDirectory(prefix="registry-collision-research-")
        self.addCleanup(d.cleanup)
        self.r=ModelRegistry(ScratchConfig(d.name))
    def test_corrupted_or_aliased_registry_never_resets_history(self):
        path=Path(self.r.registry_path)
        first=self.r.register("baseline","/tmp/base","frozen")
        before=path.read_text()
        for bad in (
            "{not-json",
            "{}",
            json.dumps({"versions":[
                {"version_id":first,"label":"a","checkpoint_path":"/tmp/a",
                 "base_model":"frozen","created_at":1},
                {"version_id":first,"label":"b","checkpoint_path":"/tmp/b",
                 "base_model":"frozen","created_at":2}],"ab_tests":[]}),
        ):
            with self.subTest(bad=bad[:18]):
                path.write_text(bad)
                with self.assertRaises((ValueError, TypeError)):
                    ModelRegistry(self.r.cfg)
                self.assertEqual(path.read_text(),bad)
        path.write_text(before)
        self.assertEqual(ModelRegistry(self.r.cfg).get(first).label,"baseline")

    def test_same_second_and_label_never_overwrite(self):
        with patch("src.model_registry.time.time",return_value=100.0):
            a=self.r.register("synthetic","/tmp/synthetic-1","base")
            b=self.r.register("synthetic","/tmp/synthetic-2","base")
            c=self.r.register("synthetic","/tmp/synthetic-3","base")
        self.assertEqual(len({a,b,c}),3)
        self.assertEqual(len(self.r.list_versions()),3)
        self.assertNotEqual(self.r.get(a).checkpoint_path,self.r.get(b).checkpoint_path)
    def test_overlapping_ab_tests_cannot_overwrite_or_steal_active(self):
        r=self.r
        base=r.register("old","/tmp/old","frozen")
        other=r.register("one","/tmp/one","frozen")
        third=r.register("two","/tmp/two","frozen")
        self.assertTrue(r.promote(base,operator_confirmed=True))
        with patch("src.model_registry.time.time",return_value=100.0):
            tid=r.start_ab_test(base,other)
            self.assertIsNotNone(tid)
            self.assertIsNone(r.start_ab_test(base,third))
            self.assertIsNone(r.start_ab_test(other,third))
        self.assertEqual(r.get(base).status,"active")
        self.assertEqual(r.active_version().version_id,base)
        self.assertEqual(r.get(other).ab_test_group,tid)
        self.assertEqual(len(r.list_ab_tests(status="running")),1)
    def test_invalid_metric_accumulators_do_not_mutate(self):
        r=self.r
        a=r.register("a","/tmp/a","base")
        b=r.register("b","/tmp/b","base")
        t=r.start_ab_test(a,b)
        ab=r.get_ab_test(t)
        for total,count in ((float("inf"),0),(float("nan"),0),(0,True),
                            (0,-1),(0,2**53)):
            ab.metrics_a={"score":total,"score_count":count}
            before=dict(ab.metrics_a)
            with self.subTest(total=total,count=count):
                self.assertFalse(r.record_ab_metric(t,a,"score",0.5))
                self.assertEqual(ab.metrics_a,before)
        ab.metrics_a={"score":2.0,"score_count":1}
        self.assertFalse(r.record_ab_metric(t,a,"score",0.5))
        self.assertEqual(ab.metrics_a,{"score":2.0,"score_count":1})
    def test_invalid_completion_preserves_every_status(self):
        r=self.r
        a=r.register("old","/tmp/old","base")
        b=r.register("new","/tmp/new","base")
        r.promote(a,operator_confirmed=True)
        t=r.start_ab_test(a,b)
        r.record_ab_metric(t,a,"score",0.4)
        r.record_ab_metric(t,b,"score",0.6)
        ab=r.get_ab_test(t)
        ab.metrics_b["score"]=999.
        self.assertIsNone(r.complete_ab_test(t,operator_confirmed=True))
        self.assertIsNone(ab.end_time)
        self.assertIsNone(ab.winner)
        self.assertEqual(ab.status,"running")
        self.assertEqual(r.active_version().version_id,a)
        ab.metrics_b["score"]=0.6
        self.assertEqual(r.complete_ab_test(t,operator_confirmed=True),b)
        self.assertEqual(r.active_version().version_id,b)
        self.assertEqual(r.get(a).status,"retired")
        self.assertIsNone(r.start_ab_test(b,b))

if __name__=="__main__":unittest.main()
