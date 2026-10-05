"""Tests for the scheduler's candidate pipeline.

The candidate pipeline is the only path that publishes a dataset, adapter or
merged model to the live locations, and it is fail-closed by design: every step
raises rather than promoting a partial result. These tests pin that behaviour
without touching a GPU, the network, or the real staging tree.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from conftest import make_cfg

from src.harvest import HarvestPlan, SourceStats
from src.scheduler import Scheduler


def _source(name: str, total: int = 10, label: str = "") -> SourceStats:
    return SourceStats(
        name=name,
        db_path=f"/tmp/{name}.db",
        total_sessions=total,
        last_modified=0.0,
        db_size_bytes=1,
        new_sessions=total,
        last_harvest=0.0,
        days_since_harvest=0.0,
        source_id=f"{name}:/tmp/{name}.db",
        dataset_label=label or name,
    )


def _plan(**over) -> HarvestPlan:
    src = over.pop("source", _source("opencode", label="ssd"))
    base = dict(
        should_harvest=True,
        should_train=True,
        sources=[src],
        total_new=10,
        estimated_train_hours=1.0,
        batch_labels=[src.name],
        reason="test",
        plan_id="plan-abc",
        harvest_labels=[src.name],
        dataset_labels=[src.dataset_label],
    )
    base.update(over)
    return HarvestPlan(**base)


@pytest.fixture
def sched(tmp_path):
    cfg = make_cfg(paths={"analysis_dir": str(tmp_path / "analysis")})
    (tmp_path / "analysis").mkdir(parents=True, exist_ok=True)
    return Scheduler(cfg)


# --- candidate identity -----------------------------------------------------

def test_candidate_id_is_deterministic(sched):
    assert sched._candidate_id(_plan()) == sched._candidate_id(_plan())


def test_candidate_id_tracks_source_totals(sched):
    a = sched._candidate_id(_plan())
    b = sched._candidate_id(_plan(source=_source("opencode", total=999, label="ssd")))
    assert a != b, "a candidate must not be reused when source data changed"


def test_candidate_id_tracks_eval_partition_config(sched):
    from src.config import Config
    raw = copy.deepcopy(dict(sched.cfg.raw))
    raw.setdefault("paths", {})["analysis_dir"] = sched.cfg.path("analysis_dir")
    raw["scheduler"] = {"eval_seed": 7}
    assert Scheduler(Config(raw=raw))._candidate_id(_plan()) != sched._candidate_id(_plan())


def test_candidate_id_ignores_sources_not_being_harvested(sched):
    # A second source with new sessions must not invalidate a candidate that
    # only covers the first, otherwise unrelated churn restarts the pipeline.
    two = _plan(sources=[_source("opencode", 10, "ssd"),
                         _source("nas5-main", 5000, "nas5-main")])
    assert sched._candidate_id(two) == sched._candidate_id(_plan())


def test_candidate_root_lives_under_analysis_dir(sched):
    root = sched._candidate_root("abc123")
    assert root.parent.name == "candidates"
    assert root.name == "abc123"
    assert str(root).startswith(sched.cfg.path("analysis_dir"))


# --- atomic publish ---------------------------------------------------------

def test_atomic_replace_dir_moves_into_place(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("x")
    dst = tmp_path / "dst"
    Scheduler._atomic_replace_dir(src, dst)
    assert (dst / "f.txt").read_text() == "x"
    assert not src.exists()


def test_atomic_replace_dir_replaces_existing_dir(tmp_path):
    dst = tmp_path / "dst"
    dst.mkdir()
    (dst / "stale.txt").write_text("old")
    src = tmp_path / "src"
    src.mkdir()
    (src / "new.txt").write_text("new")
    Scheduler._atomic_replace_dir(src, dst)
    assert (dst / "new.txt").exists()
    assert not (dst / "stale.txt").exists(), "stale content survived replacement"


def test_atomic_replace_dir_replaces_existing_file(tmp_path):
    dst = tmp_path / "dst"
    dst.write_text("i am a file")
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("x")
    Scheduler._atomic_replace_dir(src, dst)
    assert dst.is_dir()
    assert (dst / "f.txt").exists()


def test_atomic_replace_dir_leaves_no_tmp_behind(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("x")
    dst = tmp_path / "dst"
    Scheduler._atomic_replace_dir(src, dst)
    assert [p.name for p in tmp_path.iterdir()] == ["dst"]


def test_atomic_replace_dir_clears_collided_tmp(tmp_path):
    dst = tmp_path / "dst"
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.txt").write_text("x")
    # Simulate a crashed run that left its staging tmp behind (same pid => same
    # tmp name the implementation will compute).
    stale = tmp_path / f".dst.{os.getpid()}.tmp"
    stale.mkdir()
    (stale / "garbage").write_text("junk")
    Scheduler._atomic_replace_dir(src, dst)
    assert not stale.exists()
    assert (dst / "f.txt").read_text() == "x"


# --- pending candidate resume ----------------------------------------------

def test_pending_candidate_round_trips(sched):
    plan = _plan()
    cid = sched._candidate_id(plan)
    sched._candidate_root(cid).mkdir(parents=True)
    sched._save_pending(cid, plan)
    assert sched._pending_candidate(plan) == sched._candidate_root(cid)


def test_pending_candidate_ignored_for_other_plan(sched):
    plan = _plan()
    cid = sched._candidate_id(plan)
    sched._candidate_root(cid).mkdir(parents=True)
    sched._save_pending(cid, plan)
    assert sched._pending_candidate(_plan(plan_id="different")) is None


def test_pending_candidate_ignored_when_root_vanished(sched):
    plan = _plan()
    cid = sched._candidate_id(plan)
    sched._save_pending(cid, plan)          # root never created
    assert sched._pending_candidate(plan) is None


def test_save_pending_none_clears_state(sched):
    plan = _plan()
    sched._save_pending("abc", plan)
    sched._save_pending(None, plan)
    assert sched.state.pending_candidate is None
    assert sched.state.pending_plan_id is None
    assert sched._pending_candidate(plan) is None


def test_pending_state_survives_reload(sched):
    plan = _plan()
    cid = sched._candidate_id(plan)
    sched._candidate_root(cid).mkdir(parents=True)
    sched._save_pending(cid, plan)
    reloaded = Scheduler(sched.cfg)          # fresh process, same state file
    assert reloaded.state.pending_candidate == cid
    assert reloaded._pending_candidate(plan) == sched._candidate_root(cid)


# --- resuming a finished pipeline ------------------------------------------

def test_run_candidate_promotes_completed_manifest(sched, monkeypatch):
    plan = _plan()
    cid = sched._candidate_id(plan)
    root = sched._candidate_root(cid)
    root.mkdir(parents=True)
    (root / "candidate.json").write_text(json.dumps({
        "schema_version": 1, "candidate_id": cid, "plan_id": plan.plan_id,
        "winner": "ssd", "merged": str(root / "merged"), "benchmark": {"pass_rate": 1.0},
    }))
    promoted = {}
    monkeypatch.setattr(sched, "_promote_candidate",
                        lambda p, m: promoted.update(manifest=m))
    ok, info = sched.run_candidate(plan)
    assert ok and info["promoted"] is True
    assert promoted["manifest"]["winner"] == "ssd"


def test_run_candidate_refuses_manifest_from_another_plan(sched):
    plan = _plan()
    cid = sched.state.pending_candidate or sched._candidate_id(plan)
    root = sched._candidate_root(cid)
    root.mkdir(parents=True)
    (root / "candidate.json").write_text(json.dumps({
        "plan_id": "some-other-plan", "winner": "ssd",
    }))
    with pytest.raises(RuntimeError, match="does not match"):
        sched.run_candidate(plan)


# --- fail-closed gates ------------------------------------------------------

def test_partition_rejects_contamination(sched, monkeypatch, tmp_path):
    import src.eval as ev

    class R:
        train_path = tmp_path / "t.jsonl"
        eval_path = tmp_path / "e.jsonl"
        source_sha256 = "abc"
        train_rows, eval_rows = [1], [1]
        contamination = {"status": "contaminated"}

    monkeypatch.setattr(ev, "build_disjoint_partition", lambda *a, **k: R())
    ds = {"ssd": tmp_path / "train.ssd.jsonl"}
    with pytest.raises(RuntimeError, match="contamination"):
        sched._partition_candidate(ds, tmp_path / "root")


def test_audit_rejects_leakage_into_bench(sched, monkeypatch, tmp_path):
    import src.audit as audit
    bench = Path(sched.repo) / "eval" / "tasks" / "auto-verified.jsonl"
    monkeypatch.setattr(audit, "_load_jsonl", lambda p: [{"id": "t"}])
    monkeypatch.setattr(audit, "audit_leakage",
                        lambda train, bench: {"status": "leaked"})
    with pytest.raises(RuntimeError, match="failed contamination audit"):
        sched._audit_candidate({"ssd": tmp_path / "train.ssd.jsonl"})
    assert bench.name == "auto-verified.jsonl"


def test_audit_requires_bench_suite(sched, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(RuntimeError, match="benchmark suite not found"):
        sched._audit_candidate({"ssd": tmp_path / "train.ssd.jsonl"})


def test_format_rejects_empty_output(sched, monkeypatch, tmp_path):
    import src.format_dataset as fd
    monkeypatch.setattr(fd, "main", lambda *a, **k: 0)
    with pytest.raises(RuntimeError, match="produced no rows"):
        sched._format_candidate(_plan(), tmp_path / "root")


def test_train_requires_complete_adapter(sched, monkeypatch, tmp_path):
    partitions = {"ssd": {"train_path": str(tmp_path / "t.jsonl")}}
    monkeypatch.setattr(sched, "_run_cmd", lambda *a, **k: (0, "ok"))
    with pytest.raises(RuntimeError, match="adapter incomplete"):
        sched._train_candidate(_plan(), tmp_path / "root", partitions)


def test_eval_rejects_nan_loss(sched, monkeypatch, tmp_path):
    import src.eval as ev

    class R:
        n_held_out = 10
        loss = float("nan")

    monkeypatch.setattr(ev, "evaluate", lambda *a, **k: R())
    partitions = {"ssd": {"eval_path": str(tmp_path / "e.jsonl")}}
    with pytest.raises(RuntimeError, match="evaluation invalid"):
        sched._eval_candidate(tmp_path / "root", partitions, {"ssd": "adapter"})


def test_merge_rejects_incomplete_output(sched, monkeypatch, tmp_path):
    import src.merge as merge
    monkeypatch.setattr(merge, "merge_adapter",
                        lambda *a, **k: (tmp_path / "merged").mkdir())
    with pytest.raises(RuntimeError, match="merge output is incomplete"):
        sched._merge_candidate(tmp_path / "root", "ssd", "adapter")


def test_pipeline_failure_leaves_candidate_unpromoted(sched, monkeypatch):
    plan = _plan()
    monkeypatch.setattr(sched, "_candidate_pipeline",
                        lambda p, root: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        sched.run_candidate(plan)
    # Pending state must survive so the next run resumes the same candidate.
    assert sched.state.pending_candidate is not None
    assert sched.state.last_error == "boom"


def test_pipeline_picks_lowest_loss_as_winner(sched, monkeypatch, tmp_path):
    monkeypatch.setattr(sched, "_stage_cleaned", lambda *a: None)
    monkeypatch.setattr(sched, "_format_candidate",
                        lambda p, root: {"a": tmp_path / "a.jsonl",
                                         "b": tmp_path / "b.jsonl"})
    monkeypatch.setattr(sched, "_partition_candidate",
                        lambda ds, root: {"a": {"train_path": "t", "eval_path": "e"},
                                          "b": {"train_path": "t", "eval_path": "e"}})
    monkeypatch.setattr(sched, "_audit_candidate", lambda ds: {})
    monkeypatch.setattr(sched, "_train_candidate",
                        lambda p, root, parts: {"a": "adapter-a", "b": "adapter-b"})

    class R:
        def __init__(self, loss):
            self.loss, self.n_held_out = loss, 5

    import src.eval as ev
    monkeypatch.setattr(ev, "evaluate",
                        lambda ad, base, path, **k: R(0.1 if ad == "adapter-a" else 0.9))
    monkeypatch.setattr(sched, "_merge_candidate", lambda root, l, a: str(root / "m"))
    monkeypatch.setattr(sched, "_benchmark_candidate", lambda m: {"pass_rate": 1.0})
    manifest = sched._candidate_pipeline(_plan(), tmp_path / "root")
    assert manifest["winner"] == "a"
    assert manifest["eval_losses"] == {"a": 0.1, "b": 0.9}


# --- dry-run pre-validation ------------------------------------------------

def _stub_candidate_stages(sched, monkeypatch, *, partitions=None, audits=None):
    monkeypatch.setattr(sched, "_stage_cleaned", lambda *a: None)
    monkeypatch.setattr(sched, "_format_candidate",
                        lambda p, root: {"ssd": root / "train.ssd.jsonl"})
    monkeypatch.setattr(
        sched, "_partition_candidate",
        lambda ds, root: partitions if partitions is not None else {
            "ssd": {"train_path": "t", "eval_path": "e", "source_sha256": "abc",
                    "seed": 42, "frac": 0.1, "n_train": 90, "n_eval": 10}})
    monkeypatch.setattr(
        sched, "_audit_candidate",
        lambda ds: audits if audits is not None else {"ssd": {"status": "clean"}})


def test_run_candidate_dry_run_skips_gpu_stages(sched, monkeypatch):
    plan = _plan()
    _stub_candidate_stages(sched, monkeypatch)
    called = []
    for stage in ("_train_candidate", "_eval_candidate", "_merge_candidate",
                  "_benchmark_candidate", "_promote_candidate"):
        monkeypatch.setattr(sched, stage,
                            lambda *a, _s=stage, **k: called.append(_s))
    ok, info = sched.run_candidate(plan, dry_run=True)
    assert ok is True
    assert info["dry_run"] is True
    assert called == [], f"dry-run executed GPU/promotion stages: {called}"
    assert info["skipped_stages"] == ["train", "eval", "merge", "benchmark",
                                      "promote"]


def test_dry_run_reports_partition_and_audit(sched, monkeypatch):
    _stub_candidate_stages(sched, monkeypatch)
    _, info = sched.run_candidate(_plan(), dry_run=True)
    assert info["partitions"]["ssd"]["n_train"] == 90
    assert info["partitions"]["ssd"]["source_sha256"] == "abc"
    assert info["audits"]["ssd"]["status"] == "clean"


def test_dry_run_writes_a_separate_manifest(sched, monkeypatch):
    # It must not write candidate.json: a later real run would then promote a
    # pipeline that never trained anything.
    _stub_candidate_stages(sched, monkeypatch)
    ok, info = sched.run_candidate(_plan(), dry_run=True)
    root = sched._candidate_root(info["candidate_id"])
    assert (root / "candidate-dryrun.json").is_file()
    assert not (root / "candidate.json").exists()


def test_dry_run_does_not_set_pending_state(sched, monkeypatch):
    _stub_candidate_stages(sched, monkeypatch)
    sched.run_candidate(_plan(), dry_run=True)
    assert sched.state.pending_candidate is None


def test_dry_run_still_fails_closed_on_contamination(sched, monkeypatch):
    _stub_candidate_stages(sched, monkeypatch,
                           partitions={"ssd": {"contamination": "bad"}})
    def boom(*a, **k):
        raise RuntimeError("partition contamination for ssd")
    monkeypatch.setattr(sched, "_partition_candidate", boom)
    with pytest.raises(RuntimeError, match="contamination"):
        sched.run_candidate(_plan(), dry_run=True)


def test_run_once_dry_run_prevalidates_candidate(sched, monkeypatch):
    plan = _plan()
    monkeypatch.setattr("src.harvest.plan_harvest", lambda cfg: plan)
    _stub_candidate_stages(sched, monkeypatch)
    result = sched.run_once(dry_run=True)
    assert result.phase == "dry-run"
    assert result.success is True
    assert "pre-validated" in result.message
    assert result.train_stats is None, "dry-run must not train"


def test_run_once_dry_run_reports_prevalidation_failure(sched, monkeypatch):
    plan = _plan()
    monkeypatch.setattr("src.harvest.plan_harvest", lambda cfg: plan)

    def boom(self, *a, **k):
        raise RuntimeError("candidate ssd failed contamination audit")
    monkeypatch.setattr(Scheduler, "_dry_run_candidate", boom)
    result = sched.run_once(dry_run=True)
    assert result.success is False
    assert result.phase == "dry-run"
    assert "contamination" in result.message


def test_run_once_dry_run_skips_validation_when_not_training(sched, monkeypatch):
    plan = _plan(should_train=False)
    monkeypatch.setattr("src.harvest.plan_harvest", lambda cfg: plan)
    result = sched.run_once(dry_run=True)
    assert result.success is True
    assert "pre-validated" not in result.message


# --- run_once wiring: candidate pipeline replaces train + eval_and_select ---

def _wire_run_once(sched, monkeypatch, *, candidate, harvest_ok=True,
                   deploy_ok=True):
    """Stub everything run_once touches except the phase under test."""
    plan = _plan()
    monkeypatch.setattr("src.harvest.plan_harvest", lambda cfg: plan)
    monkeypatch.setattr("src.harvest.record_extraction", lambda *a, **k: None)
    # scheduler.py imports this at module level, so the name to patch is the
    # one bound in src.scheduler, not the attribute on src.locking.
    monkeypatch.setattr("src.scheduler.unmanaged_training_processes",
                        lambda *a, **k: [])
    calls = {"candidate": 0, "deploy": [], "train": 0, "eval_select": 0}

    def fake_harvest(p):
        return harvest_ok, {"sources": p.harvest_labels}

    def fake_run_candidate(p, dry_run=False):
        calls["candidate"] += 1
        if isinstance(candidate, Exception):
            raise candidate
        return candidate

    def fake_deploy(label):
        calls["deploy"].append(label)
        return deploy_ok, {"message": "ok"}

    monkeypatch.setattr(sched, "harvest", fake_harvest)
    monkeypatch.setattr(sched, "run_candidate", fake_run_candidate)
    monkeypatch.setattr(sched, "deploy", fake_deploy)
    monkeypatch.setattr(sched, "train",
                        lambda *a: calls.__setitem__("train", calls["train"] + 1))
    monkeypatch.setattr(sched, "eval_and_select",
                        lambda: calls.__setitem__("eval_select",
                                                  calls["eval_select"] + 1))
    return calls


def test_run_once_uses_the_candidate_pipeline(sched, monkeypatch):
    calls = _wire_run_once(
        sched, monkeypatch,
        candidate=(True, {"winner": "ssd", "eval_losses": {"ssd": 0.5},
                          "benchmark": {"pass_rate": 1.0}}))
    result = sched.run_once()
    assert result.success is True
    assert result.phase == "complete"
    assert calls["candidate"] == 1
    assert calls["deploy"] == ["ssd"], "winner not passed to deploy"


def test_run_once_no_longer_calls_legacy_train_or_eval(sched, monkeypatch):
    calls = _wire_run_once(
        sched, monkeypatch,
        candidate=(True, {"winner": "ssd", "eval_losses": {}, "benchmark": {}}))
    sched.run_once()
    assert calls["train"] == 0, "legacy train() is still being used"
    assert calls["eval_select"] == 0, "legacy eval_and_select() is still being used"


def test_candidate_exception_fails_the_run_without_deploying(sched, monkeypatch):
    calls = _wire_run_once(
        sched, monkeypatch,
        candidate=RuntimeError("candidate ssd failed contamination audit"))
    result = sched.run_once()
    assert result.success is False
    assert result.phase == "candidate"
    assert "contamination" in result.message
    assert calls["deploy"] == [], "deployed despite a failed candidate"


def test_candidate_failure_does_not_deploy(sched, monkeypatch):
    calls = _wire_run_once(sched, monkeypatch, candidate=(False, {"error": "boom"}))
    result = sched.run_once()
    assert result.success is False
    assert calls["deploy"] == []


def test_candidate_without_winner_fails_closed(sched, monkeypatch):
    calls = _wire_run_once(sched, monkeypatch, candidate=(True, {}))
    result = sched.run_once()
    assert result.success is False
    assert result.phase == "candidate"
    assert "no winner" in result.message
    assert calls["deploy"] == []


def test_successful_run_records_state(sched, monkeypatch):
    _wire_run_once(sched, monkeypatch,
                   candidate=(True, {"winner": "ssd", "eval_losses": {},
                                     "benchmark": {}}))
    sched.run_once()
    assert sched.state.runs_completed == 1
    assert sched.state.runs_failed == 0
    assert sched.state.current_phase == "idle"
    assert sched.state.last_error is None


def test_failed_candidate_increments_failures_and_keeps_error(sched, monkeypatch):
    _wire_run_once(sched, monkeypatch,
                   candidate=RuntimeError("partition contamination for ssd"))
    sched.run_once()
    assert sched.state.runs_failed == 1
    assert "contamination" in sched.state.last_error
    assert sched.state.current_phase == "idle"


def test_candidate_stats_are_reported_on_success(sched, monkeypatch):
    _wire_run_once(
        sched, monkeypatch,
        candidate=(True, {"winner": "ssd", "eval_losses": {"ssd": 0.42},
                          "audits": {"ssd": {"status": "clean"}},
                          "benchmark": {"pass_rate": 0.8}}))
    result = sched.run_once()
    assert result.train_stats["candidate"]["winner"] == "ssd"
    # The candidate detail (losses, audits, benchmark) rides on candidate_stats;
    # deploy_stats belongs to the deploy phase.
    assert result.candidate_stats["eval_losses"] == {"ssd": 0.42}
    assert result.candidate_stats["benchmark"]["pass_rate"] == 0.8
    assert result.candidate_stats["audits"]["ssd"]["status"] == "clean"


def test_harvest_failure_short_circuits_before_the_candidate(sched, monkeypatch):
    calls = _wire_run_once(sched, monkeypatch,
                           candidate=(True, {"winner": "ssd"}),
                           harvest_ok=False)
    result = sched.run_once()
    assert result.success is False
    assert result.phase == "harvest"
    assert calls["candidate"] == 0


def test_promote_lands_where_deploy_looks(tmp_path, monkeypatch):
    """Regression: promotion used os.path.dirname(output_dir), putting the
    merged model one level above the directory deploy.py and quantize.py read.
    """
    ckpt = tmp_path / "outputs" / "checkpoints"
    cfg = make_cfg(train={"output_dir": str(ckpt)},
                   paths={"analysis_dir": str(tmp_path / "analysis")})
    sched = Scheduler(cfg)
    (tmp_path / "analysis").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("src.harvest.record_harvest", lambda *a, **k: None)

    merged = tmp_path / "cand" / "merged"
    merged.mkdir(parents=True)
    (merged / "config.json").write_text("{}")

    plan = _plan()
    sched._promote_candidate(plan, {"winner": "ssd", "merged": str(merged)})

    expected = ckpt / "toolcall-v5-3b-ssd-merged"
    assert expected.is_dir(), f"merged model not where deploy.py looks ({expected})"
    assert (expected / "config.json").is_file()
    assert not (tmp_path / "outputs" / "toolcall-v5-3b-ssd-merged").exists()
