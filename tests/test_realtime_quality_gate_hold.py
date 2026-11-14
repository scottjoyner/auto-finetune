"""Realtime format→QA→verified-training binding, with fake child processes."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from src import realtime_pipeline as realtime
from src import quality_bundle as qb


class Cfg:
    def __init__(self, root, model="LiquidAI/LFM3-1.7B"):
        self.root, self.model = root, model
    def path(self, key):
        assert key in ("dataset_dir", "analysis_dir")
        return str(self.root)
    def get(self, section, key, default=None):
        return self.model if (section, key) == ("train", "model_name") else default


def pipeline(tmp_path, model="LiquidAI/LFM3-1.7B"):
    p = object.__new__(realtime.RealtimePipeline)
    p.cfg = Cfg(tmp_path, model)
    p.state = realtime.PipelineState()
    p._save_state = lambda: None
    p._run_cmd = lambda *a, **kw: (0, "")
    p._count_cleaned_files = lambda: 2
    p._count_dataset_rows = lambda: 2
    return p


def fixture_data(root):
    source = root / qb.DATA_NAME
    rows = [
        {"messages": [
            {"role": "user", "content": "synthetic question one"},
            {"role": "assistant", "content": "synthetic answer one"},
        ]},
        {"messages": [
            {"role": "user", "content": "synthetic question two"},
            {"role": "assistant", "content": "synthetic answer two"},
        ]},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sidecar = source.with_suffix(".provenance.jsonl")
    sidecar.write_text("".join(json.dumps({"row": i, "session_id": f"fake-{i}"})
                               + "\n" for i in range(len(rows))))
    return source, sidecar


def test_cleaning_must_not_read_formatted_dataset(monkeypatch, tmp_path):
    p = pipeline(tmp_path)
    monkeypatch.setattr(realtime, "publish_quality_bundle",
                        lambda *a, **k: pytest.fail("premature QA"))
    assert p._clean() is True
    assert p.state.total_cleaned == 2
    assert p.state.quality_bundle_id is None


@pytest.mark.parametrize("error", [
    ValueError("malformed"), OSError("disk unavailable"),
    ValueError("provenance invalid"),
])
def test_format_quality_error_holds(monkeypatch, tmp_path, error):
    p = pipeline(tmp_path)
    fixture_data(tmp_path)
    monkeypatch.setattr(realtime, "publish_quality_bundle",
                        lambda *a, **k: (_ for _ in ()).throw(error))
    assert p._format() is False
    assert p.state.last_error == "quality_bundle_unavailable"
    assert p.state.quality_bundle_id is None


def test_format_source_missing_is_hold(tmp_path):
    p = pipeline(tmp_path)
    assert p._format() is False
    assert p.state.last_error == "quality_bundle_unavailable"


def test_format_then_qa_creates_verified_bundle(tmp_path):
    p = pipeline(tmp_path)
    fixture_data(tmp_path)
    assert p._format() is True
    bid = p.state.quality_bundle_id
    assert bid.startswith("qb-")
    bundle = qb.bundle_path(tmp_path, bid)
    assert all((bundle / f).exists() for f in (
        qb.DATA_NAME, qb.PROVENANCE_NAME, qb.RECEIPT_NAME))
    assert p._verified_bundle_dir() == str(bundle)
    assert p.state.total_formatted == 2


def test_repeated_format_reuses_identical_bundle(tmp_path):
    p = pipeline(tmp_path)
    fixture_data(tmp_path)
    assert p._format()
    first = p.state.quality_bundle_id
    assert p._format()
    assert p.state.quality_bundle_id == first


def test_source_changes_get_new_bundle_and_old_receipt_holds(tmp_path):
    p = pipeline(tmp_path)
    source, sidecar = fixture_data(tmp_path)
    assert p._format()
    original = p.state.quality_bundle_id
    source.write_text(source.read_text() + json.dumps({
        "messages": [
            {"role": "user", "content": "third question"},
            {"role": "assistant", "content": "third reply"},
        ]}) + "\n")
    sidecar.write_text(sidecar.read_text() +
                       json.dumps({"row": 2, "session_id": "fake-3"}) + "\n")
    with pytest.raises(ValueError):
        p._verified_bundle_dir()
    assert p._format()
    assert p.state.quality_bundle_id != original


@pytest.mark.parametrize("tamper", [qb.DATA_NAME, qb.PROVENANCE_NAME, qb.RECEIPT_NAME])
def test_tampered_bundle_denied_before_trainer(tmp_path, tamper):
    p = pipeline(tmp_path)
    fixture_data(tmp_path)
    assert p._format()
    bundle = qb.bundle_path(tmp_path, p.state.quality_bundle_id)
    (bundle / tamper).write_text("MODIFIED")
    p._run_cmd = lambda *a, **kw: pytest.fail("trainer launched")
    assert p._train() is False
    assert p.state.last_error == "quality_training_binding_unavailable"


def test_unapproved_model_family_never_launches(tmp_path):
    p = pipeline(tmp_path, model="Qwen/Qwen2.5-0.5B")
    fixture_data(tmp_path)
    assert p._format()
    p._run_cmd = lambda *a, **kw: pytest.fail("wrong model launched")
    assert p._train() is False
    assert p.state.last_error == "lfm3_model_identity_unapproved"


def test_train_command_targets_only_vetted_directory(tmp_path):
    p = pipeline(tmp_path)
    fixture_data(tmp_path)
    assert p._format()
    seen = []
    def fake_run(cmd, **kwargs):
        seen.append((cmd, kwargs))
        return 78, "synthetic admission HOLD"
    p._run_cmd = fake_run
    assert p._train() is False
    assert p.state.runs_failed == 1
    cmd, kwargs = seen[0]
    assert "--source=neo4j-traces" in cmd
    assert kwargs["extra_env"]["TRAIN_DATASET_DIR"] == p._verified_bundle_dir()
    assert kwargs["extra_env"]["TRAIN_MODEL_NAME"] == "LiquidAI/LFM3-1.7B"
    assert Path(kwargs["extra_env"]["TRAIN_DATASET_DIR"]).name.startswith("qb-")


def test_freshness_exception_fails_closed(tmp_path, monkeypatch):
    p = pipeline(tmp_path)
    fixture_data(tmp_path)
    assert p._format()
    monkeypatch.setattr(realtime, "assess_freshness",
                        lambda cfg: (_ for _ in ()).throw(OSError("telemetry unavailable")))
    assert p._should_train() is False
    assert p.state.last_error == "freshness_unavailable"


def test_valid_qa_can_reach_policy_without_running_trainer(tmp_path, monkeypatch):
    p = pipeline(tmp_path)
    fixture_data(tmp_path)
    assert p._format()
    p.state.new_since_train = 60
    monkeypatch.setattr(realtime, "assess_freshness",
                        lambda cfg: SimpleNamespace(overall_staleness="fresh",
                                                    recommendations=[]))
    assert p._should_train() is True


def test_failed_formatter_never_publishes_bundle(tmp_path):
    p = pipeline(tmp_path)
    fixture_data(tmp_path)
    p._run_cmd = lambda *a, **kw: (78, "formatter HOLD")
    assert p._format() is False
    assert not list(tmp_path.glob("quality-bundles/qb-*"))


def test_failed_quality_stage_prevents_training(tmp_path):
    p = pipeline(tmp_path)
    p._extract = lambda: True
    p._clean = lambda: True
    p._format = lambda: False
    p._train = lambda: pytest.fail("trainer after QA failure")
    p._health_check = lambda: {"status": "synthetic_unknown"}
    results = p.run_once()
    assert results["extraction"] and results["cleaning"]
    assert results["formatting"] is False
    assert results["training"] is False


def test_synthetic_success_does_not_fabricate_checkpoint_registry(tmp_path):
    p = pipeline(tmp_path)
    fixture_data(tmp_path)
    assert p._format()
    p.state.new_since_train = 10
    p._run_cmd = lambda *a, **kw: (0, "synthetic process exit")
    assert p._train() is True
    assert p.state.runs_completed == 1
    assert p.state.new_since_train == 0
    assert not (tmp_path / "model-registry.json").exists()
