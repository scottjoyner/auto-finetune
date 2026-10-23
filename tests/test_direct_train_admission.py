"""No model/dataset I/O: direct trainer-main fail-closed admission regression."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import train
from src import training_admission


def forbidden(*args, **kwargs):
    raise AssertionError("training attempted dataset/model work before admission")


@pytest.mark.parametrize("kwargs", [
    {},
    {"source": "hermes"},
    {"source": "mixed", "label": "qwen3-8b-sft-r1"},
    {"source": "mixed", "max_examples": 1},
    {"tokenized_dir": "/does-not-exist"},
])
def test_direct_train_main_hold_precedes_dataset_and_gpu(kwargs, monkeypatch, capsys):
    monkeypatch.setattr(training_admission, "read_live",
                        lambda: SimpleNamespace(allowed=False, reason="stop_file"))
    monkeypatch.setattr(train, "_reject_held_out", forbidden)
    monkeypatch.setattr(train, "validate_dataset", forbidden)
    monkeypatch.setattr(train, "_resolve_backend", forbidden)
    assert train.main(SimpleNamespace(path=lambda name: "/disposable-tests"), **kwargs) == 78
    assert "HOLD: stop_file" in capsys.readouterr().out


def test_direct_train_main_fails_closed_if_admission_crashes(monkeypatch, capsys):
    monkeypatch.setattr(training_admission, "read_live", forbidden)
    monkeypatch.setattr(train, "_reject_held_out", forbidden)
    monkeypatch.setattr(train, "validate_dataset", forbidden)
    assert train.main(SimpleNamespace(path=lambda name: "/disposable-tests")) == 78
    assert "HOLD: admission_unavailable" in capsys.readouterr().out


def test_admitted_path_does_not_skip_holdout_verification(monkeypatch):
    monkeypatch.setattr(training_admission, "read_live",
                        lambda: SimpleNamespace(allowed=True, reason="admitted"))
    marker = RuntimeError("holdout-validation-reached")
    def raise_marker(*args, **kwargs):
        raise marker
    monkeypatch.setattr(train, "_reject_held_out", raise_marker)
    with pytest.raises(RuntimeError, match="holdout-validation-reached"):
        train.main(SimpleNamespace(path=lambda name: "/disposable-tests"))


def test_dry_run_does_not_check_active_gpu_training_admission(monkeypatch):
    monkeypatch.setattr(training_admission, "read_live", forbidden)
    marker = RuntimeError("cpu-only-dataset-check-reached")
    def raise_marker(*args, **kwargs):
        raise marker
    monkeypatch.setattr(train, "_reject_held_out", raise_marker)
    with pytest.raises(RuntimeError, match="cpu-only-dataset-check-reached"):
        train.main(SimpleNamespace(path=lambda name: "/disposable-tests"), dry_run=True)
