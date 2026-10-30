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


def test_training_imports_not_required_before_denied_admission():
    """A model-library-less CPU host must still be able to refuse a train run.

    New subprocess isolates sys.modules and deliberately rejects importing
    transformers/torch; no model execution, GPU or dataset reads are possible.
    """
    import subprocess
    import sys
    code = '''
import builtins
original_import = builtins.__import__
def deny_model_stack(name, *args, **kwargs):
    if name.split(".")[0] in ("transformers", "torch"):
        raise ImportError("model library deliberately unavailable in admission test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = deny_model_stack
from src import train, training_admission
training_admission.read_live=lambda: training_admission.Decision(False,"stop_file")
assert train.main(None) == 78
print("DENIED_WITHOUT_MODEL_IMPORTS")
'''
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=20,
                            check=False)
    assert result.returncode == 0, result.stderr
    assert "DENIED_WITHOUT_MODEL_IMPORTS" in result.stdout
    assert "HOLD: stop_file" in result.stdout


def test_lfm3_model_loader_is_after_admission(monkeypatch):
    """Denial should happen before runtime model-library imports or patches."""
    monkeypatch.setattr(training_admission,"read_live",
                        lambda: SimpleNamespace(allowed=False,reason="stop_file"))
    monkeypatch.setattr(train,"_prepare_lfm3_model_imports", forbidden)
    monkeypatch.setattr(train,"patch_transformers_for_lfm3", forbidden)
    assert train.main(None) == 78
