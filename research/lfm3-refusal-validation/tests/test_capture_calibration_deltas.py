"""Capture preflight and custody-contract tests; no model files or inference."""
import hashlib
from pathlib import Path
import sys

import pytest
from src import capture_calibration_deltas as capture
from src import calibration_direction_stability as stability


def test_digest_reads_file_without_outputting_bytes(tmp_path):
    p = tmp_path / 'fake-model.bin'
    p.write_bytes(b'purely synthetic file')
    assert capture.sha256(p) == hashlib.sha256(b'purely synthetic file').hexdigest()


def test_custody_checks_exact_eight_source_paired_cases(monkeypatch, tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    expected_paths = {
        model / 'model.safetensors': stability.MODEL_SHA,
        tmp_path / 'pairs.jsonl': stability.PAIRS_SHA,
        tmp_path / 'direction.safetensors': stability.DIRECTION_FILE_SHA,
    }
    monkeypatch.setattr(capture, 'sha256', lambda p: expected_paths.get(Path(p), 'bad'))
    rows = [dict(id=p, benign='synthetic benign', adversarial='synthetic adversarial')
            for p in stability.ALL_IDS]
    monkeypatch.setattr(capture, 'read_jsonl', lambda path: rows)
    result = capture.pinned_inputs(model, tmp_path/'pairs.jsonl', tmp_path/'direction.safetensors')
    assert len(result) == 8 and result[0]['id'] == 'p01' and result[-1]['id'] == 'p08'
    rows[5]['id'] = 'p02'
    with pytest.raises(ValueError, match='contract'):
        capture.pinned_inputs(model, tmp_path/'pairs.jsonl', tmp_path/'direction.safetensors')


def test_mismatched_checkpoint_digest_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, 'sha256', lambda path: '0'*64)
    with pytest.raises(ValueError, match='SHA256'):
        capture.pinned_inputs(tmp_path, tmp_path/'pairs', tmp_path/'directions')


def test_capacity_guard_blocks_before_model_load(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['capture_calibration_deltas', '--private-out', str(tmp_path/'private.json')])
    def block(_):
        raise RuntimeError('storage preflight failed as designed')
    def must_not_load(*args, **kwargs):
        raise AssertionError('heavy inference attempted despite storage gate')
    monkeypatch.setattr(capture, 'require_research_write_capacity', block)
    monkeypatch.setattr(capture, 'load_local_model', must_not_load)
    with pytest.raises(RuntimeError, match='storage preflight'):
        capture.main()
    assert not (tmp_path/'private.json').exists()


def test_vector_capture_never_edits_weights_or_runs_generation(monkeypatch):
    import torch
    ids = stability.ALL_IDS
    rows = [dict(id=k, benign=f'{k}-benign', adversarial=f'{k}-adv') for k in ids]
    def synthetic_layer(_model, name):
        hidden = torch.zeros(stability.WIDTH, dtype=torch.float32)
        if name.endswith('-adv'):
            hidden[0] = 1.0
        return {stability.LAYER: hidden}
    monkeypatch.setattr(capture, 'encode', lambda tokenizer, s: s)
    monkeypatch.setattr(capture, 'hidden_by_layer', synthetic_layer)
    d = torch.zeros(stability.WIDTH)
    d[0] = 1.0
    result = capture.snapshot_vectors(object(), object(), rows, d)
    assert result['source_model_sha256'] == stability.MODEL_SHA
    assert result['pair_deltas']['p01'][0] == 1.0
    assert result['pair_deltas']['p08'][1] == 0.0
    assert stability.audit(result)['heldout_fitted_alignment_mean'] == 1.0
    assert set(result) == stability.REQUIRED_FIELDS
