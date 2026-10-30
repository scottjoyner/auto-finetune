from pathlib import Path
import pytest
from src.verify_checkpoint import verify


def test_verify_fails_without_provenance(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify(source=tmp_path, target=tmp_path, vector=tmp_path)


def test_verifier_is_not_deployment_authorization():
    from src.build_abliterated_checkpoint import orthogonalize
    import torch
    updated, _, _ = orthogonalize(torch.eye(2), torch.tensor([1., 0.]), 1)
    assert updated[0].abs().sum() == 0
