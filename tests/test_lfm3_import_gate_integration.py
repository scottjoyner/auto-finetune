"""Regression: the LFM3 trainer must remain importable before ML dependencies.

This tests the admission-facing entrypoint only. No GPU, training data, model
weights, training bypass, or production resource state is used.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "src" / "train.py"
CONFTEST = ROOT / "tests" / "conftest.py"


def test_train_import_requires_no_torch_or_transformers():
    # A subprocess with a mandatory import blocker catches both direct and
    # transitively imported heavy dependencies, even when installed locally.
    script = """
import builtins
import sys
original = builtins.__import__
def gated_import(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch.") or name == "transformers" or name.startswith("transformers."):
        raise AssertionError("heavy model library imported before admission")
    return original(name, *args, **kwargs)
builtins.__import__ = gated_import
import src.train
assert "transformers" not in sys.modules
assert "torch" not in sys.modules
print("IMPORT_GATE_PASS")
"""
    env = os.environ.copy()
    env.pop("AF_SKIP_UNMANAGED_CHECK", None)
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=8, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "IMPORT_GATE_PASS"


def test_pytest_conftest_does_not_globally_bypass_admission():
    tree = ast.parse(CONFTEST.read_text())
    # The conftest may use os.environ for other non-admission settings; it must
    # not write to this security-sensitive bypass env var anywhere.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.Assign, ast.AugAssign, ast.AnnAssign)):
            segment = ast.get_source_segment(CONFTEST.read_text(), node) or ""
            assert "AF_SKIP_UNMANAGED_CHECK" not in segment


def test_trainer_only_initializes_model_runtime_after_admission():
    tree = ast.parse(TRAIN.read_text())
    entrypoints = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    assert len(entrypoints) == 1
    main = entrypoints[0]
    source = TRAIN.read_text()
    snippet = ast.get_source_segment(source, main)
    assert snippet is not None
    assert 'from src.training_admission import read_live' in snippet
    assert 'return 78' in snippet
    assert snippet.index('from src.training_admission import read_live') < snippet.index('patch_transformers_for_lfm3()')
    assert 'from src.lfm3_chat_template_guard import ensure_native_lfm3_chat_template' in source
    assert 'tokenizer.chat_template =' not in source


def test_no_module_level_transformers_or_torch_imports():
    tree = ast.parse(TRAIN.read_text())
    for node in tree.body:
        if isinstance(node, ast.Import):
            for name in node.names:
                assert not name.name.startswith(("transformers", "torch"))
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("transformers", "torch"))
