"""Synthetic fail-closed launch-wrapper checks, with no GPU or model loading."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = (
    "post-reboot-gpu-train.sh",
    "post-reboot-check.sh",
    "post-queue.sh",
    "scripts/sft_cascade_watcher.sh",
    "scripts/serve_lfm25.sh",
    "scripts/lfm25_showdown.sh",
    "scripts/legacy_boot/chain-bigmodels.sh",
    "scripts/legacy_boot/upgrade-transformers-gate.sh",
)
REAL_INTERPRETER = "/media/scott/data/finetune-venv/bin/python"
GATE = " -m src.training_admission"


@pytest.mark.parametrize("wrapper", WRAPPERS)
def test_shell_wrapper_fails_closed_before_any_heavy_operation(wrapper, tmp_path):
    source = (ROOT / wrapper).read_text()
    assert REAL_INTERPRETER + GATE in source
    assert source.count(GATE) >= 1
    # First gate is required ahead of *all* side-effecting script commands;
    # extra checks after waits prevent old marker conditions from bypassing.
    first_guard = source.index(GATE)
    if wrapper.startswith("scripts/legacy_boot/"):
        assert first_guard < source.index("while ! grep")
    elif wrapper == "post-reboot-gpu-train.sh":
        assert first_guard < source.index("mkdir -p")
        assert first_guard < source.index("import torch")
    elif wrapper == "post-queue.sh":
        assert first_guard < source.index("mkdir -p")
        assert source.count(GATE) >= 2
    elif wrapper == "scripts/sft_cascade_watcher.sh":
        assert first_guard < source.index("while ! grep")
        assert source.count(GATE) >= 2
    elif wrapper == "scripts/serve_lfm25.sh":
        assert first_guard < source.index("python -m src.cli merge")
        assert first_guard < source.index("pkill -f")
    elif wrapper == "scripts/lfm25_showdown.sh":
        assert first_guard < source.index("for port in")
    elif wrapper == "post-reboot-check.sh":
        assert first_guard < source.index("mkdir -p")
    assert re.search(r"exit 78", source)

    # Rewrite only the interpreter path in a disposable copy to inject a
    # synthetic denied gate. Never run these wrappers on the CI host as-is.
    shim = tmp_path / "mock-deny-python"
    shim.write_text("#!/bin/sh\nexit 78\n")
    shim.chmod(0o700)
    wrapper_file = tmp_path / "wrapper.sh"
    wrapper_file.write_text(source.replace(REAL_INTERPRETER, str(shim)))
    run = subprocess.run(
        ["bash", str(wrapper_file)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    assert run.returncode == 78, (wrapper, run.stderr[-300:])
    assert "admission HOLD" in run.stderr
    assert not (tmp_path / "logs").exists()


def test_bootchain_has_gate_after_wait_before_side_effects():
    chain = (ROOT / "scripts/legacy_boot/chain-bigmodels.sh").read_text()
    upgrade = (ROOT / "scripts/legacy_boot/upgrade-transformers-gate.sh").read_text()
    assert chain.count(GATE) >= 2
    assert upgrade.count(GATE) >= 2
    assert chain.rfind(GATE) < chain.index("exec setsid ./launch-next.sh")
    assert upgrade.rfind(GATE) < upgrade.index("/media/scott/data/finetune-venv/bin/pip install")


def test_post_queue_and_serve_revalidate_after_wait_and_before_restart():
    queue = (ROOT / "post-queue.sh").read_text()
    serve = (ROOT / "scripts/serve_lfm25.sh").read_text()
    assert queue.rfind(GATE) < queue.index("eval-all --loss-only")
    assert serve.rfind(GATE) < serve.index("pkill -f")
