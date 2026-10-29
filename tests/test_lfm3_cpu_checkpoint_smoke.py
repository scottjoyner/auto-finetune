"""Command line safety contract: does not load model weights in tests."""
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "research" / "lfm3_cpu_checkpoint_smoke.py"


def test_no_opt_in_does_not_load_checkpoint(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--checkpoint", str(tmp_path)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert '"reason": "full_checkpoint_load_not_authorized"' in result.stdout
    assert "weight_promotion_permitted" in result.stdout
    assert str(tmp_path) not in result.stdout + result.stderr


def test_no_checkpoint_does_not_run():
    result = subprocess.run([sys.executable, str(SCRIPT)],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "required" in result.stderr
    assert "Traceback" not in result.stderr


def test_help_does_not_load_torch():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert "--run" in result.stdout
    assert "--checkpoint" in result.stdout


def test_explicit_run_missing_checkpoint_fails_closed(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--run",
                             "--checkpoint", str(tmp_path / "absent")],
                            capture_output=True, text=True, timeout=15,
                            env={"PATH": "/usr/bin:/bin"})
    assert result.returncode == 2
    assert "checkpoint_directory_missing_or_linked" in result.stdout
    assert str(tmp_path) not in result.stdout
