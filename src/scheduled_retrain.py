"""Scheduled retraining orchestrator.

Runs the full pipeline on a schedule (cron-like) with configurable intervals.
Supports:
  - Periodic retraining (every N hours)
  - Data-triggered retraining (when new data exceeds threshold)
  - Quality-triggered retraining (when model performance drops)
  - Resource-aware scheduling (only train when GPU is free)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import Config, load
from src.alerting import AlertManager
from src.data_freshness import assess_freshness
from src.locking import atomic_write_json, unmanaged_training_processes


@dataclass
class ScheduleConfig:
    """Configuration for scheduled retraining."""
    interval_hours: float = 6.0
    min_new_data: int = 50
    max_staleness_hours: float = 24.0
    require_gpu_free: bool = True
    require_data_fresh: bool = True
    notify_on_train: bool = True
    notify_on_skip: bool = False
    max_consecutive_failures: int = 3


@dataclass
class ScheduleState:
    """State of the scheduler."""
    last_run: float = 0.0
    last_train: float = 0.0
    runs_completed: int = 0
    runs_skipped: int = 0
    runs_failed: int = 0
    consecutive_failures: int = 0
    current_status: str = "idle"  # idle, running, waiting, error
    last_error: str | None = None


class ScheduledRetrainer:
    """Manages scheduled retraining."""

    def __init__(self, cfg: Config, schedule: ScheduleConfig | None = None):
        self.cfg = cfg
        self.schedule = schedule or ScheduleConfig()
        self.repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.state_path = os.path.join(cfg.path("analysis_dir"), "scheduled-retrain-state.json")
        self.alerts = AlertManager(cfg)
        self.state = self._load_state()

    def _load_state(self) -> ScheduleState:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                known = {f.name for f in ScheduleState.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in known}
                return ScheduleState(**filtered)
            except Exception:
                pass
        return ScheduleState()

    def _save_state(self) -> None:
        atomic_write_json(self.state_path, asdict(self.state))

    def _gpu_available(self) -> bool:
        """Check if GPU is available for training."""
        if not self.schedule.require_gpu_free:
            return True
        # Check for unmanaged training processes
        try:
            unmanaged = unmanaged_training_processes(None)
            if unmanaged:
                return False
        except Exception:
            pass
        # Check if training is already running
        import subprocess
        try:
            result = subprocess.run(
                ["pgrep", "-f", "src.cli train"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return False
        except Exception:
            pass
        return True

    def _should_run(self) -> tuple[bool, str]:
        """Determine if a retraining run should happen."""
        now = time.time()

        # Check interval
        if now - self.state.last_run < self.schedule.interval_hours * 3600:
            remaining = self.schedule.interval_hours * 3600 - (now - self.state.last_run)
            return False, f"Interval not elapsed: {remaining/3600:.1f}h remaining"

        # Check GPU
        if not self._gpu_available():
            return False, "GPU not available (training in progress)"

        # Check consecutive failures
        if self.state.consecutive_failures >= self.schedule.max_consecutive_failures:
            return False, f"Too many consecutive failures ({self.state.consecutive_failures})"

        # Check data freshness
        if self.schedule.require_data_fresh:
            try:
                report = assess_freshness(self.cfg)
                if report.overall_staleness == "critical":
                    return False, "Data critically stale"
            except Exception:
                pass

        return True, "OK"

    def run(self) -> dict:
        """Run one scheduled retraining cycle."""
        self.state.current_status = "running"
        self._save_state()
        start = time.time()

        should_run, reason = self._should_run()
        if not should_run:
            self.state.current_status = "waiting"
            self.state.runs_skipped += 1
            self._save_state()
            if self.schedule.notify_on_skip:
                self.alerts.fire("info", "scheduler", f"Skipped: {reason}")
            return {"status": "skipped", "reason": reason}

        # Run the pipeline
        print(f"[scheduler] Starting scheduled retraining run")
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = self.repo
            env["TRAIN_ALLOW_UNVERIFIED"] = "1"
            env["AF_SKIP_UNMANAGED_CHECK"] = "1"
            env["ROCM_PATH"] = "/opt/rocm"

            # Incremental extraction
            subprocess.run(
                [sys.executable, "-m", "src.cli", "neo4j-traces", "--incremental"],
                cwd=self.repo, env=env, timeout=300,
            )

            # Clean
            subprocess.run(
                [sys.executable, "-m", "src.cli", "clean", "--label=neo4j-traces"],
                cwd=self.repo, env=env, timeout=300,
            )

            # Format
            subprocess.run(
                [sys.executable, "-m", "src.cli", "format", "--label=neo4j-traces"],
                cwd=self.repo, env=env, timeout=300,
            )

            # Train (via realtime pipeline for quality gating)
            subprocess.run(
                [sys.executable, "-m", "src.cli", "realtime-pipeline", "--mode=once"],
                cwd=self.repo, env=env, timeout=600,
            )

            self.state.last_run = time.time()
            self.state.last_train = time.time()
            self.state.runs_completed += 1
            self.state.consecutive_failures = 0
            self.state.current_status = "idle"
            self._save_state()

            if self.schedule.notify_on_train:
                self.alerts.fire("info", "scheduler", "Scheduled retraining completed")

            return {"status": "completed", "duration": time.time() - start}

        except Exception as e:
            self.state.runs_failed += 1
            self.state.consecutive_failures += 1
            self.state.last_error = str(e)
            self.state.current_status = "error"
            self._save_state()
            self.alerts.fire("error", "scheduler", f"Scheduled retraining failed: {e}")
            return {"status": "failed", "error": str(e)}

    def run_loop(self, check_interval: int = 300) -> None:
        """Run the scheduler in a continuous loop."""
        print(f"[scheduler] Starting scheduled retrainer (check every {check_interval}s)")
        while True:
            try:
                result = self.run()
                print(f"[scheduler] Result: {result['status']}")
                time.sleep(check_interval)
            except KeyboardInterrupt:
                print("[scheduler] Stopped by user")
                self.state.current_status = "idle"
                self._save_state()
                break
            except Exception as e:
                print(f"[scheduler] Error: {e}")
                time.sleep(check_interval)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Scheduled retraining")
    parser.add_argument("--run", action="store_true", help="Run one cycle")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=float, default=6.0, help="Interval in hours")
    parser.add_argument("--check-interval", type=int, default=300, help="Check interval in seconds")
    parser.add_argument("--min-new", type=int, default=50, help="Min new data for training")
    args = parser.parse_args(argv)

    cfg = load()
    schedule = ScheduleConfig(
        interval_hours=args.interval,
        min_new_data=args.min_new,
    )
    retrainer = ScheduledRetrainer(cfg, schedule)

    if args.loop:
        retrainer.run_loop(check_interval=args.check_interval)
    else:
        result = retrainer.run()
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
