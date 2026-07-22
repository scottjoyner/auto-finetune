"""Scheduler for orchestrating harvest-train-deploy pipeline.

Manages the full lifecycle:
1. Detect new data (harvest.py)
2. Extract and clean
3. Train on queue
4. Eval and pick best
5. Deploy to inference nodes

Usage:
    python -m src.cli scheduler-status
    python -m src.cli scheduler-run [--dry-run]
    python -m src.cli scheduler-loop [--interval=3600]
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.config import Config


SCHEDULER_STATE_FILE = "scheduler-state.json"


@dataclass
class SchedulerState:
    """State of the scheduler loop."""
    last_run: float
    last_harvest: float
    last_train: float
    last_deploy: float
    runs_completed: int
    runs_failed: int
    current_phase: str  # idle, harvesting, training, deploying
    last_error: str | None


@dataclass
class RunResult:
    """Result of a scheduler run."""
    success: bool
    phase: str
    message: str
    duration_seconds: float
    harvest_stats: dict | None = None
    train_stats: dict | None = None
    deploy_stats: dict | None = None


class Scheduler:
    """Orchestrates the auto-harvest pipeline."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state_path = os.path.join(cfg.path("analysis_dir"), SCHEDULER_STATE_FILE)
        self.state = self._load_state()
        self.repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.venv_python = "/media/scott/data/finetune-venv/bin/python"

    def _load_state(self) -> SchedulerState:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                data = json.load(f)
            return SchedulerState(**data)
        return SchedulerState(
            last_run=0, last_harvest=0, last_train=0, last_deploy=0,
            runs_completed=0, runs_failed=0, current_phase="idle",
            last_error=None,
        )

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(asdict(self.state), f, indent=2)

    def _run_cmd(self, cmd: list[str], timeout: int = 3600) -> tuple[int, str]:
        """Run a command and return (returncode, output)."""
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = self.repo
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=self.repo, env=env,
            )
            output = result.stdout + result.stderr
            return result.returncode, output
        except subprocess.TimeoutExpired:
            return 1, "timeout"
        except Exception as e:
            return 1, str(e)

    def harvest(self) -> tuple[bool, dict]:
        """Run the harvest phase: extract + clean new data."""
        from src.harvest import plan_harvest, record_harvest

        plan = plan_harvest(self.cfg)

        if not plan.should_harvest:
            return True, {"skipped": True, "reason": plan.reason}

        print("[scheduler] starting harvest...")
        harvest_results = {"sources": plan.batch_labels, "total_new": plan.total_new}

        # Extract from each source with new data
        for label in plan.batch_labels:
            if label == "hermes":
                cmd = [self.venv_python, "-m", "src.cli", "hermes"]
            else:
                cmd = [self.venv_python, "-m", "src.cli", f"extract --label={label}"]

            rc, output = self._run_cmd(cmd, timeout=1800)
            if rc != 0:
                return False, {"error": f"extract failed for {label}: {output[-500:]}"}
            harvest_results[f"extract_{label}"] = "ok"

        # Clean all
        cmd = [self.venv_python, "-m", "src.cli", "clean"]
        rc, output = self._run_cmd(cmd, timeout=1800)
        if rc != 0:
            return False, {"error": f"clean failed: {output[-500:]}"}
        harvest_results["clean"] = "ok"

        # Record harvest
        for label in plan.batch_labels:
            record_harvest(self.cfg, label, plan.total_new)

        return True, harvest_results

    def train(self, labels: list[str] | None = None) -> tuple[bool, dict]:
        """Run the training phase on queued datasets."""
        from src.harvest import plan_harvest

        if labels is None:
            plan = plan_harvest(self.cfg)
            labels = plan.batch_labels

        if not labels:
            return True, {"skipped": True, "reason": "no labels to train"}

        print(f"[scheduler] training labels: {labels}")
        train_results = {"labels": labels}

        # Format datasets (safe while no training running)
        for label in labels:
            cmd = [self.venv_python, "-m", "src.cli", f"format --label={label}"]
            rc, output = self._run_cmd(cmd, timeout=1800)
            if rc != 0:
                print(f"[scheduler] WARNING: format failed for {label}")

        # Train each label sequentially
        for label in labels:
            print(f"[scheduler] training {label}...")
            cmd = [self.venv_python, "-m", "src.cli", f"train --label={label}"]
            rc, output = self._run_cmd(cmd, timeout=28800)  # 8h timeout
            if rc != 0:
                return False, {"error": f"train failed for {label}: {output[-500:]}"}
            train_results[f"train_{label}"] = "ok"

        return True, train_results

    def eval_and_select(self) -> tuple[bool, str, dict]:
        """Run eval, pick best adapter, merge."""
        print("[scheduler] running eval-all...")
        cmd = [self.venv_python, "-m", "src.cli", "eval-all", "--loss-only"]
        rc, output = self._run_cmd(cmd, timeout=7200)
        if rc != 0:
            return False, "", {"error": f"eval-all failed: {output[-500:]}"}

        # Pick best
        cmd = [self.venv_python, "-m", "src.cli", "best", "--metric=loss"]
        rc, output = self._run_cmd(cmd, timeout=300)
        if rc != 0:
            return False, "", {"error": "best failed"}

        # Parse winner from output
        winner = "combined"
        for line in output.split("\n"):
            if "toolcall-v5-3b-" in line:
                import re
                m = re.search(r"toolcall-v5-3b-([a-z0-9-]+)", line)
                if m:
                    winner = m.group(1)
                    break

        print(f"[scheduler] winner: {winner}")

        # Merge
        cmd = [self.venv_python, "-m", "src.cli", f"merge --label={winner}"]
        rc, output = self._run_cmd(cmd, timeout=3600)
        if rc != 0:
            return False, winner, {"error": f"merge failed: {output[-500:]}"}

        return True, winner, {"winner": winner}

    def deploy(self, label: str) -> tuple[bool, dict]:
        """Deploy the merged model to inference nodes."""
        from src.deploy import deploy_model

        print(f"[scheduler] deploying {label}...")
        result = deploy_model(self.cfg, label, target="local")

        return result.success, {
            "deployed": result.success,
            "path": result.deploy_path,
            "message": result.message,
        }

    def run_once(self, dry_run: bool = False) -> RunResult:
        """Run one complete cycle of the pipeline."""
        start = time.time()
        self.state.current_phase = "harvesting"
        self._save_state()

        if dry_run:
            from src.harvest import plan_harvest
            plan = plan_harvest(self.cfg)
            return RunResult(
                success=True, phase="dry-run",
                message=f"would harvest: {plan.should_harvest}, train: {plan.should_train}",
                duration_seconds=0,
                harvest_stats={"plan": plan.reason},
            )

        try:
            # Phase 1: Harvest
            print("[scheduler] === HARVEST ===")
            ok, harvest_stats = self.harvest()
            if not ok:
                self.state.current_phase = "idle"
                self.state.runs_failed += 1
                self.state.last_error = harvest_stats.get("error")
                self._save_state()
                return RunResult(
                    success=False, phase="harvest",
                    message=harvest_stats.get("error", "harvest failed"),
                    duration_seconds=time.time() - start,
                    harvest_stats=harvest_stats,
                )

            # Phase 2: Train
            self.state.current_phase = "training"
            self._save_state()
            print("[scheduler] === TRAIN ===")
            ok, train_stats = self.train()
            if not ok:
                self.state.current_phase = "idle"
                self.state.runs_failed += 1
                self.state.last_error = train_stats.get("error")
                self._save_state()
                return RunResult(
                    success=False, phase="train",
                    message=train_stats.get("error", "train failed"),
                    duration_seconds=time.time() - start,
                    harvest_stats=harvest_stats,
                    train_stats=train_stats,
                )

            # Phase 3: Eval and select
            print("[scheduler] === EVAL ===")
            ok, winner, eval_stats = self.eval_and_select()
            if not ok:
                self.state.current_phase = "idle"
                self.state.runs_failed += 1
                self.state.last_error = eval_stats.get("error")
                self._save_state()
                return RunResult(
                    success=False, phase="eval",
                    message=eval_stats.get("error", "eval failed"),
                    duration_seconds=time.time() - start,
                    harvest_stats=harvest_stats,
                    train_stats=train_stats,
                    deploy_stats=eval_stats,
                )

            # Phase 4: Deploy
            self.state.current_phase = "deploying"
            self._save_state()
            print("[scheduler] === DEPLOY ===")
            ok, deploy_stats = self.deploy(winner)

            # Update state
            self.state.current_phase = "idle"
            self.state.last_run = time.time()
            self.state.last_harvest = time.time()
            self.state.last_train = time.time()
            self.state.last_deploy = time.time()
            self.state.runs_completed += 1
            self.state.last_error = None
            self._save_state()

            return RunResult(
                success=ok, phase="complete",
                message=f"deployed {winner}" if ok else "deploy failed",
                duration_seconds=time.time() - start,
                harvest_stats=harvest_stats,
                train_stats=train_stats,
                deploy_stats=deploy_stats,
            )

        except Exception as e:
            self.state.current_phase = "idle"
            self.state.runs_failed += 1
            self.state.last_error = str(e)
            self._save_state()
            return RunResult(
                success=False, phase="error",
                message=str(e),
                duration_seconds=time.time() - start,
            )

    def loop(self, interval: int = 3600):
        """Run the scheduler in a loop."""
        print(f"[scheduler] starting loop (interval={interval}s)")
        while True:
            result = self.run_once()
            print(f"[scheduler] run complete: {result.success} "
                  f"phase={result.phase} msg={result.message}")

            if not result.success:
                print(f"[scheduler] sleeping {interval}s before retry...")
            else:
                print(f"[scheduler] sleeping {interval}s until next check...")

            time.sleep(interval)


def main(cfg: Config, argv: list[str]) -> int:
    """CLI handler for scheduler commands."""
    cmd = argv[1] if len(argv) > 1 else "scheduler-status"

    scheduler = Scheduler(cfg)

    if cmd == "scheduler-status":
        state = scheduler.state
        print("[scheduler-status]")
        print(f"  last_run: {state.last_run}")
        print(f"  runs_completed: {state.runs_completed}")
        print(f"  runs_failed: {state.runs_failed}")
        print(f"  current_phase: {state.current_phase}")
        if state.last_error:
            print(f"  last_error: {state.last_error}")
        return 0

    if cmd == "scheduler-run":
        dry_run = "--dry-run" in argv
        result = scheduler.run_once(dry_run=dry_run)
        print(f"[scheduler-run] success={result.success} phase={result.phase}")
        print(f"  message: {result.message}")
        if result.harvest_stats:
            print(f"  harvest: {result.harvest_stats}")
        if result.train_stats:
            print(f"  train: {result.train_stats}")
        if result.deploy_stats:
            print(f"  deploy: {result.deploy_stats}")
        return 0 if result.success else 1

    if cmd == "scheduler-loop":
        interval = 3600
        for arg in argv:
            if arg.startswith("--interval="):
                interval = int(arg.split("=", 1)[1])
        scheduler.loop(interval)
        return 0

    print("Commands: scheduler-status | scheduler-run [--dry-run] | scheduler-loop [--interval=3600]")
    return 0
