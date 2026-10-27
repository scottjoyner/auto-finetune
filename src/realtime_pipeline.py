"""Real-time pipeline orchestrator.

Ties together extraction, cleaning, formatting, and training into a
continuous, event-driven pipeline. Monitors data freshness, triggers
retraining when enough new data accumulates, and validates data quality
as it arrives.

Usage:
    python -m src.realtime_pipeline [--mode=watch|once] [--interval=60]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.config import Config, load
from src.locking import atomic_write_json
from src.pipeline_health import check_pipeline, validate_synergy


@dataclass
class PipelineState:
    """State of the real-time pipeline."""
    last_extraction: float = 0.0
    last_clean: float = 0.0
    last_format: float = 0.0
    last_train: float = 0.0
    last_health_check: float = 0.0
    total_extracted: int = 0
    total_cleaned: int = 0
    total_formatted: int = 0
    total_trained: int = 0
    new_since_train: int = 0
    runs_completed: int = 0
    runs_failed: int = 0
    current_phase: str = "idle"
    last_error: str | None = None
    # Thresholds
    min_new_for_train: int = 50
    min_interval_train: float = 3600  # seconds between training runs


class RealtimePipeline:
    """Orchestrates the real-time data pipeline."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state_path = os.path.join(cfg.path("analysis_dir"), "realtime-pipeline-state.json")
        self.state = self._load_state()
        self.repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _load_state(self) -> PipelineState:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                # Filter to known fields
                known = {f.name for f in PipelineState.__dataclass_fields__.values()}
                filtered = {k: v for k, v in data.items() if k in known}
                return PipelineState(**filtered)
            except Exception:
                pass
        return PipelineState()

    def _save_state(self) -> None:
        try:
            atomic_write_json(self.state_path, asdict(self.state))
        except Exception:
            pass

    def _run_cmd(self, cmd: list[str], timeout: int = 600,
                 extra_env: dict | None = None) -> tuple[int, str]:
        """Run a pipeline command."""
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = self.repo
            env.update(extra_env or {})
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=self.repo, env=env,
            )
            return result.returncode, (result.stdout + result.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            return 1, "timeout"
        except Exception as e:
            return 1, str(e)

    def _count_raw_files(self) -> int:
        """Count raw JSON files."""
        raw_dir = self.cfg.path("raw_dir")
        count = 0
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if f.endswith(".json"):
                    count += 1
        return count

    def _count_cleaned_files(self) -> int:
        """Count cleaned JSON files."""
        cleaned_dir = self.cfg.path("cleaned_dir")
        count = 0
        if os.path.isdir(cleaned_dir):
            for root, _, files in os.walk(cleaned_dir):
                for f in files:
                    if f.endswith(".json"):
                        count += 1
        return count

    def _count_dataset_rows(self) -> int:
        """Count total dataset rows."""
        dataset_dir = self.cfg.path("dataset_dir")
        total = 0
        if os.path.isdir(dataset_dir):
            for f in os.listdir(dataset_dir):
                if f.endswith(".jsonl"):
                    path = os.path.join(dataset_dir, f)
                    try:
                        with open(path) as fh:
                            for line in fh:
                                if line.strip():
                                    total += 1
                    except Exception:
                        pass
        return total

    def _extract(self) -> bool:
        """Run incremental extraction from all sources."""
        self.state.current_phase = "extracting"
        self._save_state()
        print("[pipeline] Phase: extraction")

        # Neo4j traces (incremental)
        rc, out = self._run_cmd(
            [sys.executable, "-m", "src.cli", "neo4j-traces", "--incremental"],
            timeout=300,
            extra_env={"NEO4J_TRACES_PASSWORD": "Pg5EajbfcuPYbhGSO2uBOv30",
                      "ROCM_PATH": "/opt/rocm"},
        )
        if rc != 0:
            print(f"[pipeline] Extraction failed: {out[-500:]}")
            self.state.last_error = out[-200:]
            return False

        # Count new files
        new_count = self._count_raw_files() - self.state.total_extracted
        self.state.total_extracted = self._count_raw_files()
        self.state.last_extraction = time.time()
        self.state.new_since_train += max(0, new_count)
        print(f"[pipeline] Extracted {new_count} new files (total: {self.state.total_extracted})")
        return True

    def _clean(self) -> bool:
        """Run cleaning on new data."""
        self.state.current_phase = "cleaning"
        self._save_state()
        print("[pipeline] Phase: cleaning")

        rc, out = self._run_cmd(
            [sys.executable, "-m", "src.cli", "clean", "--label=neo4j-traces"],
            timeout=300,
        )
        if rc != 0:
            print(f"[pipeline] Cleaning failed: {out[-500:]}")
            self.state.last_error = out[-200:]
            return False

        self.state.total_cleaned = self._count_cleaned_files()
        self.state.last_clean = time.time()
        print(f"[pipeline] Cleaned: {self.state.total_cleaned} total files")
        return True

    def _format(self) -> bool:
        """Run formatting on cleaned data."""
        self.state.current_phase = "formatting"
        self._save_state()
        print("[pipeline] Phase: formatting")

        rc, out = self._run_cmd(
            [sys.executable, "-m", "src.cli", "format", "--label=neo4j-traces"],
            timeout=300,
        )
        if rc != 0:
            print(f"[pipeline] Formatting failed: {out[-500:]}")
            self.state.last_error = out[-200:]
            return False

        self.state.total_formatted = self._count_dataset_rows()
        self.state.last_format = time.time()
        print(f"[pipeline] Formatted: {self.state.total_formatted} total rows")
        return True

    def _should_train(self) -> bool:
        """Determine if training should be triggered."""
        if self.state.new_since_train < self.state.min_new_for_train:
            print(f"[pipeline] Not enough new data for training: {self.state.new_since_train} < {self.state.min_new_for_train}")
            return False
        if time.time() - self.state.last_train < self.state.min_interval_train:
            remaining = self.state.min_interval_train - (time.time() - self.state.last_train)
            print(f"[pipeline] Training cooldown: {remaining:.0f}s remaining")
            return False
        return True

    def _train(self) -> bool:
        """Run training on new data."""
        self.state.current_phase = "training"
        self._save_state()
        print("[pipeline] Phase: training")

        env = {
            "TRAIN_MODEL_NAME": "Qwen/Qwen2.5-0.5B-Instruct",
            "TRAIN_MAX_SEQ_LENGTH": "64",
            "TRAIN_LOAD_4BIT": "0",
            "TRAIN_ALLOW_UNVERIFIED": "1",
            "AF_SKIP_UNMANAGED_CHECK": "1",
            "ROCM_PATH": "/opt/rocm",
        }
        rc, out = self._run_cmd(
            [sys.executable, "-m", "src.cli", "train",
             "--source=neo4j-traces", "--live",
             f"--max-examples={min(self.state.new_since_train, 500)}"],
            timeout=600,
            extra_env=env,
        )
        if rc != 0:
            print(f"[pipeline] Training failed: {out[-500:]}")
            self.state.last_error = out[-200:]
            self.state.runs_failed += 1
            return False

        self.state.total_trained += self.state.new_since_train
        self.state.new_since_train = 0
        self.state.last_train = time.time()
        self.state.runs_completed += 1
        print(f"[pipeline] Training complete: {self.state.total_trained} total trained")
        return True

    def _health_check(self) -> dict:
        """Run pipeline health check."""
        self.state.current_phase = "health_check"
        self._save_state()
        result = check_pipeline(self.cfg)
        issues = validate_synergy(self.cfg)
        self.state.last_health_check = time.time()
        print(f"[pipeline] Health: {result['status']} | Synergy: {'OK' if not issues else issues}")
        return result

    def run_once(self) -> dict:
        """Run one full pipeline cycle."""
        print(f"[pipeline] === Starting pipeline cycle ===")
        start = time.time()

        results = {
            "extraction": self._extract(),
            "cleaning": False,
            "formatting": False,
            "training": False,
            "health": None,
        }

        if results["extraction"]:
            results["cleaning"] = self._clean()
        if results["cleaning"]:
            results["formatting"] = self._format()
        if results["formatting"] and self._should_train():
            results["training"] = self._train()

        results["health"] = self._health_check()

        self.state.current_phase = "idle"
        self._save_state()

        duration = time.time() - start
        print(f"[pipeline] === Cycle complete in {duration:.1f}s ===")
        return results

    def run_loop(self, interval: int = 300) -> None:
        """Run the pipeline in a continuous loop."""
        print(f"[pipeline] Starting real-time pipeline (interval={interval}s)")
        while True:
            try:
                self.run_once()
                print(f"[pipeline] Sleeping {interval}s...")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("[pipeline] Stopped by user")
                self.state.current_phase = "idle"
                self._save_state()
                break
            except Exception as e:
                print(f"[pipeline] Error: {e}")
                self.state.last_error = str(e)
                self._save_state()
                time.sleep(interval)


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Real-time pipeline orchestrator", prog="realtime-pipeline")
    parser.add_argument("--mode", choices=["once", "loop"], default="once",
                        help="Run mode: once or loop")
    parser.add_argument("--interval", type=int, default=300,
                        help="Loop interval in seconds")
    parser.add_argument("--min-new", type=int, default=50,
                        help="Minimum new records to trigger training")
    args = parser.parse_args(argv)

    cfg = load()
    pipeline = RealtimePipeline(cfg)
    pipeline.state.min_new_for_train = args.min_new

    if args.mode == "loop":
        pipeline.run_loop(interval=args.interval)
    else:
        results = pipeline.run_once()
        print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
