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

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src import flags
from src.config import Config
from src.locking import atomic_write_json, lock_dir, unmanaged_training_processes

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
    pending_candidate: str | None = None
    pending_plan_id: str | None = None
    pending_labels: list[str] | None = None
    pending_dataset_labels: list[str] | None = None


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
    # Set only by --dry-run: the candidate pipeline's pre-validation preview
    # (datasets, partitions, audits) for the plan that would have run.
    candidate_stats: dict | None = None


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
            data.setdefault("pending_candidate", None)
            data.setdefault("pending_plan_id", None)
            data.setdefault("pending_labels", None)
            data.setdefault("pending_dataset_labels", None)
            return SchedulerState(**data)
        return SchedulerState(
            last_run=0, last_harvest=0, last_train=0, last_deploy=0,
            runs_completed=0, runs_failed=0, current_phase="idle",
            last_error=None,
        )

    def _save_state(self):
        atomic_write_json(self.state_path, asdict(self.state))

    def _run_cmd(self, cmd: list[str], timeout: int = 3600,
                 extra_env: dict[str, str] | None = None) -> tuple[int, str]:
        """Run a command and return (returncode, output)."""
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = self.repo
            env.update(extra_env or {})
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

    def _candidate_id(self, plan) -> str:
        payload = {
            "plan_id": plan.plan_id,
            "sources": [
                {
                    "source_id": source.source_id,
                    "total_sessions": source.total_sessions,
                    "dataset_label": source.dataset_label,
                }
                for source in plan.sources
                if source.name in plan.harvest_labels
            ],
            "format": self.cfg.get("format", default={}) or {},
            "clean": self.cfg.get("clean", default={}) or {},
            "eval": {
                "frac": float(self.cfg.get("scheduler", "eval_frac", default=0.1) or 0.1),
                "seed": int(self.cfg.get("scheduler", "eval_seed", default=42) or 42),
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]

    def _candidate_root(self, candidate_id: str) -> Path:
        return Path(self.cfg.path("analysis_dir")) / "candidates" / candidate_id

    def _pending_candidate(self, plan) -> Path | None:
        if self.state.pending_candidate and self.state.pending_plan_id == plan.plan_id:
            root = self._candidate_root(self.state.pending_candidate)
            if root.exists():
                return root
        return None

    def _save_pending(self, candidate_id: str | None, plan) -> None:
        self.state.pending_candidate = candidate_id
        self.state.pending_plan_id = plan.plan_id if candidate_id else None
        self.state.pending_labels = list(plan.harvest_labels) if candidate_id else None
        self.state.pending_dataset_labels = list(plan.dataset_labels or []) if candidate_id else None
        self._save_state()

    @staticmethod
    def _atomic_replace_dir(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        os.replace(src, tmp)
        if dst.exists():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        os.replace(tmp, dst)

    def _stage_cleaned(self, plan, root: Path) -> None:
        cleaned = Path(self.cfg.path("cleaned_dir"))
        staged = root / "cleaned"
        staged.mkdir(parents=True, exist_ok=True)
        for source in plan.sources:
            if source.name not in plan.harvest_labels:
                continue
            dataset_label = source.dataset_label or source.name
            if source.name == "hermes":
                dst = staged / dataset_label
                dst.mkdir(parents=True, exist_ok=True)
                for path in sorted(cleaned.glob("*.json")):
                    shutil.copy2(path, dst / path.name)
                continue
            src = cleaned / dataset_label
            if src.is_dir():
                shutil.copytree(src, staged / dataset_label, dirs_exist_ok=True)

    def _format_candidate(self, plan, root: Path) -> dict[str, Path]:
        from src.analyze import benchmark_session_ids
        from src.format_dataset import main as format_main

        candidate_cfg = Config(raw=dict(self.cfg.raw))
        candidate_cfg.raw.setdefault("paths", {})["cleaned_dir"] = str(root / "cleaned")
        candidate_cfg.raw["paths"]["dataset_dir"] = str(root / "datasets")
        datasets = root / "datasets"
        datasets.mkdir(parents=True, exist_ok=True)
        # Every candidate is held out from the benchmark suite, not just the
        # main corpus: an earlier holdout recovered nothing because the suite
        # stores its id under `id` rather than `task_id`.
        held_out = benchmark_session_ids(
            Path(self.repo) / "eval" / "tasks" / "auto-verified.jsonl")
        outputs: dict[str, Path] = {}
        for source_name, dataset_label in zip(plan.harvest_labels, plan.dataset_labels or plan.harvest_labels):
            source = next((s for s in plan.sources if s.name == source_name), None)
            if source is None:
                continue
            count = format_main(candidate_cfg, label=dataset_label, exclude=held_out)
            output = datasets / f"train.{dataset_label}.jsonl"
            if count <= 0 or not output.is_file():
                raise RuntimeError(f"candidate format produced no rows for {dataset_label}")
            outputs[dataset_label] = output
        if held_out:
            from src.format_dataset import verify_holdout
            v = verify_holdout(str(datasets), held_out)
            if v["status"] == "contaminated":
                raise RuntimeError(
                    f"candidate corpus contains {len(v['leaked'])} held-out benchmark "
                    f"sessions: {v['leaked'][:5]}")
            if v["status"] == "unverifiable":
                raise RuntimeError(
                    "candidate corpus has no provenance sidecars; cannot confirm the "
                    "benchmark holdout held")
        return outputs

    def _partition_candidate(self, datasets: dict[str, Path], root: Path) -> dict[str, dict]:
        from src.eval import build_disjoint_partition

        frac = float(self.cfg.get("scheduler", "eval_frac", default=0.1) or 0.1)
        seed = int(self.cfg.get("scheduler", "eval_seed", default=42) or 42)
        partitions: dict[str, dict] = {}
        for label, source_path in datasets.items():
            result = build_disjoint_partition(source_path, root / "partitions" / label,
                                              frac=frac, seed=seed, label=label)
            if result.contamination["status"] != "clean":
                raise RuntimeError(f"partition contamination for {label}")
            partitions[label] = {
                "train_path": str(result.train_path),
                "eval_path": str(result.eval_path),
                "source_sha256": result.source_sha256,
                "seed": seed,
                "frac": frac,
                "n_train": len(result.train_rows),
                "n_eval": len(result.eval_rows),
            }
        return partitions

    def _audit_candidate(self, datasets: dict[str, Path]) -> tuple[dict[str, dict],
                                                                  dict[str, Path]]:
        """Audit each mix against the bench suite, decontaminating if needed.

        Returns ``(audits, datasets)`` where ``datasets`` may point at
        rewritten, decontaminated files. A candidate that cannot reach clean
        still raises -- decontamination drops rows, it never relaxes the
        standard.
        """
        from src.audit import _load_jsonl, audit_leakage, decontaminate

        bench_path = Path(self.repo) / "eval" / "tasks" / "auto-verified.jsonl"
        if not bench_path.is_file():
            raise RuntimeError(f"benchmark suite not found: {bench_path}")
        bench_rows = _load_jsonl(str(bench_path))
        audits: dict[str, dict] = {}
        out: dict[str, Path] = {}
        for label, train_path in datasets.items():
            rows = _load_jsonl(str(train_path))
            result = audit_leakage(rows, bench_rows)
            if result["status"] != "clean":
                # Any non-clean status is handled the same way: try to drop the
                # offending rows, and still raise unless that reaches clean.
                # Only special-casing "contaminated" would let "leaked" and
                # "not_evaluable" pass through untouched.
                kept, dropped, result = decontaminate(rows, bench_rows)
                if result["status"] != "clean":
                    raise RuntimeError(
                        f"candidate {label} failed contamination audit: "
                        f"{result['status']} ({result['n_hits']} hits in "
                        f"{result['n_train']} rows)")
                clean_path = Path(str(train_path).replace(".jsonl", ".decontaminated.jsonl"))
                with open(clean_path, "w") as f:
                    for row in kept:
                        f.write(json.dumps(row) + "\n")
                result = dict(result, n_dropped=len(dropped),
                              dropped_refs=dropped,
                              cleaned_path=str(clean_path))
                print(f"[audit] decontaminated {label}: dropped {len(dropped)} of "
                      f"{len(rows)} rows that carried benchmark text verbatim "
                      f"({result['hit_rate']:.1%} of bench tasks affected)")
                out[label] = clean_path
            else:
                out[label] = train_path
            audits[label] = result
        return audits, out

    def _train_candidate(self, plan, root: Path, partitions: dict[str, dict]) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for source_name, dataset_label in zip(plan.harvest_labels, plan.dataset_labels or plan.harvest_labels):
            source = next((s for s in plan.sources if s.name == source_name), None)
            if source is None or dataset_label not in partitions:
                continue
            adapter = root / "adapters" / dataset_label
            cmd = [self.venv_python, "-m", "src.cli", "train", f"--label={dataset_label}"]
            env = {
                "TRAIN_DATASET_DIR": partitions[dataset_label]["train_path"],
                "TRAIN_OUTPUT_DIR": str(adapter),
            }
            rc, output = self._run_cmd(cmd, timeout=int(
                self.cfg.get("scheduler", "train_timeout_seconds", default=86400) or 86400),
                extra_env=env)
            if rc != 0:
                raise RuntimeError(f"candidate train failed for {dataset_label}: {output[-500:]}")
            if not (adapter / "adapter_config.json").is_file():
                raise RuntimeError(f"candidate adapter incomplete for {dataset_label}")
            outputs[dataset_label] = str(adapter)
        return outputs

    def _eval_candidate(self, root: Path, partitions: dict[str, dict],
                        adapters: dict[str, str]) -> dict[str, float]:
        from src.eval import evaluate

        base = self.cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
        losses: dict[str, float] = {}
        for label, partition in partitions.items():
            result = evaluate(adapters[label], str(base), partition["eval_path"],
                              loss_only=True)
            if result.n_held_out <= 0 or result.loss != result.loss:
                raise RuntimeError(f"candidate evaluation invalid for {label}")
            losses[label] = result.loss
        return losses

    def _merge_candidate(self, root: Path, label: str, adapter: str) -> str:
        from src.merge import merge_adapter

        base = self.cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
        merged = root / "merged"
        merge_adapter(adapter, str(base), str(merged), rocm=False)
        if not (merged / "config.json").is_file() or not (merged / "tokenizer.json").is_file():
            raise RuntimeError("candidate merge output is incomplete")
        return str(merged)

    def _benchmark_candidate(self, merged: str) -> dict:
        from src.bench import bench_suite, load_tasks, make_driver
        from src.train import _detect_rocm

        tasks_path = Path(self.repo) / "eval" / "tasks" / "auto-verified.jsonl"
        tasks = load_tasks(str(tasks_path))
        driver = make_driver("subagent", model_path=merged, rocm=_detect_rocm())
        results = bench_suite(driver, tasks, merged, "subagent")
        passed = sum(1 for result in results if result.success)
        return {"tasks": len(results), "passed": passed,
                "pass_rate": passed / len(results) if results else 0.0,
                "results": [asdict(result) for result in results]}

    def _candidate_pipeline(self, plan, root: Path) -> dict:
        self._stage_cleaned(plan, root)
        datasets = self._format_candidate(plan, root)
        audits, datasets = self._audit_candidate(datasets)
        partitions = self._partition_candidate(datasets, root)
        adapters = self._train_candidate(plan, root, partitions)
        losses = self._eval_candidate(root, partitions, adapters)
        winner = min(losses, key=losses.get)
        merged = self._merge_candidate(root, winner, adapters[winner])
        benchmark = self._benchmark_candidate(merged)
        manifest = {
            "schema_version": 1,
            "candidate_id": root.name,
            "plan_id": plan.plan_id,
            "sources": [asdict(source) for source in plan.sources],
            "dataset_labels": list(plan.dataset_labels or []),
            "datasets": {label: str(path) for label, path in datasets.items()},
            "partitions": partitions,
            "audits": audits,
            "eval_losses": losses,
            "winner": winner,
            "adapter": adapters[winner],
            "merged": merged,
            "benchmark": benchmark,
        }
        atomic_write_json(root / "candidate.json", manifest)
        return manifest

    def _promote_candidate(self, plan, manifest: dict) -> None:
        winner = manifest["winner"]
        merged = manifest["merged"]
        # Into train.output_dir itself, NOT its parent. This used to take
        # os.path.dirname(output_dir), landing the merged model one level up
        # (outputs/ rather than outputs/checkpoints/), while every consumer --
        # deploy.py, quantize.py, merge -- looks for
        # <output_dir>/toolcall-v5-3b-<label>-merged. Since the pipeline was
        # never wired into run_once this never surfaced; with it wired, deploy
        # would have run straight after promotion and not found the model.
        output_root = Path(str(self.cfg.get(
            "train", "output_dir",
            default="/media/scott/data/finetune-staging/outputs/checkpoints")))
        target = output_root / f"toolcall-v5-3b-{winner}-merged"
        target.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_replace_dir(Path(merged), target)
        from src.harvest import record_harvest
        selected = [source for source in plan.sources if source.name in plan.harvest_labels]
        record_harvest(self.cfg, selected, plan_id=plan.plan_id)
        self._save_pending(None, plan)

    def _dry_run_candidate(self, plan, candidate_id: str, root: Path) -> dict:
        """Run the candidate pipeline's cheap stages and stop.

        Validates everything that can fail before the GPU is committed: corpus
        staging, formatting, the disjoint held-out split (which fails closed on
        contamination) and the bench leakage audit. Skips train/eval/merge/
        benchmark/promote.

        Deliberately writes candidate-dryrun.json rather than candidate.json,
        and does not touch pending state, so a later real run neither mistakes
        this for a finished pipeline nor resumes from it.
        """
        root.mkdir(parents=True, exist_ok=True)
        self._stage_cleaned(plan, root)
        datasets = self._format_candidate(plan, root)
        audits, datasets = self._audit_candidate(datasets)
        partitions = self._partition_candidate(datasets, root)
        preview = {
            "schema_version": 0,
            "dry_run": True,
            "candidate_id": candidate_id,
            "plan_id": plan.plan_id,
            "dataset_labels": list(plan.dataset_labels or []),
            "datasets": {label: str(path) for label, path in datasets.items()},
            "partitions": {
                label: {"n_train": part["n_train"], "n_eval": part["n_eval"],
                        "source_sha256": part["source_sha256"],
                        "seed": part["seed"], "frac": part["frac"]}
                for label, part in partitions.items()
            },
            "audits": audits,
            "skipped_stages": ["train", "eval", "merge", "benchmark", "promote"],
        }
        path = root / "candidate-dryrun.json"
        atomic_write_json(path, preview)
        return {"dry_run": True, "candidate_id": candidate_id,
                "manifest": str(path), **preview}

    def run_candidate(self, plan, dry_run: bool = False) -> tuple[bool, dict]:
        candidate_id = self.state.pending_candidate or self._candidate_id(plan)
        root = self._candidate_root(candidate_id)
        if dry_run:
            return True, self._dry_run_candidate(plan, candidate_id, root)
        manifest_path = root / "candidate.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("plan_id") != plan.plan_id:
                raise RuntimeError("pending candidate does not match the current plan")
            self._promote_candidate(plan, manifest)
            return True, {"candidate_id": candidate_id, "promoted": True}
        root.mkdir(parents=True, exist_ok=True)
        self._save_pending(candidate_id, plan)
        try:
            manifest = self._candidate_pipeline(plan, root)
            self._promote_candidate(plan, manifest)
            return True, {"candidate_id": candidate_id, "promoted": True,
                          "winner": manifest["winner"],
                          "eval_losses": manifest.get("eval_losses"),
                          "audits": manifest.get("audits"),
                          "benchmark": manifest["benchmark"]}
        except Exception as exc:
            self.state.last_error = str(exc)
            self._save_state()
            raise

    def harvest(self, plan) -> tuple[bool, dict]:
        """Extract and clean without advancing the promotion watermark."""
        from src.harvest import record_extraction

        if not plan.should_harvest:
            return True, {"skipped": True, "reason": plan.reason}

        print("[scheduler] starting harvest...")
        harvest_results = {"sources": plan.harvest_labels, "total_new": plan.total_new}

        for label in plan.harvest_labels:
            if label == "hermes":
                cmd = [self.venv_python, "-m", "src.cli", "hermes"]
            elif label == "opencode":
                cmd = [self.venv_python, "-m", "src.cli", "extract", "--label=ssd"]
            else:
                cmd = [self.venv_python, "-m", "src.cli", "extract", f"--label={label}"]

            rc, output = self._run_cmd(cmd, timeout=1800)
            if rc != 0:
                return False, {"error": f"extract failed for {label}: {output[-500:]}"}
            harvest_results[f"extract_{label}"] = "ok"

        cmd = [self.venv_python, "-m", "src.cli", "clean"]
        rc, output = self._run_cmd(cmd, timeout=1800)
        if rc != 0:
            return False, {"error": f"clean failed: {output[-500:]}"}
        harvest_results["clean"] = "ok"

        selected = [s for s in plan.sources if s.name in plan.harvest_labels]
        record_extraction(self.cfg, selected)
        return True, harvest_results

    def train(self, labels: list[str]) -> tuple[bool, dict]:
        """Run the training phase on queued datasets."""
        if not labels:
            return True, {"skipped": True, "reason": "no labels to train"}

        print(f"[scheduler] training labels: {labels}")
        train_results = {"labels": labels}

        # Format datasets (safe while no training running)
        for label in labels:
            if label == "hermes":
                cmd = [self.venv_python, "-m", "src.cli", "format", "--source=hermes"]
            else:
                dataset_label = "ssd" if label == "opencode" else label
                cmd = [self.venv_python, "-m", "src.cli", "format",
                       f"--label={dataset_label}"]
            rc, output = self._run_cmd(cmd, timeout=1800)
            if rc != 0:
                return False, {"error": f"format failed for {label}: {output[-500:]}"}

        # Train each label sequentially
        for label in labels:
            print(f"[scheduler] training {label}...")
            dataset_label = "hermes" if label == "hermes" else ("ssd" if label == "opencode" else label)
            flag = f"--source={dataset_label}" if label == "hermes" else f"--label={dataset_label}"
            cmd = [self.venv_python, "-m", "src.cli", "train", flag]
            configured = self.cfg.get("train", "output_dir", default="outputs/checkpoints")
            output_root = os.path.dirname(str(configured))
            output_dir = os.path.join(output_root, f"toolcall-v5-3b-{dataset_label}")
            timeout = int(self.cfg.get("scheduler", "train_timeout_seconds", default=86400) or 86400)
            rc, output = self._run_cmd(
                cmd, timeout=timeout, extra_env={"TRAIN_OUTPUT_DIR": output_dir})
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
        cmd = [self.venv_python, "-m", "src.cli", "merge", f"--label={winner}"]
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
        from src.harvest import plan_harvest
        plan = plan_harvest(self.cfg)

        if dry_run:
            preview: dict = {
                "success": True, "phase": "dry-run",
                "message": (f"would harvest: {plan.should_harvest}, "
                            f"train: {plan.should_train}"),
                "duration_seconds": 0,
                "harvest_stats": {"plan": plan.reason},
            }
            # Pre-validate the candidate pipeline's cheap stages, so a bad
            # corpus, a contaminated split or bench leakage surfaces now rather
            # than after the GPU hours are spent. Anything that raises here is
            # the fail-closed behaviour being reported, not a new failure mode.
            if plan.should_train:
                try:
                    candidate_id = self.state.pending_candidate or self._candidate_id(plan)
                    _, details = self.run_candidate(plan, dry_run=True)
                    preview["message"] += (
                        f"; candidate {candidate_id} pre-validated "
                        f"(train/eval/merge/benchmark/promote skipped)")
                    preview["candidate_stats"] = details
                except Exception as exc:  # noqa: BLE001
                    return RunResult(
                        False, "dry-run",
                        f"candidate pre-validation failed: {exc}",
                        time.time() - start,
                        harvest_stats={"plan": plan.reason},
                    )
            return RunResult(**preview)

        source_errors = [s.error for s in plan.sources if s.error]
        if source_errors:
            return RunResult(False, "planning", plan.reason, time.time() - start)
        if plan.should_train:
            legacy = unmanaged_training_processes(lock_dir(self.cfg))
            if legacy:
                return RunResult(
                    False, "busy",
                    f"legacy unleased trainer active; refusing dataset/GPU phases: {legacy}",
                    time.time() - start,
                )
        if not plan.should_harvest:
            return RunResult(True, "skipped", plan.reason, time.time() - start,
                             harvest_stats={"skipped": True, "plan": plan.reason})

        self.state.current_phase = "harvesting"
        self._save_state()

        try:
            # Phase 1: Harvest
            print("[scheduler] === HARVEST ===")
            ok, harvest_stats = self.harvest(plan)
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

            self.state.last_harvest = time.time()
            if not plan.should_train:
                self.state.current_phase = "idle"
                self.state.last_run = time.time()
                self.state.runs_completed += 1
                self.state.last_error = None
                self._save_state()
                return RunResult(
                    success=True, phase="harvested",
                    message="new traces harvested; training threshold not reached",
                    duration_seconds=time.time() - start,
                    harvest_stats=harvest_stats,
                )

            # Phases 2+3: the candidate pipeline.
            #
            # This replaces the legacy train() + eval_and_select() pair, which
            # wrote adapters straight into outputs/checkpoints and picked the
            # winner by regex-parsing the stdout of `src.cli best`. The candidate
            # pipeline instead stages, formats, partitions, audits, trains,
            # evals, merges and benchmarks inside a candidate root, and only
            # then promotes atomically and records the harvest watermark -- so a
            # failure at any stage leaves the live artifacts untouched and the
            # source eligible for the next run.
            #
            # Consequences to be aware of:
            #  * adapters now land in the candidate root, not
            #    outputs/checkpoints/<label>/, so the weekly ml-state backup
            #    (which walks outputs/checkpoints) sees only promoted merges;
            #  * a promotion now additionally requires the bench suite to run,
            #    so a candidate that trains cleanly but benchmarks badly will
            #    not be promoted.
            self.state.current_phase = "training"
            self._save_state()
            print("[scheduler] === CANDIDATE PIPELINE ===")
            try:
                ok, candidate_stats = self.run_candidate(plan)
            except Exception as exc:  # noqa: BLE001
                self.state.current_phase = "idle"
                self.state.runs_failed += 1
                self.state.last_error = str(exc)
                self._save_state()
                return RunResult(
                    success=False, phase="candidate",
                    message=f"candidate pipeline failed: {exc}",
                    duration_seconds=time.time() - start,
                    harvest_stats=harvest_stats,
                )

            if not ok:
                self.state.current_phase = "idle"
                self.state.runs_failed += 1
                self.state.last_error = str(candidate_stats.get("error")
                                            or "candidate pipeline failed")
                self._save_state()
                return RunResult(
                    success=False, phase="candidate",
                    message=self.state.last_error,
                    duration_seconds=time.time() - start,
                    harvest_stats=harvest_stats,
                    candidate_stats=candidate_stats,
                )

            winner = candidate_stats.get("winner", "")
            train_stats = {"candidate": candidate_stats}
            if not winner:
                self.state.current_phase = "idle"
                self.state.runs_failed += 1
                self.state.last_error = "candidate produced no winner"
                self._save_state()
                return RunResult(
                    success=False, phase="candidate",
                    message="candidate produced no winner",
                    duration_seconds=time.time() - start,
                    harvest_stats=harvest_stats,
                    candidate_stats=candidate_stats,
                )
            print(f"[scheduler] candidate winner: {winner}")

            # Phase 4: Deploy
            self.state.current_phase = "deploying"
            self._save_state()
            print("[scheduler] === DEPLOY ===")
            ok, deploy_stats = self.deploy(winner)

            if not ok:
                self.state.current_phase = "idle"
                self.state.last_run = time.time()
                self.state.runs_failed += 1
                self.state.last_error = str(deploy_stats.get("message") or "deploy failed")
                self._save_state()
                return RunResult(
                    success=False, phase="deploy",
                    message=self.state.last_error,
                    duration_seconds=time.time() - start,
                    harvest_stats=harvest_stats,
                    train_stats=train_stats,
                    deploy_stats=deploy_stats,
                )

            # Update state
            self.state.current_phase = "idle"
            self.state.last_run = time.time()
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
                candidate_stats=candidate_stats,
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
        if result.candidate_stats:
            cand = result.candidate_stats
            print(f"  candidate {cand.get('candidate_id')} (pre-validated)")
            for label, part in (cand.get("partitions") or {}).items():
                audit = (cand.get("audits") or {}).get(label, {})
                print(f"    {label}: train={part['n_train']} eval={part['n_eval']} "
                      f"contamination={audit.get('status', '?')}")
            print(f"    skipped: {', '.join(cand.get('skipped_stages', []))}")
        if result.train_stats:
            print(f"  train: {result.train_stats}")
        if result.deploy_stats:
            print(f"  deploy: {result.deploy_stats}")
        return 0 if result.success else 1

    if cmd == "scheduler-loop":
        interval = 3600
        value, err = flags.parse_number(argv, "--interval", cast=int,
                                        minimum=1)
        if err:
            print(f"[error] {err}")
            return 2
        if value is not None:
            interval = value
        scheduler.loop(interval)
        return 0

    print("Commands: scheduler-status | scheduler-run [--dry-run] | scheduler-loop [--interval=3600]")
    return 0
