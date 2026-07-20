"""CLI entrypoint: `python -m src.cli <command>`."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src.config import load
from src.train import TrainError


def _parse_source(argv: list[str]) -> str | None:
    """Extract --source=<name> from argv (hermes|opencode)."""
    for a in argv:
        if a.startswith("--source="):
            return a.split("=", 1)[1]
    return None


def _parse_label(argv: list[str]) -> str | None:
    """Extract --label=<name> from argv."""
    for a in argv:
        if a.startswith("--label="):
            return a.split("=", 1)[1]
    return None


def _parse_int_flag(argv: list[str], name: str) -> int | None:
    """Extract --name=<int> from argv (e.g. --max-examples=100)."""
    for a in argv:
        if a.startswith(f"{name}="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _parse_str_flag(argv: list[str], name: str) -> str | None:
    """Extract --name=<value> from argv (e.g. --project=portfolio)."""
    for a in argv:
        if a.startswith(f"{name}="):
            return a.split("=", 1)[1]
    return None


def _has_flag(argv: list[str], name: str) -> bool:
    """True if --name appears in argv."""
    return any(a == name for a in argv)


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "help"
    cfg = load()
    source = _parse_source(argv)
    label = _parse_label(argv)
    project = _parse_str_flag(argv, "project")
    keep_reasoning = _has_flag(argv, "--keep-reasoning")
    try:
        if cmd == "extract":
            from src.extract_opencode import main as run
            cfg.ensure_dirs()
            return run(cfg, label=label, project=project)
        if cmd == "hermes":
            from src.extract_hermes import main as run
            cfg.ensure_dirs()
            return run(cfg)
        if cmd == "clean":
            from src.clean import main as run
            return run(cfg, label=label, keep_reasoning=keep_reasoning)
        if cmd == "format":
            from src.format_dataset import main as run
            # --all-split: produce hermes-only, opencode-only, and merged
            if "--all-split" in argv:
                n = 0
                for s in ("hermes", "opencode", None):
                    n += run(cfg, source=s, label=label)
                return n
            return run(cfg, source=source, label=label)
        if cmd == "combine":
            from src.format_dataset import combine as run
            run(cfg)
            return 0
        if cmd == "train":
            from src.train import main as run
            dry = "--dry-run" in argv
            max_ex = _parse_int_flag(argv, "--max-examples")
            return run(cfg, dry_run=dry, source=source, label=label, max_examples=max_ex)
        if cmd == "eval-split":
            from src.eval import build_held_out
            frac = float(_parse_str_flag(argv, "--frac") or "0.1")
            if not label:
                print("[error] eval-split requires --label=<name>")
                return 2
            build_held_out(cfg.path("dataset_dir"), label, frac=frac)
            return 0
        if cmd == "eval":
            from src.eval import evaluate, evaluate_baseline
            if not label:
                print("[error] eval requires --label=<name>")
                return 2
            from src.train import _detect_rocm
            dset = cfg.path("dataset_dir")
            out_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
            adapter = os.environ.get("TRAIN_OUTPUT_DIR") or os.path.join(out_base, f"toolcall-v5-3b-{label}")
            held = Path(dset).parent / "eval" / f"held-out-{label}.jsonl"
            if not held.exists():
                print(f"[eval] no held-out split at {held}; run `eval-split --label={label}` first")
                return 2
            base = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
            print(f"[eval] baseline (base model) on {label}...")
            b = evaluate_baseline(base, str(held), rocm=_detect_rocm())
            print(f"[eval] adapter {adapter} on {label}...")
            a = evaluate(adapter, base, str(held), rocm=_detect_rocm())
            print(json.dumps({"baseline": b.as_dict(), "adapter": a.as_dict()}, indent=2))
            return 0
        if cmd == "eval-all":
            from src.eval import evaluate_all, format_report, write_report
            from src.train import _detect_rocm
            dset = cfg.path("dataset_dir")
            out_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
            eval_dir = Path(dset).parent / "eval"
            base = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
            loss_only = "--loss-only" in argv
            results = evaluate_all(base, out_base, str(eval_dir), rocm=_detect_rocm(), loss_only=loss_only)
            print(format_report(results))
            if "--report" in argv:
                path = write_report(results, out_dir=str(Path(dset).parent / "eval-reports"), tag="eval-all")
                print(f"[eval-all] report written -> {path}")
            return 0
        if cmd == "best":
            from src.eval import evaluate_all, best_adapter, format_report
            from src.train import _detect_rocm
            dset = cfg.path("dataset_dir")
            out_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
            eval_dir = Path(dset).parent / "eval"
            base = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
            metric = _parse_str_flag(argv, "--metric") or "loss"
            results = evaluate_all(base, out_base, str(eval_dir), rocm=_detect_rocm(), loss_only=True)
            winner = best_adapter(results, metric=metric)
            if winner is None:
                print("[best] no finished adapters found")
                return 1
            print(f"[best] by {metric}: {winner.adapter} (loss={winner.loss:.4f}, "
                  f"tool_exact={winner.tool.exact_match:.3f})")
            return 0
        if cmd == "sanity":
            from src.eval import sanity_check_adapters
            from src.train import _detect_rocm
            out_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
            base = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
            labels = [x for x in ("ssd", "nas5-main", "nas5-20260717", "opencode-all",
                                   "opencode-portfolio", "hermes-reasoning", "combined")]
            status = sanity_check_adapters(base, out_base, labels, rocm=_detect_rocm())
            for k, v in status.items():
                print(f"  {k:<22} {v}")
            return 0
        if cmd == "merge":
            from src.merge import merge_adapter
            from src.train import _detect_rocm
            if not label:
                print("[error] merge requires --label=<name>")
                return 2
            base = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
            out_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
            adapter = os.environ.get("TRAIN_OUTPUT_DIR") or os.path.join(out_base, f"toolcall-v5-3b-{label}")
            if not os.path.exists(os.path.join(adapter, "adapter_config.json")):
                print(f"[error] adapter not found: {adapter}")
                return 2
            merged = os.path.join(out_base, f"toolcall-v5-3b-{label}-merged")
            merge_adapter(adapter, base, merged, rocm=_detect_rocm())
            return 0
        if cmd == "report":
            # One-shot consolidated eval: loss table + probe + best, written to
            # a local report. Runs eval-all (loss_only if training busy) plus
            # probe + best, then persists everything via write_report.
            from src.eval import (evaluate_all, format_report, write_report,
                                  best_adapter, grade_probe, grade_probe_baseline,
                                  load_probe_set)
            from src.train import _detect_rocm
            dset = cfg.path("dataset_dir")
            out_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
            eval_dir = Path(dset).parent / "eval"
            base = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
            # loss-only is safe while training runs; full gen skipped to avoid GPU fight
            results = evaluate_all(base, out_base, str(eval_dir), rocm=_detect_rocm(), loss_only=True)
            print(format_report(results))
            winner = best_adapter(results, "loss")
            if winner:
                print(f"[report] best by loss: {winner.adapter} (loss={winner.loss:.4f})")
            # probe (qualitative) — needs generation; run if GPU looks free-ish.
            probe_path = os.path.join(os.path.dirname(__file__), "..", "eval", "probe.jsonl")
            pb = pa = None
            if label:
                adapter = os.environ.get("TRAIN_OUTPUT_DIR") or os.path.join(out_base, f"toolcall-v5-3b-{label}")
                if os.path.exists(os.path.join(adapter, "adapter_config.json")):
                    print(f"[report] probe on {label}...")
                    pa = grade_probe(adapter, base, probe_path, rocm=_detect_rocm(), label=label)
                    pb = grade_probe_baseline(base, probe_path, rocm=_detect_rocm(), label=label)
                    print(f"  base {pb.checks_passed}/{pb.checks_total}  adapter {pa.checks_passed}/{pa.checks_total}")
            path = write_report(results, probe_base=pb, probe_adapter=pa,
                                out_dir=str(Path(dset).parent / "eval-reports"), tag="report")
            print(f"[report] written -> {path}")
            return 0
        if cmd == "probe":
            from src.eval import grade_probe, grade_probe_baseline
            from src.train import _detect_rocm
            if not label:
                print("[error] probe requires --label=<name>")
                return 2
            base = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
            out_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
            adapter = os.environ.get("TRAIN_OUTPUT_DIR") or os.path.join(out_base, f"toolcall-v5-3b-{label}")
            probe_path = os.path.join(os.path.dirname(__file__), "..", "eval", "probe.jsonl")
            print(f"[probe] baseline (base model)...")
            b = grade_probe_baseline(base, probe_path, rocm=_detect_rocm(), label=label)
            print(f"[probe] adapter {adapter}...")
            a = grade_probe(adapter, base, probe_path, rocm=_detect_rocm(), label=label)
            print(json.dumps({"baseline": b.as_dict(), "adapter": a.as_dict()}, indent=2))
            return 0
        if cmd == "compare":
            from src.eval import compare_probes, format_probe_comparison
            from src.train import _detect_rocm
            base = cfg.get("train", "model_name", default="Qwen/Qwen2.5-7B-Instruct")
            out_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
            probe_path = os.path.join(os.path.dirname(__file__), "..", "eval", "probe.jsonl")
            labels = [x for x in ("ssd", "nas5-main", "nas5-20260717", "opencode-all",
                                   "opencode-portfolio", "hermes-reasoning", "combined")]
            # optional scope: --label=X runs only that bucket's probes (vs base)
            scope = label
            print(f"[compare] running curated probes across base + finished adapters"
                  f"{(' (scope=' + scope + ')') if scope else ''} ...")
            results = compare_probes(out_base, base, probe_path, labels,
                                     rocm=_detect_rocm(), label=scope)
            print(format_probe_comparison(results))
            return 0
        if cmd == "bench":
            from src.bench import (load_tasks, make_driver, bench_suite,
                                   format_bench_results)
            from src.train import _detect_rocm
            # register the local-chat (standard HF model) runner
            import src.drivers_localchat  # noqa: F401  (self-registers)
            # runner selection
            runner = _parse_str_flag(argv, "--runner") or "self"
            tasks_path = (_parse_str_flag(argv, "--tasks")
                          or os.path.join(os.path.dirname(__file__), "..",
                                         "eval", "tasks", "sample.jsonl"))
            tasks = load_tasks(tasks_path)
            # which model to drive
            model_arg = _parse_str_flag(argv, "--model")
            if runner == "self":
                if not model_arg:
                    print("[error] bench --runner=self requires --model=<local HF dir>")
                    return 2
                driver = make_driver("self", model_path=model_arg,
                                     rocm=_detect_rocm())
                model_name = model_arg
            elif runner == "api":
                base_url = _parse_str_flag(argv, "--base-url")
                model = model_arg or _parse_str_flag(argv, "--api-model")
                if (not base_url or not model) and "--fleet" in argv:
                    from src.fleet import pick_model
                    hint = _parse_str_flag(argv, "--fleet-hint")
                    picked = pick_model(hint)
                    if picked is None:
                        print("[error] bench --fleet found no online large model in endpoints.json")
                        return 2
                    base_url, model = picked["base_url"], picked["model"]
                    print(f"[bench] fleet picked: {model} @ {base_url}")
                if not (base_url and model):
                    print("[error] bench --runner=api needs --base-url + --api-model (or --fleet)")
                    return 2
                driver = make_driver("api", base_url=base_url, model=model)
                model_name = model
            else:
                driver = make_driver(runner, model_path=model_arg)
                model_name = model_arg or runner
            print(f"[bench] runner={runner} model={model_name} tasks={len(tasks)}")
            results = bench_suite(driver, tasks, model_name, runner)
            print(format_bench_results(results))
            return 0
        if cmd == "bench-matrix":
            from src.bench import (load_tasks, bench_matrix,
                                    format_bench_matrix)
            from src.train import _detect_rocm
            import src.drivers_localchat  # noqa: F401  (self-registers "local-chat")
            tasks_path = (_parse_str_flag(argv, "--tasks")
                          or os.path.join(os.path.dirname(__file__), "..",
                                         "eval", "tasks", "sample.jsonl"))
            tasks = load_tasks(tasks_path)
            specs_json = _parse_str_flag(argv, "--specs")
            preset = _parse_str_flag(argv, "--preset")
            specs: list = []
            if specs_json:
                specs = json.loads(specs_json)
            elif preset == "local-refs":
                # local, no-network reference set: base + any finished adapters
                out_base = "/media/scott/data/finetune-staging/outputs/checkpoints"
                base_local = (os.environ.get("LOCAL_REF_MODEL")
                              or "/home/scott/.cache/huggingface/hub/"
                                 "models--Qwen--Qwen2.5-7B-Instruct")
                specs.append({"name": "qwen2.5-7b", "runner": "local-chat",
                              "model_path": base_local})
                for lb in ("ssd", "nas5-main", "nas5-20260717", "opencode-all",
                           "opencode-portfolio", "hermes-reasoning", "combined"):
                    ap = os.path.join(out_base, f"toolcall-v5-3b-{lb}")
                    if os.path.exists(os.path.join(ap, "adapter_config.json")):
                        specs.append({"name": f"ft-{lb}", "runner": "subagent",
                                      "model_path": ap, "variant": "auto"})
            elif preset == "fleet":
                from src.fleet import pick_model, MODELS
                for m in MODELS:
                    specs.append({"name": m["model"], "runner": "api",
                                  "base_url": m["base_url"], "model": m["model"]})
            if not specs:
                print("[error] bench-matrix needs --specs=<json> or --preset=local-refs|fleet")
                return 2
            rocm = _detect_rocm()
            print(f"[bench-matrix] {len(specs)} specs, {len(tasks)} tasks, rocm={rocm}")
            matrix = bench_matrix(tasks, specs, rocm=rocm)
            print(format_bench_matrix(matrix))
            if "--report" in argv:
                rep = os.path.join(os.path.dirname(__file__), "..", "eval",
                                   "bench-matrix.md")
                Path(rep).write_text(format_bench_matrix(matrix))
                print(f"[bench-matrix] written -> {rep}")
            return 0
        if cmd == "all":
            from src.extract_opencode import main as run_extract
            from src.extract_hermes import main as run_hermes
            from src.clean import main as run_clean
            from src.format_dataset import main as run_format
            cfg.ensure_dirs()
            run_extract(cfg)
            run_hermes(cfg)
            run_clean(cfg)
            for s in ("hermes", "opencode", None):
                run_format(cfg, source=s)
            print("[all] extraction -> cleaning -> formatting complete. Run `train` on a GPU machine.")
            return 0
    except TrainError as e:
        print(f"[error] {e}")
        return 2
    print(__doc__)
    print("Commands: extract | hermes | clean | format | combine | train | eval | eval-all | eval-split | probe | best | sanity | merge | report | compare | bench | bench-matrix | all")
    print("Flags:    --source=hermes|opencode  --label=<name>  --all-split  --dry-run  --max-examples=<n>  --frac=<held-out-frac>  --loss-only  --report  --metric=<loss|tool_exact>")
    print("Bench:    --runner=self|subagent|api|hermes|local-chat  --model=<dir>  --tasks=<jsonl>  --fleet [--fleet-hint=]  --base-url=  --api-model=")
    print("Bench-matrix: --specs=<json-list> | --preset=local-refs|fleet   --tasks=<jsonl>  --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
