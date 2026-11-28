"""Pure, deterministic coverage for src.cli dispatch + argument wiring.

Every command handler reached here is a stubbed callable injected into
``sys.modules`` (or a real function replaced with a lambda). Nothing in this
file loads a model, touches a GPU, opens a network/DB connection, or spawns a
subprocess. Commands that would require those side effects are deliberately
*not* executed; their boundary is reported as NO DATA in the accompanying
report.

The focus is the ~200 statements of dispatch/spec glue in ``cli._dispatch``
and the resource-lease wrapper in ``cli.main``: unknown commands, missing
required args, correct dispatch target, and exact kwargs forwarded to stubs.
"""
from __future__ import annotations

import contextlib
import sys
import types

import pytest

import src.cli as cli
from conftest import make_cfg
from src.config import Config


# --------------------------------------------------------------------------- #
# helpers / fixtures
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def stub_module(name: str, **attrs):
    """Inject a fake module into sys.modules for the duration of a test.

    ``cli._dispatch`` imports handlers lazily with ``from src.X import main``;
    seeding sys.modules lets us exercise the dispatch line without importing
    (or running) the real handler module.
    """
    fake = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(fake, key, value)
    original = sys.modules.get(name)
    sys.modules[name] = fake
    try:
        yield fake
    finally:
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture
def cfg():
    """A Config whose directory-creating method is a no-op (no real FS writes)."""
    c = make_cfg()
    c.ensure_dirs = lambda *a, **k: None
    return c


# Commands dispatching as ``run(argv[2:])`` (handler takes no cfg).
PASS_ARGV = {
    "neo4j-watch": "src.neo4j_watcher",
    "neo4j-cdc": "src.neo4j_cdc",
    "realtime-pipeline": "src.realtime_pipeline",
    "model-registry": "src.model_registry",
    "feedback": "src.feedback_loop",
    "freshness": "src.data_freshness",
    "alerts": "src.alerting",
    "autoscale": "src.auto_scaler",
    "fuse": "src.multi_source_fusion",
    "auto-promote": "src.auto_promote",
    "export": "src.model_export",
    "validate-config": "src.config_validator",
    "scheduled-retrain": "src.scheduled_retrain",
    "quality": "src.data_quality",
    "pipeline-health": "src.pipeline_health",
    "subagent-harness": "src.subagent_harness",
}

# Commands dispatching as ``run(cfg, argv)`` (handler reads argv[1]).
PASS_CFG_ARGV = {
    "deploy": "src.deploy",
    "deploy-status": "src.deploy",
    "rollback": "src.deploy",
    "multi-deploy-status": "src.deploy",
    "discover-nodes": "src.deploy",
    "registry-list": "src.registry",
    "registry-add": "src.registry",
    "scheduler-status": "src.scheduler",
    "scheduler-run": "src.scheduler",
    "scheduler-loop": "src.scheduler",
    "notify": "src.notify",
    "notify-history": "src.notify",
    "metrics-record": "src.metrics",
    "metrics-compare": "src.metrics",
    "metrics-regression": "src.metrics",
    "metrics-history": "src.metrics",
    "metrics-summary": "src.metrics",
    "quantize": "src.quantize",
    "quantize-status": "src.quantize",
    "dataset-version-create": "src.dataset_version",
    "dataset-version-list": "src.dataset_version",
    "dataset-version-diff": "src.dataset_version",
    "dataset-version-restore": "src.dataset_version",
    "cost-record": "src.cost",
    "cost-summary": "src.cost",
    "cost-history": "src.cost",
    "binarize": "src.binarize",
    "audit": "src.audit",
    "verify-gap": "src.verify_gap",
}


# --------------------------------------------------------------------------- #
# pure helper edge cases (no side effects)
# --------------------------------------------------------------------------- #
def test_parse_flags_edge_cases():
    # First matching flag wins; empty values are preserved.
    assert cli._parse_source(["x", "--source=hermes", "--source=opencode"]) == "hermes"
    assert cli._parse_label(["x", "--label="]) == ""
    assert cli._parse_int_flag(["--cap=0", "--cap=9"], "--cap") == 0
    assert cli._parse_int_flag(["--cap=-3"], "--cap") == -3
    assert cli._parse_int_flag(["--cap=1.5"], "--cap") is None
    # Repeated flags keep order and only match the exact prefix.
    assert cli._parse_str_flags(
        ["--dataset=a", "--dataset=b", "--datasetx=no"], "--dataset") == ["a", "b"]
    assert cli._parse_str_flags([], "--dataset") == []
    # Exact-token flag matching only.
    assert cli._has_flag(["--dry-runx"], "--dry-run") is False
    assert cli._has_flag(["--dry-run"], "--dry-run") is True


def test_lmstudio_specs_empty_when_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(tmp_path / "absent"))
    assert cli._lmstudio_specs() == []


def test_lmstudio_specs_builds_from_gguf_tree(monkeypatch, tmp_path):
    root = tmp_path / "models"
    (root / "model-a").mkdir(parents=True)
    (root / "model-b").mkdir()
    (root / "model-a" / " weights.gguf").write_bytes(b"")
    (root / "model-b" / "weights.gguf").write_bytes(b"")
    (root / "ignore.txt").write_text("x")
    monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(root))
    monkeypatch.setenv("LMSTUDIO_URL", "http://example/v1")
    specs = cli._lmstudio_specs()
    assert [s["name"] for s in specs] == ["model-a", "model-b"]
    assert all(s["runner"] == "api" for s in specs)
    assert specs[0]["base_url"] == "http://example/v1"
    assert specs[0]["model"] == "model-a"


# --------------------------------------------------------------------------- #
# dispatch target + wiring: run(argv[2:]) handlers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("command,module_name", sorted(PASS_ARGV.items()))
def test_pass_argv_dispatches_to_module(command, module_name, cfg):
    calls = []

    def fake_main(argv):
        calls.append(argv)
        return 23

    with stub_module(module_name, main=fake_main):
        rc = cli._dispatch(["cli", command, "--foo=bar", "--baz"], cfg=cfg)
    assert rc == 23
    assert calls == [["--foo=bar", "--baz"]]


# --------------------------------------------------------------------------- #
# dispatch target + wiring: run(cfg, argv) handlers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("command,module_name", sorted(PASS_CFG_ARGV.items()))
def test_pass_cfg_argv_dispatches_to_module(command, module_name, cfg):
    calls = []

    def fake_main(cfg_arg, argv):
        calls.append((cfg_arg, argv))
        return 24

    with stub_module(module_name, main=fake_main):
        argv = ["cli", command, "--label=x"]
        rc = cli._dispatch(argv, cfg=cfg)
    assert rc == 24
    assert len(calls) == 1
    assert calls[0][0] is cfg
    assert calls[0][1] == argv


# --------------------------------------------------------------------------- #
# dispatch target + wiring: run(cfg, **kwargs) handlers
# --------------------------------------------------------------------------- #
def test_extract_dispatch_kwargs(cfg, capsys):
    calls = []
    with stub_module("src.extract_opencode", main=lambda *a, **k: calls.append((a, k)) or 5):
        rc = cli._dispatch(
            ["cli", "extract", "--label=ssd", "--project=portfolio"], cfg=cfg)
        rc2 = cli._dispatch(
            ["cli", "extract", "--label=x", "project=portfolio"], cfg=cfg)
    assert rc == 0 and rc2 == 0
    # Dispatch calls _parse_str_flag(argv, "project") with no leading dashes,
    # so a conventional `--project=` token is NOT matched (project stays None);
    # only a bare `project=` token yields a value. This pins current behavior.
    assert calls == [
        ((cfg,), {"label": "ssd", "project": None}),
        ((cfg,), {"label": "x", "project": "portfolio"}),
    ]
    assert capsys.readouterr().out.count("[extract] wrote 5 sessions") == 2


def test_hermes_dispatch_kwargs(cfg):
    calls = []
    with stub_module("src.extract_hermes", main=lambda *a, **k: calls.append((a, k)) or 7):
        rc = cli._dispatch(["cli", "hermes"], cfg=cfg)
    assert rc == 0
    assert calls == [((cfg,), {})]


def test_neo4j_traces_incremental_flag(cfg):
    calls = []
    with stub_module("src.extract_neo4j_traces", main=lambda *a, **k: calls.append(k) or 2):
        assert cli._dispatch(["cli", "neo4j-traces", "--incremental"], cfg=cfg) == 0
        assert cli._dispatch(["cli", "neo4j-traces"], cfg=cfg) == 0
    assert calls == [{"incremental": True}, {"incremental": False}]


def test_clean_dispatch_kwargs(cfg):
    calls = []
    with stub_module("src.clean", main=lambda *a, **k: calls.append(k) or 3):
        rc = cli._dispatch(
            ["cli", "clean", "--label=ssd", "--keep-reasoning"], cfg=cfg)
    assert rc == 0
    assert calls == [{"label": "ssd", "keep_reasoning": True}]


def test_train_dispatch_kwargs_and_rc(cfg):
    calls = []

    def fake_train(cfg_arg, **kwargs):
        calls.append(kwargs)
        return 91

    with stub_module("src.train", main=fake_train):
        rc = cli._dispatch(
            ["cli", "train", "--label=ssd", "--source=opencode",
             "--max-examples=100", "--dry-run"], cfg=cfg)
    assert rc == 91
    assert calls == [{
        "dry_run": True, "source": "opencode", "label": "ssd", "max_examples": 100}]


def test_dedup_dispatch_threshold(cfg):
    calls = []
    with stub_module("src.dedup", main=lambda *a, **k: calls.append(k) or 42):
        rc = cli._dispatch(["cli", "dedup", "--threshold=0.75"], cfg=cfg)
    assert rc == 42
    assert calls == [{"label": None, "threshold": 0.75}]


def test_profile_dispatch_kwargs(cfg):
    calls = []
    with stub_module("src.profile", main=lambda *a, **k: calls.append(k) or 43):
        rc = cli._dispatch(["cli", "profile", "--label=ssd", "--out=/tmp/o"], cfg=cfg)
    assert rc == 43
    assert calls == [{"label": "ssd", "out_dir": "/tmp/o"}]


def test_pretokenize_dispatch_defaults(cfg):
    calls = []
    with stub_module("src.pretokenize", main=lambda *a, **k: calls.append(k) or 44):
        rc = cli._dispatch(["cli", "pretokenize"], cfg=cfg)
    assert rc == 44
    assert calls == [{"label": None, "model": None, "max_length": 2048}]


def test_harvest_status_dispatch(cfg):
    calls = []
    with stub_module("src.harvest", main=lambda *a, **k: calls.append(a) or 45):
        rc = cli._dispatch(["cli", "harvest-status"], cfg=cfg)
    assert rc == 45
    assert len(calls) == 1 and calls[0][0] is cfg


def test_auto_balance_dispatch_kwargs(cfg):
    calls = []
    with stub_module("src.auto_balance", main=lambda *a, **k: calls.append(k) or 46):
        rc = cli._dispatch(
            ["cli", "auto-balance", "--cap=200", "--out=/tmp/b"], cfg=cfg)
    assert rc == 46
    assert calls == [{"label": None, "cap": 200, "out_dir": "/tmp/b"}]


# --------------------------------------------------------------------------- #
# exception -> exit-code mapping (pure stubs raising)
# --------------------------------------------------------------------------- #
def test_merge_preflight_error_maps_to_78(cfg, capsys):
    from src.merge import MergePreflightError

    def raise_mpe(*a, **k):
        raise MergePreflightError("preflight refused")

    with stub_module("src.train", main=raise_mpe):
        rc = cli._dispatch(["cli", "train"], cfg=cfg)
    assert rc == 78
    assert "preflight refused" in capsys.readouterr().out


def test_train_error_maps_to_2(cfg, capsys):
    from src.train import TrainError

    def raise_te(*a, **k):
        raise TrainError("no train.jsonl")

    with stub_module("src.train", main=raise_te):
        rc = cli._dispatch(["cli", "train"], cfg=cfg)
    assert rc == 2
    assert "no train.jsonl" in capsys.readouterr().out


def test_value_error_maps_to_2(cfg, capsys):
    def raise_ve(*a, **k):
        raise ValueError("bad int")

    with stub_module("src.dedup", main=raise_ve):
        rc = cli._dispatch(["cli", "dedup"], cfg=cfg)
    assert rc == 2
    assert "bad int" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# edge / failure paths that fail before any side effect
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("argv", [
    ["cli", "eval"],
    ["cli", "eval-split"],
    ["cli", "merge"],
    ["cli", "probe"],
    ["cli", "dpo"],                       # missing --model
    ["cli", "bench-compare"],             # missing --base/--adapter
    ["cli", "manifest"],                  # missing --out
    ["cli", "bench", "--runner=self"],    # missing --model
    ["cli", "bench", "--runner=api"],     # missing --base-url
])
def test_missing_required_args_return_2(argv, cfg):
    assert cli._dispatch(argv, cfg=cfg) == 2


def test_unknown_command_falls_through_to_help(cfg, capsys):
    assert cli._dispatch(["cli", "totally-unknown"], cfg=cfg) == 0
    assert "Commands:" in capsys.readouterr().out


def test_strata_missing_bucket_map_returns_2(cfg, monkeypatch, capsys):
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)
    assert cli._dispatch(["cli", "strata"], cfg=cfg) == 2
    assert "bucket map not found" in capsys.readouterr().out


def test_strata_empty_holdout_returns_2(cfg, monkeypatch, capsys):
    # The "no benchmark sessions" guard lives in `strata`. Stub both the
    # benchmark-id reader and the bucket-map file so no real file is read.
    class FakePath:
        def __init__(self, path):
            pass

        def read_text(self):
            return "{}"

    with stub_module("src.analyze", benchmark_session_ids=lambda path: set()):
        monkeypatch.setattr(cli.os.path, "exists", lambda p: True)
        monkeypatch.setattr(cli, "Path", FakePath)
        rc = cli._dispatch(["cli", "strata", "--holdout=x"], cfg=cfg)
    assert rc == 2
    assert "no benchmark sessions recovered" in capsys.readouterr().out


def test_bench_matrix_lmstudio_no_gguf_returns_2(cfg, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_lmstudio_specs", lambda: [])
    assert cli._dispatch(["cli", "bench-matrix", "--preset=lmstudio"], cfg=cfg) == 2
    assert "no *.gguf found" in capsys.readouterr().out


def test_bench_matrix_fast_no_refs_returns_2(cfg, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_local_ref_specs", lambda: [])
    monkeypatch.setattr(cli, "_lmstudio_specs", lambda: [])
    import src.fleet as fleet
    monkeypatch.setattr(fleet, "list_models", lambda: [])
    assert cli._dispatch(["cli", "bench-matrix", "--preset=fast"], cfg=cfg) == 2
    assert "no references found" in capsys.readouterr().out


def test_bench_matrix_all_no_refs_returns_2(cfg, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_local_ref_specs", lambda: [])
    monkeypatch.setattr(cli, "_lmstudio_specs", lambda: [])
    import src.fleet as fleet
    monkeypatch.setattr(fleet, "list_models", lambda: [])
    assert cli._dispatch(["cli", "bench-matrix", "--preset=all"], cfg=cfg) == 2
    assert "no local / lmstudio / fleet references found" in capsys.readouterr().out


def test_bench_matrix_no_specs_returns_2(cfg):
    assert cli._dispatch(["cli", "bench-matrix"], cfg=cfg) == 2


# --------------------------------------------------------------------------- #
# main() lease/admission wiring (all collaborators stubbed)
# --------------------------------------------------------------------------- #
@pytest.fixture
def lease_env(monkeypatch):
    """Neutralize every real side effect in cli.main."""
    import contextlib as _ctx
    import src.locking as locking
    import src.training_admission as admission

    c = make_cfg()
    monkeypatch.setattr(cli, "load", lambda *a, **k: c)
    monkeypatch.setattr(locking, "command_leases", lambda *a, **k: _ctx.nullcontext())
    monkeypatch.setattr(locking, "unmanaged_training_processes", lambda *a, **k: [])
    monkeypatch.setattr(locking, "command_resources", lambda *a, **k: [])
    monkeypatch.setattr(admission, "read_live",
                        lambda: admission.Decision(True, "synthetic_test"))
    return c


def test_main_forwards_argv_and_cfg_to_dispatch(lease_env, monkeypatch):
    captured = {}

    def fake_dispatch(argv, cfg=None):
        captured["argv"] = argv
        captured["cfg"] = cfg
        return 17

    monkeypatch.setattr(cli, "_dispatch", fake_dispatch)
    argv = ["cli", "feedback", "--label=x"]
    assert cli.main(argv) == 17
    assert captured == {"argv": argv, "cfg": lease_env}


def test_main_runs_real_dispatch_through_stubbed_handler(lease_env):
    with stub_module("src.feedback_loop", main=lambda argv: 31):
        assert cli.main(["cli", "feedback", "--label=x"]) == 31


def test_main_unknown_command_returns_help(lease_env, capsys):
    assert cli.main(["cli", "bogus"]) == 0
    assert "Commands:" in capsys.readouterr().out


def test_main_train_admission_hold_returns_78(monkeypatch, capsys):
    import src.training_admission as admission
    monkeypatch.setattr(admission, "read_live",
                        lambda: admission.Decision(False, "HOLD resource"))
    assert cli.main(["cli", "train"]) == 78
    assert "HOLD resource" in capsys.readouterr().out


def test_main_dry_run_skips_admission_check(lease_env, monkeypatch):
    import src.training_admission as admission

    def must_not_run():
        raise AssertionError("admission must be skipped for --dry-run")

    monkeypatch.setattr(admission, "read_live", must_not_run)
    monkeypatch.setattr(cli, "_dispatch", lambda argv, cfg=None: 0)
    assert cli.main(["cli", "train", "--dry-run"]) == 0


def test_main_coordination_status(lease_env, monkeypatch, capsys):
    import src.locking as locking
    monkeypatch.setattr(locking, "lock_dir", lambda cfg: "/tmp/fake-locks")
    monkeypatch.setattr(locking, "active_owner_records", lambda d: [{"agent": "a"}])
    monkeypatch.setattr(locking, "unmanaged_training_processes", lambda d: [])
    assert cli.main(["cli", "coordination-status"]) == 0
    out = capsys.readouterr().out
    assert "lock_dir=/tmp/fake-locks" in out
    assert "unmanaged_trainers" in out


def test_main_resource_busy_returns_75(lease_env, monkeypatch, capsys):
    from src.locking import ResourceBusy
    import src.locking as locking

    def busy(*a, **k):
        raise ResourceBusy("gpu held")

    monkeypatch.setattr(locking, "command_leases", busy)
    assert cli.main(["cli", "feedback"]) == 75
    assert "gpu held" in capsys.readouterr().out


def test_main_unmanaged_trainer_guard_returns_75(lease_env, monkeypatch, cfg, capsys):
    from types import SimpleNamespace
    import src.locking as locking

    monkeypatch.delenv("AF_SKIP_UNMANAGED_CHECK", raising=False)
    monkeypatch.setattr(locking, "command_resources",
                        lambda *a, **k: [SimpleNamespace(name="gpu", shared=False)])
    monkeypatch.setattr(locking, "unmanaged_training_processes", lambda d: ["pid:123"])
    assert cli.main(["cli", "feedback"]) == 75
    assert "unmanaged training process" in capsys.readouterr().out


def test_main_config_is_config_instance(lease_env):
    # Sanity: the stubbed loader supplies a real Config to the dispatch layer.
    assert isinstance(lease_env, Config)
