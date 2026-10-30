"""Every dispatched command is a synthetic no-op; no real command executes."""
import sys
import types

import pytest
from src import cli
from src.config import Config


def make_stub(monkeypatch,module,called):
    stub=types.ModuleType(module)
    stub.main=lambda *a,**k: called.append((a,k)) or 37
    monkeypatch.setitem(sys.modules,module,stub)


@pytest.mark.parametrize("command,module",[
    ("neo4j-watch","src.neo4j_watcher"),
    ("neo4j-cdc","src.neo4j_cdc"),
    ("realtime-pipeline","src.realtime_pipeline"),
    ("model-registry","src.model_registry"),
    ("feedback","src.feedback_loop"),
    ("freshness","src.data_freshness"),
    ("alerts","src.alerting"),
    ("autoscale","src.auto_scaler"),
    ("fuse","src.multi_source_fusion"),
    ("auto-promote","src.auto_promote"),
    ("export","src.model_export"),
    ("validate-config","src.config_validator"),
    ("scheduled-retrain","src.scheduled_retrain"),
    ("quality","src.data_quality"),
    ("pipeline-health","src.pipeline_health"),
    ("subagent-harness","src.subagent_harness"),
])
def test_delegate_forwards_flags_and_exit_status_only(command,module,monkeypatch):
    calls=[]
    make_stub(monkeypatch,module,calls)
    cfg=Config(raw={})
    argv=["cli",command,"--synthetic=1","--dry-run"]
    assert cli._dispatch(argv,cfg=cfg)==37
    assert calls==[((["--synthetic=1","--dry-run"],),{})]


@pytest.mark.parametrize("command,module",[
    ("registry-list","src.registry"),
    ("registry-add","src.registry"),
    ("scheduler-status","src.scheduler"),
    ("scheduler-run","src.scheduler"),
    ("scheduler-loop","src.scheduler"),
    ("notify","src.notify"),
    ("notify-history","src.notify"),
    ("metrics-record","src.metrics"),
    ("metrics-compare","src.metrics"),
    ("metrics-regression","src.metrics"),
    ("metrics-history","src.metrics"),
    ("metrics-summary","src.metrics"),
    ("quantize","src.quantize"),
    ("quantize-status","src.quantize"),
    ("dataset-version-list","src.dataset_version"),
    ("dataset-version-diff","src.dataset_version"),
    ("cost-history","src.cost"),
    ("binarize","src.binarize"),
    ("audit","src.audit"),
])
def test_full_argv_and_config_are_forwarded_including_command(command,module,monkeypatch):
    calls=[]
    make_stub(monkeypatch,module,calls)
    cfg=Config(raw={})
    argv=["cli",command,"--test-only=yes"]
    assert cli._dispatch(argv,cfg=cfg)==37
    assert calls==[((cfg,argv),{})]


def test_balance_handler_resolves_safe_parameters(monkeypatch):
    calls=[]
    make_stub(monkeypatch,"src.auto_balance",calls)
    cfg=Config(raw={})
    argv=["cli","auto-balance","--label=toy","--cap=33","--out=/tmp/synthetic"]
    assert cli._dispatch(argv,cfg=cfg)==37
    assert calls==[((cfg,),{"label":"toy","cap":33,"out_dir":"/tmp/synthetic"})]
