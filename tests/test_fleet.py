"""Tests for src.fleet (lan fleet-router helper)."""
from __future__ import annotations

import json
from pathlib import Path

from src import fleet as F


def _fake_endpoints(tmp_path: Path) -> str:
    data = {
        "nodes": [
            {"name": "big", "online": True, "base_url": "http://10.0.0.1:1234",
             "available_models": ["qwen3.6-35b-a3b-claude-opus", "tiny-1b"],
             "loaded_models": []},
            {"name": "small", "online": False, "base_url": "http://10.0.0.2:1234",
             "available_models": ["gpt-oss-20b"], "loaded_models": []},
            {"name": "mid", "online": True, "base_url": "http://10.0.0.3:1234",
             "available_models": ["gemma-4-12b", "lfm2-1.2b-tool"], "loaded_models": []},
        ]
    }
    p = tmp_path / "endpoints.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_list_models_only_large(tmp_path):
    ep = _fake_endpoints(tmp_path)
    models = F.list_models(ep, only_large=True)
    names = {m["model"] for m in models}
    # offline node excluded; small models excluded
    assert "qwen3.6-35b-a3b-claude-opus" in names
    assert "gemma-4-12b" in names
    assert "tiny-1b" not in names
    assert "lfm2-1.2b-tool" not in names
    assert all(m["node"] != "small" for m in models)


def test_pick_model_hint(tmp_path):
    ep = _fake_endpoints(tmp_path)
    picked = F.pick_model("gemma", ep)
    assert picked["model"] == "gemma-4-12b"
    picked = F.pick_model(None, ep)
    assert picked is not None
