"""Coverage tests for format_dataset filesystem-backed paths (main, combine)."""
from __future__ import annotations

import json

from src import format_dataset as F
from src.config import Config

SESSION = {
    "source": "hermes",
    "messages": [
        {"role": "user", "parts": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "parts": [{"type": "text", "text": "yo"}]},
    ],
}


def _cfg(cleaned: str, ds: str) -> Config:
    return Config({
        "paths": {"cleaned_dir": cleaned, "dataset_dir": ds},
        "format": {"template": "chatml", "system_prompt": "SYS",
                   "max_turns_per_example": 0, "max_chars_per_example": 24000},
    })


def test_main_formats_cleaned_dir(tmp_path):
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    (cleaned / "s1.json").write_text(json.dumps(SESSION))
    ds = tmp_path / "datasets"
    cfg = _cfg(str(cleaned), str(ds))
    total = F.main(cfg, source=None)
    # main writes each output file exactly once; a single source file yields
    # exactly one example in the merged train.jsonl (the old code double-wrote
    # and returned 2 — that was the bug).
    assert total == 1
    out = ds / "train.jsonl"
    assert out.exists()
    ex = json.loads(out.read_text().splitlines()[0])
    assert "messages" in ex


def test_combine_merges_labeled_datasets(tmp_path):
    ds = tmp_path / "datasets"
    ds.mkdir()
    cfg = _cfg(str(tmp_path / "cleaned"), str(ds))
    for i, label in enumerate(("ssd", "nas5-main", "opencode-portfolio")):
        (ds / f"train.{label}.jsonl").write_text(
            json.dumps({"messages": [{"role": "user", "content": f"x{i}"}]}) + "\n")
    total = F.combine(cfg)
    assert total == 3
    combined = ds / "train.combined.jsonl"
    assert combined.exists()
    assert len(combined.read_text().splitlines()) == 3
