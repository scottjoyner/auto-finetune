"""Load and validate config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Scratch lives on the data mount, not /tmp. /tmp is a separate filesystem that
# is cleared on reboot, so generated shell functions and deploy packages written
# there did not survive. This is the same path the docs and cron already export
# as TMPDIR.
SCRATCH_ROOT = "/media/scott/data/finetune-staging/tmp"


def scratch_dir(*parts: str, create: bool = True) -> str:
    """Return a writable scratch directory, preferring TMPDIR.

    Resolution order: $TMPDIR, then the staging tmp on the data mount, then
    tempfile's default. The last fallback matters because this module is also
    imported by subagent_2b_agent.py, which is deployed to the destroyer node,
    where the data mount does not exist -- so this must never raise.
    """
    import tempfile

    candidates = [os.environ.get("TMPDIR"), SCRATCH_ROOT,
                  tempfile.gettempdir(), "/tmp"]
    for base in candidates:
        if not base:
            continue
        path = os.path.join(base, *parts) if parts else base
        if not create:
            return path
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".writable")
            with open(probe, "w"):
                pass
            os.remove(probe)
            return path
        except OSError:
            continue
    # Nothing was writable; hand back the last resort without creating it.
    return os.path.join(tempfile.gettempdir(), *parts) if parts else tempfile.gettempdir()


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    # convenience accessors
    def get(self, *keys, default=None):
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def paths(self) -> dict:
        p = self.get("paths", default={}) or {}
        root = project_root()
        return {k: v if os.path.isabs(str(v)) else os.path.join(root, v)
                for k, v in p.items()}

    def path(self, name: str) -> str:
        return self.paths.get(name, os.path.join(project_root(), "data", name))

    def ensure_dirs(self) -> None:
        for p in self.paths.values():
            os.makedirs(p, exist_ok=True)


_DEFAULTS = {
    "sources": {"opencode": {"db_path": "", "extra_dbs": []}, "hermes": {"dir": "", "enabled": False}},
    "extract": {"include_agents": [], "exclude_agents": [], "min_messages": 2, "skip_corrupt_rows": True},
    "clean": {"redact_secrets": True, "drop_empty_turns": True, "max_chars_per_message": 32000,
              "keep_reasoning_as_context": False, "dedupe": True},
    "format": {"template": "chatml",
               "system_prompt": "You are a helpful coding assistant.",
               "max_turns_per_example": 0},
    "train": {"model_name": "unsloth/Qwen3-8B-Instruct", "lora_r": 32, "lora_alpha": 64,
              "lora_dropout": 0.0, "max_seq_length": 8192, "load_in_4bit": True,
              "learning_rate": 2e-4, "num_train_epochs": 3, "per_device_train_batch_size": 2,
              "gradient_accumulation_steps": 8, "warmup_ratio": 0.03, "weight_decay": 0.01,
              "output_dir": "outputs/checkpoints", "hub_model_id": "", "push_to_hub": False},
    "paths": {"raw_dir": "data/raw", "cleaned_dir": "data/cleaned",
              "dataset_dir": "data/datasets", "lock_dir": "data/locks"},
}


def load(path: str | None = None) -> Config:
    path = path or os.path.join(project_root(), "config.yaml")
    user: dict = {}
    if os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
    merged = _deep_merge(_DEFAULTS, user)
    return Config(raw=merged)
