"""Model version registry and tracking.

Maintains a registry of all trained models, their lineage, metrics,
and deployment status. Used by the scheduler and deployer.

Usage:
    python -m src.cli registry-list
    python -m src.cli registry-add --label=<name> --checkpoint=<path>
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.config import Config


REGISTRY_FILE = "model-registry.json"


@dataclass
class ModelEntry:
    """A registered model version."""
    model_id: str  # e.g., "toolcall-v5-3b-combined-v3"
    label: str  # e.g., "combined"
    version: int
    checkpoint_path: str
    created_at: float
    # Training metrics
    train_loss: float | None = None
    eval_loss: float | None = None
    tool_exact_match: float | None = None
    # Dataset info
    dataset_label: str | None = None
    dataset_size: int | None = None
    # Lineage
    base_model: str | None = None
    parent_model: str | None = None  # for iterative finetuning
    # Status
    status: str = "trained"  # trained, deployed, active, failed, archived
    deployed_to: list[str] | None = None
    # Artifacts
    adapter_path: str | None = None
    merged_path: str | None = None
    report_path: str | None = None


class ModelRegistry:
    """Registry for tracking model versions and lineage."""

    def __init__(self, base_dir: str):
        self.registry_path = os.path.join(base_dir, REGISTRY_FILE)
        self.models: dict[str, ModelEntry] = {}
        self._load()

    def _load(self):
        """Load registry from disk."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                data = json.load(f)
            for k, v in data.items():
                self.models[k] = ModelEntry(**v)

    def _save(self):
        """Save registry to disk."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        data = {k: asdict(v) for k, v in self.models.items()}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        label: str,
        checkpoint_path: str,
        base_model: str | None = None,
        parent_model: str | None = None,
        **kwargs,
    ) -> ModelEntry:
        """Register a new model version."""
        # Find next version number
        existing = [m for m in self.models.values() if m.label == label]
        next_version = max((m.version for m in existing), default=0) + 1

        model_id = f"toolcall-v5-3b-{label}-v{next_version}"

        entry = ModelEntry(
            model_id=model_id,
            label=label,
            version=next_version,
            checkpoint_path=checkpoint_path,
            created_at=time.time(),
            base_model=base_model,
            parent_model=parent_model,
            **kwargs,
        )

        self.models[model_id] = entry
        self._save()

        return entry

    def update(self, model_id: str, **kwargs):
        """Update a model entry."""
        if model_id not in self.models:
            raise KeyError(f"model {model_id} not found")

        entry = self.models[model_id]
        for k, v in kwargs.items():
            if hasattr(entry, k):
                setattr(entry, k, v)

        self._save()

    def get(self, model_id: str) -> ModelEntry | None:
        """Get a model by ID."""
        return self.models.get(model_id)

    def get_latest(self, label: str) -> ModelEntry | None:
        """Get the latest version of a model label."""
        candidates = [m for m in self.models.values() if m.label == label]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.version)

    def get_active(self) -> ModelEntry | None:
        """Get the currently active model."""
        for m in self.models.values():
            if m.status == "active":
                return m
        return None

    def set_active(self, model_id: str):
        """Set a model as active, deactivating others."""
        for m in self.models.values():
            if m.status == "active":
                m.status = "deployed"

        if model_id in self.models:
            self.models[model_id].status = "active"
            self._save()

    def list_models(self, label: str | None = None) -> list[ModelEntry]:
        """List models, optionally filtered by label."""
        models = list(self.models.values())
        if label:
            models = [m for m in models if m.label == label]
        return sorted(models, key=lambda m: (-m.created_at, -m.version))

    def get_lineage(self, model_id: str) -> list[ModelEntry]:
        """Get the lineage of a model (parent chain)."""
        lineage = []
        current = self.models.get(model_id)
        while current:
            lineage.append(current)
            current = self.models.get(current.parent_model or "")
        return lineage

    def prune_old_versions(self, label: str, keep: int = 3):
        """Archive old versions, keeping only the latest N."""
        candidates = sorted(
            [m for m in self.models.values() if m.label == label],
            key=lambda m: -m.version,
        )

        for m in candidates[keep:]:
            if m.status not in ("active", "deployed"):
                m.status = "archived"
                self._save()


def main(cfg: Config, argv: list[str]) -> int:
    """CLI handler for registry commands."""
    cmd = argv[1] if len(argv) > 1 else "registry-list"

    registry_dir = os.path.join(cfg.path("analysis_dir"), "registry")
    registry = ModelRegistry(registry_dir)

    if cmd == "registry-list":
        label = None
        for arg in argv:
            if arg.startswith("--label="):
                label = arg.split("=", 1)[1]

        models = registry.list_models(label)
        if not models:
            print("[registry] no models registered")
            return 0

        print(f"[registry] {len(models)} models:")
        for m in models:
            size_mb = 0
            if os.path.exists(m.checkpoint_path):
                size_mb = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fs in os.walk(m.checkpoint_path)
                    for f in fs
                ) / (1024 * 1024)

            status = f"[{m.status}]"
            metrics = ""
            if m.eval_loss is not None:
                metrics += f" loss={m.eval_loss:.4f}"
            if m.tool_exact_match is not None:
                metrics += f" tool={m.tool_exact_match:.3f}"

            print(f"  {m.model_id} {status} {size_mb:.0f}MB{metrics}")
            print(f"    created={m.created_at} checkpoint={m.checkpoint_path}")
        return 0

    if cmd == "registry-add":
        label = None
        checkpoint = None
        base_model = None
        for arg in argv:
            if arg.startswith("--label="):
                label = arg.split("=", 1)[1]
            elif arg.startswith("--checkpoint="):
                checkpoint = arg.split("=", 1)[1]
            elif arg.startswith("--base="):
                base_model = arg.split("=", 1)[1]

        if not label or not checkpoint:
            print("[error] registry-add requires --label=<name> --checkpoint=<path>")
            return 2

        entry = registry.register(label, checkpoint, base_model=base_model)
        print(f"[registry] registered {entry.model_id}")
        return 0

    print("Commands: registry-list [--label=<name>]")
    print("          registry-add --label=<name> --checkpoint=<path> [--base=<model>]")
    return 0
