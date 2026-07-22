"""Dataset versioning for reproducibility.

Tracks dataset snapshots with hashes, metadata, and lineage so
training runs can be reproduced exactly.

Usage:
    python -m src.cli dataset-version-create --label=<name>
    python -m src.cli dataset-version-list
    python -m src.cli dataset-version-diff --v1=<id> --v2=<id>
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.config import Config


VERSIONS_FILE = "dataset-versions.json"


@dataclass
class DatasetVersion:
    """A versioned snapshot of a dataset."""
    version_id: str  # e.g., "v20260720-1234"
    label: str
    created_at: float
    # Content hash (SHA256 of concatenated file hashes)
    content_hash: str
    # Stats
    num_examples: int
    total_bytes: int
    num_files: int
    # Source info
    source_dir: str
    # Lineage (which versions were used to create this)
    parent_versions: list[str]
    # Metadata
    description: str | None = None
    tags: list[str] | None = None


class DatasetVersioner:
    """Version datasets for reproducibility."""

    def __init__(self, base_dir: str):
        self.versions_path = os.path.join(base_dir, VERSIONS_FILE)
        self.versions: dict[str, DatasetVersion] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.versions_path):
            with open(self.versions_path) as f:
                data = json.load(f)
            for k, v in data.items():
                self.versions[k] = DatasetVersion(**v)

    def _save(self):
        os.makedirs(os.path.dirname(self.versions_path), exist_ok=True)
        data = {k: asdict(v) for k, v in self.versions.items()}
        with open(self.versions_path, "w") as f:
            json.dump(data, f, indent=2)

    def _hash_file(self, path: str) -> str:
        """Hash a single file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _hash_directory(self, dir_path: str) -> tuple[str, int, int]:
        """Hash all files in a directory.

        Returns:
            (combined_hash, total_bytes, num_files)
        """
        file_hashes = []
        total_bytes = 0
        num_files = 0

        for path in sorted(Path(dir_path).rglob("*")):
            if path.is_file():
                file_hash = self._hash_file(str(path))
                file_hashes.append(f"{path.name}:{file_hash}")
                total_bytes += path.stat().st_size
                num_files += 1

        combined = hashlib.sha256("|".join(file_hashes).encode()).hexdigest()
        return combined, total_bytes, num_files

    def create_version(
        self,
        dataset_dir: str,
        label: str,
        description: str | None = None,
        tags: list[str] | None = None,
        parent_versions: list[str] | None = None,
    ) -> DatasetVersion:
        """Create a new version of a dataset."""
        # Generate version ID with counter to avoid collisions
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        counter = 0
        version_id = f"v{timestamp}"
        while version_id in self.versions:
            counter += 1
            version_id = f"v{timestamp}-{counter}"

        # Hash the dataset
        content_hash, total_bytes, num_files = self._hash_directory(dataset_dir)

        # Count examples (JSONL files)
        num_examples = 0
        for path in Path(dataset_dir).rglob("*.jsonl"):
            with open(path) as f:
                num_examples += sum(1 for line in f if line.strip())

        version = DatasetVersion(
            version_id=version_id,
            label=label,
            created_at=time.time(),
            content_hash=content_hash,
            num_examples=num_examples,
            total_bytes=total_bytes,
            num_files=num_files,
            source_dir=dataset_dir,
            parent_versions=parent_versions or [],
            description=description,
            tags=tags,
        )

        self.versions[version_id] = version
        self._save()

        return version

    def get_version(self, version_id: str) -> DatasetVersion | None:
        return self.versions.get(version_id)

    def get_latest(self, label: str | None = None) -> DatasetVersion | None:
        candidates = list(self.versions.values())
        if label:
            candidates = [v for v in candidates if v.label == label]
        if not candidates:
            return None
        return max(candidates, key=lambda v: v.created_at)

    def list_versions(self, label: str | None = None) -> list[DatasetVersion]:
        versions = list(self.versions.values())
        if label:
            versions = [v for v in versions if v.label == label]
        return sorted(versions, key=lambda v: -v.created_at)

    def diff(self, v1_id: str, v2_id: str) -> dict:
        """Compare two dataset versions."""
        v1 = self.versions.get(v1_id)
        v2 = self.versions.get(v2_id)

        if not v1 or not v2:
            return {"error": "version not found"}

        return {
            "v1": v1.version_id,
            "v2": v2.version_id,
            "same_content": v1.content_hash == v2.content_hash,
            "num_examples": {"v1": v1.num_examples, "v2": v2.num_examples,
                            "delta": v2.num_examples - v1.num_examples},
            "total_bytes": {"v1": v1.total_bytes, "v2": v2.total_bytes,
                           "delta": v2.total_bytes - v1.total_bytes},
            "created_at": {"v1": v1.created_at, "v2": v2.created_at},
        }

    def restore(self, version_id: str, target_dir: str) -> bool:
        """Restore a dataset version to a target directory."""
        version = self.versions.get(version_id)
        if not version:
            return False

        if not os.path.exists(version.source_dir):
            return False

        os.makedirs(target_dir, exist_ok=True)
        for path in Path(version.source_dir).rglob("*"):
            if path.is_file():
                rel = path.relative_to(version.source_dir)
                dst = os.path.join(target_dir, str(rel))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(str(path), dst)

        return True


def main(cfg: Config, argv: list[str]) -> int:
    """CLI handler for dataset-version commands."""
    cmd = argv[1] if len(argv) > 1 else "dataset-version-list"

    dataset_dir = cfg.path("dataset_dir")
    analysis_dir = cfg.path("analysis_dir")
    versions_dir = os.path.join(analysis_dir, "dataset-versions")
    versioner = DatasetVersioner(versions_dir)

    label = None
    description = None
    v1_id = None
    v2_id = None

    for arg in argv:
        if arg.startswith("--label="):
            label = arg.split("=", 1)[1]
        elif arg.startswith("--description="):
            description = arg.split("=", 1)[1]
        elif arg.startswith("--v1="):
            v1_id = arg.split("=", 1)[1]
        elif arg.startswith("--v2="):
            v2_id = arg.split("=", 1)[1]

    if cmd == "dataset-version-create":
        if not label:
            label = "default"

        version = versioner.create_version(
            dataset_dir, label, description=description,
        )
        print(f"[dataset-version] created {version.version_id}")
        print(f"  label: {version.label}")
        print(f"  examples: {version.num_examples}")
        print(f"  hash: {version.content_hash[:16]}...")
        return 0

    if cmd == "dataset-version-list":
        versions = versioner.list_versions(label)
        if not versions:
            print("[dataset-version] no versions")
            return 0

        print(f"[dataset-version] {len(versions)} versions:")
        for v in versions:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(v.created_at))
            print(f"  {v.version_id} [{v.label}] {v.num_examples} examples "
                  f"{v.total_bytes / 1024 / 1024:.1f}MB [{ts}]")
            if v.description:
                print(f"    {v.description}")
        return 0

    if cmd == "dataset-version-diff":
        if not v1_id or not v2_id:
            print("[error] dataset-version-diff requires --v1=<id> --v2=<id>")
            return 2

        diff = versioner.diff(v1_id, v2_id)
        if "error" in diff:
            print(f"[error] {diff['error']}")
            return 1

        print(f"[dataset-version-diff] {diff['v1']} vs {diff['v2']}")
        print(f"  same_content: {diff['same_content']}")
        print(f"  examples: {diff['num_examples']['v1']} -> {diff['num_examples']['v2']} "
              f"({diff['num_examples']['delta']:+d})")
        print(f"  size: {diff['total_bytes']['v1'] / 1024 / 1024:.1f}MB -> "
              f"{diff['total_bytes']['v2'] / 1024 / 1024:.1f}MB")
        return 0

    if cmd == "dataset-version-restore":
        if not v2_id:
            print("[error] dataset-version-restore requires --v2=<id>")
            return 2

        target = os.path.join(dataset_dir, f"restored-{v2_id}")
        if versioner.restore(v2_id, target):
            print(f"[dataset-version] restored {v2_id} to {target}")
            return 0
        else:
            print(f"[dataset-version] failed to restore {v2_id}")
            return 1

    print("Commands:")
    print("  dataset-version-create --label=<name> [--description=<text>]")
    print("  dataset-version-list [--label=<name>]")
    print("  dataset-version-diff --v1=<id> --v2=<id>")
    print("  dataset-version-restore --v2=<id>")
    return 0
