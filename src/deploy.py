"""Model deployment to inference nodes via shared SSD mount.

Handles copying merged models to inference nodes, symlink rotation,
health checks, and rollback.

Usage:
    python -m src.cli deploy --label=<name> --target=<node>
    python -m src.cli deploy-status
    python -m src.cli rollback --label=<name>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from src.config import Config


# Default inference node paths (shared SSD mount)
DEFAULT_INFERENCE_BASE = "/media/scott/SSD_4TB/inference"
DEFAULT_MODELS_DIR = "models"
DEFAULT_ACTIVE_LINK = "active"


@dataclass
class DeployedModel:
    """A deployed model on an inference node."""
    label: str
    source_path: str
    deploy_path: str
    deployed_at: float
    version: int
    size_bytes: int
    status: str  # "active", "standby", "failed", "rolled_back"
    health_check_passed: bool


@dataclass
class DeployResult:
    """Result of a deployment operation."""
    success: bool
    label: str
    target: str
    deploy_path: str
    message: str
    duration_seconds: float


def _get_model_size(path: str) -> int:
    """Calculate total size of a model directory."""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
    return total


def _health_check(model_path: str, timeout: int = 30) -> bool:
    """Quick health check: load tokenizer and verify model files exist."""
    try:
        # Check critical files exist
        required = ["config.json", "tokenizer.json"]
        for f in required:
            if not os.path.exists(os.path.join(model_path, f)):
                return False

        # Quick tokenizer load test
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        test = tok("Hello world", return_tensors="pt")
        if not test.get("input_ids") is not None:
            return False

        return True
    except Exception:
        return False


def _symlink_rotate(target_dir: str, new_model_path: str) -> str:
    """Atomically rotate the 'active' symlink to point to new model.

    Returns the path of the previously active model (for rollback).
    """
    active_link = os.path.join(target_dir, DEFAULT_ACTIVE_LINK)
    prev_target = None

    # Read current target
    if os.path.islink(active_link):
        prev_target = os.readlink(active_link)

    # Create temp link, then atomically rename
    temp_link = os.path.join(target_dir, f".active-{os.getpid()}")
    if os.path.exists(temp_link):
        os.remove(temp_link)
    os.symlink(new_model_path, temp_link)
    os.rename(temp_link, active_link)

    return prev_target or ""


def deploy_model(
    cfg: Config,
    label: str,
    target: str = "local",
    inference_base: str | None = None,
    health_check: bool = True,
    rollback: bool = False,
) -> DeployResult:
    """Deploy a merged model to an inference node.

    Args:
        cfg: Configuration
        label: Model label (e.g., "combined")
        target: Target node ("local", "nas5", or hostname)
        inference_base: Override inference directory
        health_check: Run health check after deploy
        rollback: If True, rollback to previous version

    Returns:
        DeployResult with status
    """
    start = time.time()

    # Resolve paths
    out_base = cfg.get("train", "output_dir",
                       default="/media/scott/data/finetune-staging/outputs/checkpoints")
    source = os.path.join(out_base, f"toolcall-v5-3b-{label}-merged")

    if not os.path.exists(source):
        return DeployResult(
            success=False, label=label, target=target,
            deploy_path="", message=f"source not found: {source}",
            duration_seconds=time.time() - start,
        )

    if inference_base is None:
        inference_base = DEFAULT_INFERENCE_BASE

    # Target path based on node
    if target == "local":
        target_dir = os.path.join(inference_base, DEFAULT_MODELS_DIR)
    else:
        # For remote nodes, assume shared mount
        target_dir = os.path.join(inference_base, target, DEFAULT_MODELS_DIR)

    os.makedirs(target_dir, exist_ok=True)

    # Version tracking
    version_file = os.path.join(target_dir, "versions.json")
    versions = {}
    if os.path.exists(version_file):
        with open(version_file) as f:
            versions = json.load(f)

    current_version = versions.get(label, {}).get("version", 0)
    new_version = current_version + 1

    # Deploy path
    deploy_path = os.path.join(target_dir, f"{label}-v{new_version}")

    # Handle rollback
    if rollback:
        if current_version > 0:
            # Roll back to previous version
            prev_path = os.path.join(target_dir, f"{label}-v{current_version}")
            if os.path.exists(prev_path):
                # Restore symlink
                _symlink_rotate(target_dir, prev_path)
                return DeployResult(
                    success=True, label=label, target=target,
                    deploy_path=prev_path,
                    message=f"rolled back to v{current_version}",
                    duration_seconds=time.time() - start,
                )
        return DeployResult(
            success=False, label=label, target=target,
            deploy_path="", message="nothing to rollback",
            duration_seconds=time.time() - start,
        )

    # Copy model
    print(f"[deploy] copying {source} -> {deploy_path}")
    if os.path.exists(deploy_path):
        shutil.rmtree(deploy_path)
    shutil.copytree(source, deploy_path)

    # Health check
    check_passed = True
    if health_check:
        print("[deploy] running health check...")
        check_passed = _health_check(deploy_path)
        if not check_passed:
            print("[deploy] WARNING: health check failed")

    # Rotate symlink
    prev = _symlink_rotate(target_dir, deploy_path)

    # Update versions
    versions[label] = {
        "version": new_version,
        "deployed_at": time.time(),
        "source": source,
        "deploy_path": deploy_path,
        "prev_path": prev,
        "health_check": check_passed,
        "size_bytes": _get_model_size(deploy_path),
    }
    with open(version_file, "w") as f:
        json.dump(versions, f, indent=2)

    duration = time.time() - start
    status_msg = f"deployed v{new_version}"
    if not check_passed:
        status_msg += " (health check failed)"

    print(f"[deploy] {status_msg} in {duration:.1f}s")

    return DeployResult(
        success=check_passed, label=label, target=target,
        deploy_path=deploy_path, message=status_msg,
        duration_seconds=duration,
    )


def get_deployed_models(cfg: Config, inference_base: str | None = None) -> list[DeployedModel]:
    """List all deployed models."""
    if inference_base is None:
        inference_base = DEFAULT_INFERENCE_BASE

    target_dir = os.path.join(inference_base, DEFAULT_MODELS_DIR)
    version_file = os.path.join(target_dir, "versions.json")

    if not os.path.exists(version_file):
        return []

    with open(version_file) as f:
        versions = json.load(f)

    models = []
    active_link = os.path.join(target_dir, DEFAULT_ACTIVE_LINK)
    active_target = ""
    if os.path.islink(active_link):
        active_target = os.readlink(active_link)

    for label, info in versions.items():
        deploy_path = info.get("deploy_path", "")
        is_active = deploy_path == active_target

        models.append(DeployedModel(
            label=label,
            source_path=info.get("source", ""),
            deploy_path=deploy_path,
            deployed_at=info.get("deployed_at", 0),
            version=info.get("version", 0),
            size_bytes=info.get("size_bytes", 0),
            status="active" if is_active else "standby",
            health_check_passed=info.get("health_check", False),
        ))

    return sorted(models, key=lambda m: -m.deployed_at)


def multi_deploy(
    cfg: Config,
    label: str,
    nodes: list[str],
    inference_base: str | None = None,
    health_check: bool = True,
    parallel: bool = True,
    quorum: int = 0,
) -> list[DeployResult]:
    """Deploy a model to multiple inference nodes.

    Args:
        cfg: Configuration
        label: Model label (e.g., "combined")
        nodes: List of target nodes ("local", "nas5", hostnames, or "all")
        inference_base: Override inference directory
        health_check: Run health check after deploy on each node
        parallel: Deploy to all nodes in parallel (True) or sequential (False)
        quorum: Minimum successful deploys required (0 = all nodes)

    Returns:
        List of DeployResult per node
    """
    import concurrent.futures

    if not nodes:
        return [DeployResult(
            success=False, label=label, target="none",
            deploy_path="", message="no nodes specified",
            duration_seconds=0,
        )]

    # Resolve "all" to discovered nodes
    resolved_nodes = []
    for n in nodes:
        if n == "all":
            discovered = discover_nodes(cfg, inference_base)
            resolved_nodes.extend(discovered)
        else:
            resolved_nodes.append(n)

    # Deduplicate while preserving order
    seen = set()
    unique_nodes = []
    for n in resolved_nodes:
        if n not in seen:
            seen.add(n)
            unique_nodes.append(n)

    print(f"[multi-deploy] deploying {label} to {len(unique_nodes)} nodes: {unique_nodes}")
    start = time.time()

    def _deploy_one(node: str) -> DeployResult:
        return deploy_model(cfg, label, target=node,
                           inference_base=inference_base,
                           health_check=health_check)

    results: list[DeployResult]

    if parallel and len(unique_nodes) > 1:
        # Parallel deployment
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(unique_nodes), 4)) as pool:
            futures = {pool.submit(_deploy_one, n): n for n in unique_nodes}
            results = []
            for future in concurrent.futures.as_completed(futures):
                node = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    status = "OK" if result.success else "FAIL"
                    print(f"[multi-deploy] {node}: {status} ({result.duration_seconds:.1f}s)")
                except Exception as e:
                    results.append(DeployResult(
                        success=False, label=label, target=node,
                        deploy_path="", message=str(e),
                        duration_seconds=time.time() - start,
                    ))
                    print(f"[multi-deploy] {node}: ERROR {e}")
    else:
        # Sequential deployment
        results = []
        for node in unique_nodes:
            result = _deploy_one(node)
            results.append(result)
            status = "OK" if result.success else "FAIL"
            print(f"[multi-deploy] {node}: {status} ({result.duration_seconds:.1f}s)")

    # Check quorum
    successful = sum(1 for r in results if r.success)
    required = quorum if quorum > 0 else len(unique_nodes)

    total_duration = time.time() - start
    print(f"[multi-deploy] {successful}/{len(unique_nodes)} nodes succeeded "
          f"(quorum={required}) in {total_duration:.1f}s")

    if successful < required:
        print(f"[multi-deploy] WARNING: quorum not met ({successful} < {required})")

    return results


def discover_nodes(cfg: Config, inference_base: str | None = None) -> list[str]:
    """Discover available inference nodes from the shared mount.

    Looks for directories under the inference base that contain a
    models/ subdirectory with versions.json.
    """
    if inference_base is None:
        inference_base = DEFAULT_INFERENCE_BASE

    nodes = []

    # Check root models dir
    root_models = os.path.join(inference_base, DEFAULT_MODELS_DIR)
    if os.path.isdir(root_models):
        nodes.append("local")

    # Check subdirectories for per-node models
    if os.path.isdir(inference_base):
        for entry in os.listdir(inference_base):
            entry_path = os.path.join(inference_base, entry)
            if os.path.isdir(entry_path):
                models_dir = os.path.join(entry_path, DEFAULT_MODELS_DIR)
                if os.path.isdir(models_dir):
                    nodes.append(entry)

    return sorted(set(nodes))


def multi_rollback(
    cfg: Config,
    label: str,
    nodes: list[str],
    inference_base: str | None = None,
) -> list[DeployResult]:
    """Rollback a model on multiple nodes."""
    import concurrent.futures

    if not nodes:
        nodes = discover_nodes(cfg, inference_base)

    print(f"[multi-rollback] rolling back {label} on {len(nodes)} nodes")

    def _rollback_one(node: str) -> DeployResult:
        return deploy_model(cfg, label, target=node,
                           inference_base=inference_base,
                           rollback=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(nodes), 4)) as pool:
        futures = {pool.submit(_rollback_one, n): n for n in nodes}
        results = []
        for future in concurrent.futures.as_completed(futures):
            node = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "OK" if result.success else "FAIL"
                print(f"[multi-rollback] {node}: {status}")
            except Exception as e:
                results.append(DeployResult(
                    success=False, label=label, target=node,
                    deploy_path="", message=str(e), duration_seconds=0,
                ))
                print(f"[multi-rollback] {node}: ERROR {e}")

    successful = sum(1 for r in results if r.success)
    print(f"[multi-rollback] {successful}/{len(nodes)} nodes rolled back")
    return results


def get_multi_deploy_status(
    cfg: Config,
    nodes: list[str] | None = None,
    inference_base: str | None = None,
) -> dict[str, list[DeployedModel]]:
    """Get deployment status across multiple nodes."""
    if nodes is None:
        nodes = discover_nodes(cfg, inference_base)

    status = {}
    for node in nodes:
        if inference_base is None:
            base = DEFAULT_INFERENCE_BASE
        else:
            base = inference_base

        if node == "local":
            target_dir = os.path.join(base, DEFAULT_MODELS_DIR)
        else:
            target_dir = os.path.join(base, node, DEFAULT_MODELS_DIR)

        version_file = os.path.join(target_dir, "versions.json")
        if not os.path.exists(version_file):
            status[node] = []
            continue

        with open(version_file) as f:
            versions = json.load(f)

        models = []
        active_link = os.path.join(target_dir, DEFAULT_ACTIVE_LINK)
        active_target = ""
        if os.path.islink(active_link):
            active_target = os.readlink(active_link)

        for label, info in versions.items():
            deploy_path = info.get("deploy_path", "")
            is_active = deploy_path == active_target

            models.append(DeployedModel(
                label=label,
                source_path=info.get("source", ""),
                deploy_path=deploy_path,
                deployed_at=info.get("deployed_at", 0),
                version=info.get("version", 0),
                size_bytes=info.get("size_bytes", 0),
                status="active" if is_active else "standby",
                health_check_passed=info.get("health_check", False),
            ))

        status[node] = sorted(models, key=lambda m: -m.deployed_at)

    return status


def main(cfg: Config, argv: list[str]) -> int:
    """CLI handler for deploy commands."""
    cmd = argv[1] if len(argv) > 1 else "deploy-status"

    # Parse args
    label = None
    target = "local"
    nodes = None
    for arg in argv:
        if arg.startswith("--label="):
            label = arg.split("=", 1)[1]
        elif arg.startswith("--target="):
            target = arg.split("=", 1)[1]
        elif arg.startswith("--nodes="):
            nodes = arg.split("=", 1)[1].split(",")
        elif arg.startswith("--quorum="):
            pass  # handled in multi_deploy

    if cmd == "deploy-status":
        models = get_deployed_models(cfg)
        if not models:
            print("[deploy-status] no models deployed")
            return 0

        print("[deploy-status]")
        for m in models:
            size_mb = m.size_bytes / (1024 * 1024)
            print(f"  {m.label} v{m.version} [{m.status}] "
                  f"{size_mb:.0f}MB deployed={m.deployed_at}")
            print(f"    path={m.deploy_path}")
        return 0

    if cmd == "multi-deploy-status":
        status = get_multi_deploy_status(cfg, nodes)
        if not status:
            print("[multi-deploy-status] no nodes found")
            return 0

        print("[multi-deploy-status]")
        for node, models in status.items():
            print(f"  {node}:")
            if not models:
                print(f"    (no models)")
                continue
            for m in models:
                size_mb = m.size_bytes / (1024 * 1024)
                print(f"    {m.label} v{m.version} [{m.status}] {size_mb:.0f}MB")
        return 0

    if cmd == "discover-nodes":
        discovered = discover_nodes(cfg)
        print(f"[discover-nodes] {len(discovered)} nodes: {discovered}")
        return 0

    if cmd == "rollback":
        if not label:
            print("[error] rollback requires --label=<name>")
            return 2
        if nodes:
            results = multi_rollback(cfg, label, nodes)
            return 0 if all(r.success for r in results) else 1
        else:
            result = deploy_model(cfg, label, target=target, rollback=True)
            if result.success:
                print(f"[rollback] {result.message}")
                return 0
            else:
                print(f"[rollback] failed: {result.message}")
                return 1

    if cmd == "deploy":
        if not label:
            print("[error] deploy requires --label=<name>")
            return 2
        if nodes:
            quorum = 0
            for arg in argv:
                if arg.startswith("--quorum="):
                    quorum = int(arg.split("=", 1)[1])
            results = multi_deploy(cfg, label, nodes, quorum=quorum)
            return 0 if all(r.success for r in results) else 1
        else:
            result = deploy_model(cfg, label, target=target)
            if result.success:
                print(f"[deploy] {result.message}")
                return 0
            else:
                print(f"[deploy] failed: {result.message}")
                return 1

    print("Commands:")
    print("  deploy --label=<name> [--target=<node>]")
    print("  deploy --label=<name> --nodes=<n1,n2,...> [--quorum=N]")
    print("  deploy-status")
    print("  multi-deploy-status [--nodes=<n1,n2,...>]")
    print("  discover-nodes")
    print("  rollback --label=<name> [--nodes=<n1,n2,...>]")
    return 0
