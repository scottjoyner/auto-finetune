"""Fail-closed, non-mutating model export while release approval is unavailable.

This module deliberately does not load weights, run converters, create output
directories, or trust self-reported checkpoint provenance. The old merging,
GGUF, and ONNX implementation is available only in historical git revisions.

No self-service override is provided. A separate reviewed release-verifier,
trusted key distribution and rollback workflow must be built before export
can be enabled safely.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

SCHEMA = "auto-finetune.model-export.admission.v1"
DENIAL_REASON = "trusted_release_approval_not_implemented"
DENIAL_EXIT_CODE = 78


@dataclass
class ExportResult:
    """Compatibility result type; all export attempts remain denied."""

    format: str
    output_path: str
    size_bytes: int = 0
    success: bool = False
    error: str | None = DENIAL_REASON
    duration_seconds: float = 0.0


def _hold(fmt: str, output_path: str) -> ExportResult:
    # Never inspect, stat, resolve, or write an output path. Its content is
    # untrusted, and even a path-based check could leak filesystem details.
    return ExportResult(format=fmt, output_path=output_path)


def export_merged(
    checkpoint_path: str,
    output_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> ExportResult:
    """Deny merge: a local adapter or base-model name is not approval."""
    return _hold("merged", output_path)


def export_gguf(
    checkpoint_path: str,
    output_path: str,
    quantize: str = "q8_0",
) -> ExportResult:
    """Deny GGUF conversion without launching external converters."""
    return _hold("gguf-" + str(quantize), output_path)


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> ExportResult:
    """Deny ONNX graph export without importing Torch or Transformers."""
    return _hold("onnx", output_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Model export — release gate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["merged", "gguf", "onnx"], default="merged")
    parser.add_argument("--quantize", default="q8_0")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.parse_args(argv)
    # The command is intentionally blocked BEFORE probing any model or path.
    # Print no checkpoint path, output path, user-supplied model name or key.
    print(json.dumps({
        "schema": SCHEMA,
        "status": "blocked",
        "reason": DENIAL_REASON,
        "model_export_permitted": False,
        "artifact_written": False,
        "weights_loaded": False,
        "converter_launched": False,
    }, sort_keys=True))
    return DENIAL_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
