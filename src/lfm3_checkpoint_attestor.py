"""Offline verification of LiquidAI/LFM3-1.7B BF16 snapshot bytes.

No network, inference, remote code execution, graph access, or weight mutation.
The publisher's pinned repository file hashes must match every local file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import stat

MODEL_ID = "LiquidAI/LFM3-1.7B"
REVISION = "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd"
MODEL_CACHE_DIR = "models--LiquidAI--LFM3-1.7B"
ASSETS = {
    "model.safetensors": (3457635448, "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"),
    "config.json": (1454, "8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d"),
    "configuration_lfm3.py": (6113, "176cced35657aedfea7d7002221360582d1c05fcaca882c21508bc3a73b282e5"),
    "modeling_lfm3.py": (42359, "e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab"),
    "lfm3_conv.py": (8532, "80d4a985382a22774212759248a6b4c737bfbf9c2a47b3dd7f797dda748e4bb4"),
    "tokenizer.json": (17908635, "7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701"),
    "tokenizer_config.json": (1545, "bf028f813efc6137bc07ed045cafe71820df9a5e26a38e39954706ed1e30a457"),
    "special_tokens_map.json": (92, "7946f864cbfb62a1b041bda937244210a1a87f4a6cab3a48b791131b2cd57f88"),
    "generation_config.json": (328, "0095ba7158e938b1e2d3082d412f00d8be85e42cbbd7608b63866f7a081cf270"),
    "chat_template.jinja": (12129, "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818"),
    "LICENSE": (10574, "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c"),
}


class CheckpointRefused(ValueError):
    pass


def verify_file(path: pathlib.Path, expected_size: int, expected_sha: str) -> None:
    """Read once, bounded by the exact upstream size and SHA-256."""
    try:
        st = path.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_size != expected_size:
            raise CheckpointRefused("asset absent or unexpected size")
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                count += len(chunk)
                if count > expected_size:
                    raise CheckpointRefused("asset exceeds pinned byte size")
                digest.update(chunk)
        if count != expected_size or digest.hexdigest() != expected_sha:
            raise CheckpointRefused("asset checksum does not match pinned upstream")
    except OSError as exc:
        raise CheckpointRefused("unreadable pinned model asset") from exc


def verify_snapshot(snapshot: pathlib.Path) -> dict:
    if (snapshot.name != REVISION or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != MODEL_CACHE_DIR):
        raise CheckpointRefused("snapshot is not the approved model/revision")
    cache_base = snapshot.parent.parent.resolve(strict=True)
    snapshot_resolved = snapshot.resolve(strict=True)
    if not snapshot_resolved.is_relative_to(cache_base):
        raise CheckpointRefused("snapshot outside model cache")
    for name, (size, sha256) in ASSETS.items():
        f = snapshot / name
        if not f.resolve(strict=True).is_relative_to(cache_base):
            raise CheckpointRefused("model asset escapes its cache directory")
        verify_file(f, size, sha256)
    return {
        "schema": "lfm3-bf16-offline-assets.v1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "checkpoint_sha256": ASSETS["model.safetensors"][1],
        "config_sha256": ASSETS["config.json"][1],
        "model_code_sha256": ASSETS["modeling_lfm3.py"][1],
        "tokenizer_sha256": ASSETS["tokenizer.json"][1],
        "verified_asset_count": len(ASSETS),
        "pinned_files_verified_from_local_bytes": True,
        "publisher_metadata_authenticated_by_this_tool": False,
        "remote_code_executed": False,
        "model_weights_modified": False,
        "production_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-only", action="store_true", required=True)
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_snapshot(args.snapshot), sort_keys=True))
        return 0
    except (CheckpointRefused, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit("LFM3 checkpoint refused: " + str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
