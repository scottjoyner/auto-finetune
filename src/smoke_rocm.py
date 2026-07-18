"""ROCm / GPU smoke test for the AMD training path.

Run on the AMD (Strix Halo / 890M) box AFTER installing a ROCm torch build:

    pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
    pip install peft trl transformers datasets bitsandbytes
    export ROCM_PATH=/opt/rocm   # if not auto-detected

Then:  python -m src.smoke_rocm
"""
from __future__ import annotations

import sys
import torch


def main() -> int:
    print("torch:", torch.__version__)
    print("hip build:", getattr(torch.version, "hip", None))
    print("cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("\nNo GPU backend detected. Install a ROCm torch wheel, e.g.:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/rocm6.2")
        return 1

    dev = torch.device("cuda")
    print("device:", torch.cuda.get_device_name(0))

    # 4-bit quant sanity via bitsandbytes (works on ROCm)
    try:
        import bitsandbytes as bnb
        print("bitsandbytes:", bnb.__version__)
        from bitsandbytes.nn import Linear4bit
        print("4-bit Linear available: OK")
    except Exception as e:
        print("bitsandbytes unavailable:", e)

    # Tiny matmul to confirm the iGPU actually computes
    a = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
    b = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
    c = a @ b
    print("matmul OK, result shape:", tuple(c.shape))
    print("\nROCm path looks good. Run: python -m src.cli train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
