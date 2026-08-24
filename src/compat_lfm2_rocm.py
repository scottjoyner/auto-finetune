"""ROCm gfx1151 compatibility shim for LFM2 short-convolution layers.

The causal-conv1d package ships kernels only for select architectures; on
gfx1151 (Radeon 8050S / 8060S iGPUs) `causal_conv1d_fn` raises
hipErrorInvalidDeviceFunction, which kills LFM2-family training at the first
forward pass.

This module probes the installed kernel; when it cannot run on this device,
it installs a pure-torch fallback (unfold + multiply, no custom kernels)
into BOTH `causal_conv1d_fn`/`causal_conv1d_update` call sites used by
transformers' lfm2 modeling code AND the module-level names referenced by
`Lfm2ShortConv.forward`.

Mathematically identical causal depthwise convolution; slower (memory-bound
elementwise ops), which is acceptable for small-model finetuning.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

PROBE_DONE = False


def _pad_and_unfold(x: torch.Tensor, k: int) -> torch.Tensor:
    xp = F.pad(x, (k - 1, 0))
    return xp.unfold(-1, k, 1)  # [B, C, L, K]


def fallback_causal_conv1d_fn(x, weight, bias=None, activation=None,
                              seq_idx=None, *args, **kwargs):
    """Causal depthwise conv1d. x [B,C,L]; weight [C,K] (depthwise)."""
    k = weight.shape[-1]
    xu = _pad_and_unfold(x, k)                       # [B, C, L, K]
    w = weight.squeeze(1) if weight.dim() == 3 else weight
    out = (xu * w.view(1, -1, 1, k)).sum(-1)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    if activation == "silu":
        out = F.silu(out)
    return out


def fallback_causal_conv1d_update(x, conv_state, weight, bias=None,
                                  activation=None):
    """Single-token decode path: x [B,C]; conv_state [B,C,K-1]."""
    k = weight.shape[-1]
    w = weight.squeeze(1) if weight.dim() == 3 else weight
    concat = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)  # [B,C,K]
    out = (concat * w.view(1, -1, k)).sum(-1)
    if bias is not None:
        out = out + bias.view(1, -1)
    new_state = concat[:, :, -(k - 1):] if k > 1 else concat[:, :, :1]
    if activation == "silu":
        out = F.silu(out)
    return out, new_state


def _probe_works(fn) -> bool:
    try:
        if not torch.cuda.is_available():
            return True  # CPU path: assume fine, avoid blocking
        x = torch.randn(2, 8, 32, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(8, 1, 4, device="cuda", dtype=torch.bfloat16)
        out = fn(x, w.squeeze(-1), None)
        torch.cuda.synchronize()
        return out.shape == (2, 8, 32)
    except Exception:
        return False


def install_if_needed(force: bool | None = None):
    """Probe the real causal_conv1d kernels; install fallbacks when broken.

    force=True installs unconditionally; force=False skips; None = auto-probe.
    """
    global PROBE_DONE
    if PROBE_DONE and force is None:
        return "already"
    PROBE_DONE = True
    try:
        import causal_conv1d as cc  # noqa
        ok = _probe_works(cc.causal_conv1d_fn)
    except Exception:
        ok = False
    if ok and force is not True:
        return "native-kernels-ok"

    import transformers.models.lfm2.modeling_lfm2 as m

    m.causal_conv1d_fn = fallback_causal_conv1d_fn
    m.causal_conv1d_update = fallback_causal_conv1d_update
    return "fallback-installed"


def maybe_install_from_env() -> str:
    """Honors LFM_CONV_FALLBACK=force|auto|off (default auto)."""
    mode = os.environ.get("LFM_CONV_FALLBACK", "auto")
    if mode == "off":
        return "disabled-by-env"
    return install_if_needed(force=(mode == "force"))
