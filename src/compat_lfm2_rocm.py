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
    """Causal depthwise conv1d. x [B,C,L]; weight [C,K] (depthwise).

    seq_idx is rejected rather than ignored. causal_conv1d uses it to stop
    attention-free leakage across packed sequences; silently dropping it yields
    output identical to the seq_idx=None path, i.e. tokens near a boundary
    convolve against the previous sequence. Raising surfaces the limitation so
    packing can be disabled, instead of training on quietly corrupted batches.
    """
    if seq_idx is not None:
        raise RuntimeError(
            "causal_conv1d fallback does not support packed sequences "
            "(seq_idx was provided); disable packing or install working "
            "causal_conv1d kernels for this device")
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
    # Carry exactly k-1 tokens of history. The k=1 branch used to hand back
    # width 1, which fed straight back in on the next step and made the output
    # (stale_state + x) * w instead of x * w -- wrong by a multiple, silently.
    new_state = concat[:, :, -(k - 1):] if k > 1 else concat[:, :, :0]
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


# ---------------------------------------------------------------------------
# Training-path coverage.
#
# The fast-path fallback above is not enough for training: transformers'
# Lfm2ShortConv.slow_forward (the path taken during SFT) calls self.conv(Bx)
# i.e. plain nn.Conv1d -> MIOpen directly. On gfx1151 MIOpen's find phase can
# also glitch ("Invalid elapsed time ... elapsed <= 0" -> "No suitable
# algorithm"), which killed lfm2.5-1.2b-sft-r1 at step 0 even though a
# standalone probe of the same conv passes — shapes change per batch, so the
# find phase re-runs throughout a run and one glitch kills hours of training.
#
# apply_to_model() therefore wraps every Conv1d in the loaded model:
#   mode "rescue": try native first, fall back to unfold math on MIOpen error
#   mode "force":  never touch MIOpen conv kernels at all (fwd AND bwd safe)
# State-dict keys are untouched (we only shadow instance .forward).
# ---------------------------------------------------------------------------

def _unfold_conv1d(conv, x):
    """Replicates nn.Conv1d without MIOpen, exactly, for stride=dilation=1.

    Three things this had wrong, all of them silent:

    * padding_mode. It used a bare F.pad, which is zero padding, so a conv built
      with padding_mode='reflect'/'replicate'/'circular' returned values that
      differed from nn.Conv1d by ~1.0 absolute with no error raised. This is a
      correctness shim, so silently-wrong forward math is the worst possible
      failure mode; the exact padding mode is now used.
    * grouped convs where in_channels != out_channels. The reshape packed the
      *input* channel axis as (groups, out_channels/groups), which only lines up
      when in==out, i.e. depthwise. For e.g. Conv1d(4, 6, 3, groups=2) it raised
      a confusing "shape '[2,3,3]' is invalid" from inside reshape.
    * asymmetric padding ('same' with an odd kernel). Now mirrors nn.Conv1d's
      own _reversed_padding_repeated_twice instead of assuming symmetry.
    """
    if conv.stride[0] != 1 or conv.dilation[0] != 1:
        raise ValueError("unfold fallback supports stride=dilation=1 only")
    k = conv.kernel_size[0]
    # Use whatever PyTorch itself would pad with, so 'same'/asymmetric cases and
    # non-zero padding modes agree exactly.
    pad = tuple(getattr(conv, "_reversed_padding_repeated_twice",
                        (conv.padding[0], conv.padding[0])))
    pad_mode = {"zeros": "constant", "reflect": "reflect",
                "replicate": "replicate", "circular": "circular"}[conv.padding_mode]
    xp = F.pad(x, pad, mode=pad_mode) if any(pad) else x
    xu = xp.unfold(-1, k, 1)                      # [B, Cin, Lout, K]
    G = conv.groups
    cin = xu.shape[1]
    cout = conv.weight.shape[0]
    if cin % G or cout % G:
        raise ValueError(f"channels {cin}/{cout} not divisible by groups={G}")
    cg_out = cout // G
    cg_in = cin // G
    w = conv.weight.view(G, cg_out, cg_in, k)     # [G, cout/G, cin/G, K]
    batch, length = xu.shape[0], xu.shape[-2]
    # Axis order must mirror w exactly: [B, G, 1, cg_in, L, K] broadcasts against
    # [1, G, cg_out, cg_in, 1, K]. Putting cg_in at index 2 instead pairs input
    # channels with output channels -- which for a grouped conv whose cg_in
    # equals its cg_out still has the right *shape* and silently wrong values.
    xu = xu.reshape(batch, G, 1, cg_in, length, k)
    out = (xu * w.view(1, G, cg_out, cg_in, 1, k)).sum(dim=(-1, -3))
    out = out.reshape(batch, cout, length)
    if conv.bias is not None:
        out = out + conv.bias.view(1, -1, 1)
    return out


def _is_miopen_error(e: Exception) -> bool:
    s = str(e).lower()
    return "miopen" in s or "no suitable algorithm" in s


def _wrap_conv(conv, mode: str):
    import types

    def forced(self, x):
        return _unfold_conv1d(self, x)

    def rescue(self, x):
        try:
            # torch.nn, not a bare `nn`: this module only imports torch and
            # torch.nn.functional, so the previous `nn.Conv1d.forward` raised
            # NameError the first time MIOpen actually failed -- meaning rescue
            # mode, the default, had never once successfully fallen back.
            return torch.nn.Conv1d.forward(self, x)
        except RuntimeError as e:
            if not _is_miopen_error(e):
                raise
            return _unfold_conv1d(self, x)

    conv.forward = types.MethodType(forced if mode == "force" else rescue, conv)


def apply_to_model(model, mode: str | None = None) -> str:
    """Wrap Conv1d forwards on a loaded LFM2 model. Honors
    LFM_CONV_FALLBACK (off|auto|force); auto == rescue mode.
    Returns a short status string."""
    env = os.environ.get("LFM_CONV_FALLBACK", "auto")
    if env == "off" or (mode == "off"):
        return "disabled-by-env"
    m = mode or ("force" if env == "force" else "rescue")
    n = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Conv1d):
            _wrap_conv(mod, m)
            n += 1
    return f"{m}-mode:{n}convs"
