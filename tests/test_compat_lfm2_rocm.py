"""Tests for the ROCm gfx1151 LFM2 convolution shim.

This module's entire value proposition is that the fallback is mathematically
identical to what it replaces, while running on a device whose native kernels
fail. Every bug found here was silent -- a fallback that returns plausible
numbers which differ from nn.Conv1d is worse than one that crashes -- so the
tests below are mostly equivalence checks against nn.Conv1d itself.

All CPU, no GPU and no MIOpen required.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.compat_lfm2_rocm import (
    _is_miopen_error,
    _pad_and_unfold,
    _probe_works,
    _unfold_conv1d,
    apply_to_model,
    fallback_causal_conv1d_fn,
    fallback_causal_conv1d_update,
    install_if_needed,
    maybe_install_from_env,
)


def _close(a, b, atol=1e-5, rtol=1e-4):
    return torch.allclose(a, b, atol=atol, rtol=rtol)


# --- _unfold_conv1d must equal nn.Conv1d ----------------------------------

@pytest.mark.parametrize("spec", [
    dict(in_channels=8, out_channels=8, kernel_size=4, groups=8, padding=0),   # depthwise
    dict(in_channels=8, out_channels=8, kernel_size=5, groups=8, padding=2),   # depthwise + pad
    dict(in_channels=4, out_channels=4, kernel_size=3, groups=4, padding=1),   # depthwise k=3
    dict(in_channels=4, out_channels=6, kernel_size=3, groups=2, padding=1),   # grouped, in != out
    dict(in_channels=4, out_channels=8, kernel_size=3, groups=4, padding=1),   # grouped, in != out
    dict(in_channels=6, out_channels=6, kernel_size=3, groups=3, padding=0),   # grouped dense
    dict(in_channels=4, out_channels=8, kernel_size=1, groups=1, padding=0),   # pointwise
    dict(in_channels=8, out_channels=8, kernel_size=4, groups=8, padding=3, bias=False),
])
def test_unfold_matches_conv1d(spec):
    torch.manual_seed(0)
    conv = nn.Conv1d(**spec)
    x = torch.randn(2, spec["in_channels"], 17)
    assert _close(_unfold_conv1d(conv, x), conv(x)), (
        f"unfold fallback diverged from nn.Conv1d for {spec}")


@pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
def test_unfold_matches_every_padding_mode(mode):
    # Regression: the fallback used a bare F.pad, which is zero padding, so all
    # three non-zero modes silently returned values ~1.0 away from nn.Conv1d.
    torch.manual_seed(1)
    conv = nn.Conv1d(4, 4, 3, groups=4, padding=1, padding_mode=mode)
    x = torch.randn(2, 4, 16)
    got, want = _unfold_conv1d(conv, x), conv(x)
    assert _close(got, want), f"padding_mode={mode} diverged by {(got-want).abs().max():.3e}"


def test_unfold_matches_asymmetric_same_padding():
    # padding='same' with an odd kernel pads asymmetrically. The old code
    # assumed symmetry via conv.padding[0].
    torch.manual_seed(2)
    conv = nn.Conv1d(4, 4, 5, padding="same")
    x = torch.randn(2, 4, 16)
    assert _close(_unfold_conv1d(conv, x), conv(x))


def test_unfold_preserves_input_length_for_same_padding():
    conv = nn.Conv1d(4, 4, 5, padding="same")
    x = torch.randn(2, 4, 16)
    assert _unfold_conv1d(conv, x).shape == conv(x).shape == (2, 4, 16)


def test_unfold_refuses_unsupported_stride_and_dilation():
    for kwargs in (dict(stride=2), dict(dilation=2)):
        conv = nn.Conv1d(4, 4, 3, groups=4, padding=1, **kwargs)
        with pytest.raises(ValueError, match="stride=dilation=1"):
            _unfold_conv1d(conv, torch.randn(2, 4, 16))


def test_unfold_refuses_indivisible_channels():
    conv = nn.Conv1d(4, 4, 3, groups=4, padding=1)
    conv.groups = 3  # 4 and 4 both indivisible by 3
    with pytest.raises(ValueError, match="not divisible"):
        _unfold_conv1d(conv, torch.randn(2, 4, 16))


def test_unfold_is_differentiable():
    # The shim exists for training; gradients must flow to weight and input.
    conv = nn.Conv1d(4, 4, 3, groups=4, padding=1)
    x = torch.randn(2, 4, 12, requires_grad=True)
    _unfold_conv1d(conv, x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()


# --- causal fallback -------------------------------------------------------

def test_pad_and_unfold_shapes_and_causality():
    x = torch.arange(12, dtype=torch.float32).view(1, 1, 12)
    xu = _pad_and_unfold(x, 4)
    assert xu.shape == (1, 1, 12, 4)
    # The last window must be the 4 most recent values, unpadded at the front.
    assert xu[0, 0, -1].tolist() == [8.0, 9.0, 10.0, 11.0]
    assert xu[0, 0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_causal_fallback_is_causal():
    # Perturbing a later timestep must not change an earlier output.
    torch.manual_seed(3)
    x = torch.randn(1, 2, 16)
    w = torch.randn(2, 4)
    base = fallback_causal_conv1d_fn(x, w)
    x2 = x.clone()
    x2[..., 12:] += 5.0
    out = fallback_causal_conv1d_fn(x2, w)
    assert _close(base[..., :12], out[..., :12])
    assert not _close(base[..., 12:], out[..., 12:])


def test_causal_fallback_applies_bias_and_silu():
    x = torch.randn(1, 3, 8)
    w = torch.randn(3, 1, 4).squeeze(1)
    b = torch.randn(3)
    plain = fallback_causal_conv1d_fn(x, w, b)
    silu = fallback_causal_conv1d_fn(x, w, b, activation="silu")
    assert _close(silu, torch.nn.functional.silu(plain))


def test_causal_fallback_rejects_packed_sequences():
    # Regression: seq_idx was accepted and ignored, producing output identical
    # to the unpacked path -- i.e. cross-sequence contamination.
    x = torch.randn(1, 2, 8)
    w = torch.randn(2, 1, 4).squeeze(1)
    seq = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    with pytest.raises(RuntimeError, match="packed sequences"):
        fallback_causal_conv1d_fn(x, w, None, None, seq)


# --- decode path -----------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_decode_state_carries_k_minus_1(k):
    x = torch.randn(2, 4)
    w = torch.randn(4, 1, k)
    _, state = fallback_causal_conv1d_update(x, torch.zeros(2, 4, k - 1), w)
    assert state.shape == (2, 4, k - 1)


def test_decode_k1_state_is_zero_width():
    # Regression: the k=1 branch returned width 1, which fed back in and made
    # the next output (stale + x) * w instead of x * w.
    x = torch.randn(2, 4)
    w = torch.randn(4, 1, 1)
    b = torch.randn(4)
    _, state = fallback_causal_conv1d_update(x, torch.zeros(2, 4, 0), w, b)
    assert state.shape == (2, 4, 0)


def test_decode_k1_matches_plain_multiply():
    torch.manual_seed(4)
    x = torch.randn(2, 4)
    w = torch.randn(4)
    b = torch.randn(4)
    out, _ = fallback_causal_conv1d_update(x, torch.zeros(2, 4, 0),
                                           w.view(4, 1, 1), b)
    assert _close(out, x * w + b)


def test_decode_streaming_matches_full_forward():
    # Streaming one token at a time must equal the whole-sequence call.
    torch.manual_seed(5)
    B, C, K, L = 2, 3, 4, 7
    x = torch.randn(B, C, L)
    w = torch.randn(C, 1, K)
    b = torch.randn(C)
    full = fallback_causal_conv1d_fn(x, w.squeeze(1), b)

    state = torch.zeros(B, C, K - 1)
    streamed = []
    for t in range(L):
        out, state = fallback_causal_conv1d_update(x[:, :, t], state, w, b)
        streamed.append(out)
    assert _close(full, torch.stack(streamed, dim=-1))


def test_decode_applies_silu():
    x = torch.randn(2, 4)
    w = torch.randn(4, 1, 3)
    b = torch.randn(4)
    plain, _ = fallback_causal_conv1d_update(x, torch.zeros(2, 4, 2), w, b)
    silu, _ = fallback_causal_conv1d_update(x, torch.zeros(2, 4, 2), w, b,
                                            activation="silu")
    assert _close(silu, torch.nn.functional.silu(plain))


# --- miopen error classification -------------------------------------------

@pytest.mark.parametrize("msg,expected", [
    ("MIOPEN error: no suitable algorithm", True),
    ("MIOpen Status: Invalid elapsed time", True),
    ("No suitable algorithm found", True),
    ("HIP error: out of memory", False),
    ("mat1 and mat2 shapes cannot be multiplied", False),
])
def test_is_miopen_error(msg, expected):
    assert _is_miopen_error(RuntimeError(msg)) is expected


def test_is_miopen_error_is_case_insensitive():
    assert _is_miopen_error(RuntimeError("NO SUITABLE ALGORITHM")) is True


# --- probing and install ---------------------------------------------------

def test_probe_reports_false_on_exception():
    def boom(*a, **k):
        raise RuntimeError("hipErrorInvalidDeviceFunction")
    assert _probe_works(boom) is False


def test_probe_reports_true_when_cuda_absent(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []

    def probe(x, w, b):
        calls.append(1)
        raise RuntimeError("should not be reached on the CPU path")

    # With cuda unavailable the probe must short-circuit, not allocate.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _probe_works(probe) is True
    assert calls == []


def test_install_env_off_disables(monkeypatch):
    monkeypatch.setenv("LFM_CONV_FALLBACK", "off")
    assert maybe_install_from_env() == "disabled-by-env"


def test_install_force_installs_fallback(monkeypatch):
    monkeypatch.setenv("LFM_CONV_FALLBACK", "force")
    import src.compat_lfm2_rocm as mod
    monkeypatch.setattr(mod, "PROBE_DONE", False)
    assert install_if_needed(force=True) == "fallback-installed"
    monkeypatch.setattr(mod, "PROBE_DONE", False)


def test_install_probe_only_runs_once(monkeypatch):
    import src.compat_lfm2_rocm as mod
    monkeypatch.setattr(mod, "PROBE_DONE", False)
    install_if_needed(force=True)
    monkeypatch.setattr(mod, "PROBE_DONE", True)
    assert install_if_needed() == "already"


# --- apply_to_model --------------------------------------------------------

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 4, 3, groups=4, padding=1)
        self.other = nn.Conv1d(4, 8, 1)
        self.lin = nn.Linear(4, 4)


def test_apply_to_model_force_matches_native_forward(monkeypatch):
    torch.manual_seed(6)
    net = _Net()
    x = torch.randn(2, 4, 12)
    want = net.conv(x)
    monkeypatch.setenv("LFM_CONV_FALLBACK", "force")
    status = apply_to_model(net)
    assert status == "force-mode:2convs", status
    assert _close(net.conv(x), want)


def test_apply_to_model_off_respects_env(monkeypatch):
    net = _Net()
    monkeypatch.setenv("LFM_CONV_FALLBACK", "off")
    assert apply_to_model(net) == "disabled-by-env"
    # Nothing must have been wrapped.
    assert net.conv.forward.__func__ is nn.Conv1d.forward


def test_apply_to_model_rescue_prefers_native(monkeypatch):
    torch.manual_seed(7)
    net = _Net()
    x = torch.randn(2, 4, 12)
    want = net.conv(x)
    monkeypatch.setenv("LFM_CONV_FALLBACK", "auto")
    assert apply_to_model(net) == "rescue-mode:2convs"
    assert _close(net.conv(x), want)


def test_apply_to_model_rescue_falls_back_on_miopen_error(monkeypatch):
    torch.manual_seed(8)
    net = _Net()
    x = torch.randn(2, 4, 12)
    want = net.conv(x)  # reference before wrapping

    monkeypatch.setenv("LFM_CONV_FALLBACK", "auto")
    apply_to_model(net)

    real = nn.Conv1d.forward
    calls = {"n": 0}

    def flaky(self, inp):
        calls["n"] += 1
        raise RuntimeError("MIOpen: no suitable algorithm")

    monkeypatch.setattr(nn.Conv1d, "forward", flaky)
    got = net.conv(x)  # must take the unfold rescue path
    monkeypatch.setattr(nn.Conv1d, "forward", real)
    assert calls["n"] == 1, "native path should have been attempted first"
    assert _close(got, want)


def test_apply_to_model_rescue_reraises_non_miopen(monkeypatch):
    net = _Net()
    x = torch.randn(2, 4, 12)
    monkeypatch.setenv("LFM_CONV_FALLBACK", "auto")
    apply_to_model(net)

    real = nn.Conv1d.forward
    monkeypatch.setattr(nn.Conv1d, "forward",
                        lambda self, inp: (_ for _ in ()).throw(
                            RuntimeError("shape mismatch")))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        net.conv(x)
    monkeypatch.setattr(nn.Conv1d, "forward", real)


def test_apply_to_model_leaves_state_dict_keys_intact(monkeypatch):
    net = _Net()
    before = set(net.state_dict().keys())
    monkeypatch.setenv("LFM_CONV_FALLBACK", "force")
    apply_to_model(net)
    assert set(net.state_dict().keys()) == before


def test_apply_to_model_model_without_conv(monkeypatch):
    lin_only = nn.Sequential(nn.Linear(4, 4))
    monkeypatch.setenv("LFM_CONV_FALLBACK", "force")
    assert apply_to_model(lin_only) == "force-mode:0convs"
