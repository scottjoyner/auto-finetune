"""In-memory optimizer interruption/resume parity on fabricated PyTorch data.

Tests lossless AdamW and native Muon+AdamW optimizer-state recovery, including
the exact training step cursor, parameter-group identity and RNG state.
No source trace ingestion, external checkpoint load, file/network writes,
real LFM3 finetuning or permission to promote a resumed experiment.
"""
from __future__ import annotations
import hashlib
import hmac
import io
import json

from src.lfm3_optimizer_guard import OptimizerResearchRefused

SEED = 97
STEPS = 12
SPLIT_STEP = 5
LEARNING_RATE = 0.001


def build(mode: str):
    if mode not in ("adamw", "muon_hybrid"):
        raise OptimizerResearchRefused("unknown optimizer resume profile")
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    model = nn.Sequential(__import__("collections").OrderedDict([
        ("hidden", nn.Linear(12, 24)),
        ("act", nn.Tanh()),
        ("head", nn.Linear(24, 4)),
    ])).float().cpu()
    named = list(model.named_parameters())
    if [n for n, _ in named] != [
        "hidden.weight", "hidden.bias", "head.weight", "head.bias"
    ]:
        raise OptimizerResearchRefused("toy architecture changed")
    muon = [named[0][1]]
    rest = [v for _, v in named[1:]]
    if mode == "adamw":
        optimizers = [torch.optim.AdamW(muon + rest, lr=LEARNING_RATE, weight_decay=0.01)]
    else:
        if not hasattr(torch.optim, "Muon"):
            raise OptimizerResearchRefused("native Muon absent: never silently substitute AdamW")
        optimizers = [
            torch.optim.Muon(muon, lr=LEARNING_RATE,
                             momentum=0.95, ns_steps=5,
                             adjust_lr_fn="original", weight_decay=0.01),
            torch.optim.AdamW(rest, lr=LEARNING_RATE, weight_decay=0.01),
        ]
    return model, optimizers


def synthetic_data():
    import torch
    rng = torch.Generator(device="cpu").manual_seed(137)
    x = torch.randn((96, 12), generator=rng)
    t = torch.randn((12, 4), generator=rng)
    y = torch.tanh(x @ t)
    return x, y


def steps(model, optimizers, x, y, start, stop):
    import torch
    from torch.nn import functional as F
    if (type(start) is not int or type(stop) is not int
            or not 0 <= start < stop <= STEPS):
        raise OptimizerResearchRefused("resume step cursor outside fixed budget")
    for _ in range(start, stop):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        if not bool(torch.isfinite(loss)):
            raise OptimizerResearchRefused("nonfinite optimizer loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad)):
            raise OptimizerResearchRefused("nonfinite optimizer gradient")
        for opt in optimizers:
            opt.step()
        if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise OptimizerResearchRefused("optimizer restored to nonfinite parameters")


def capture_in_memory(model, opts, *, optimizer: str, next_step: int):
    """Only writes a BytesIO created in this process; not an external pickle."""
    if next_step != SPLIT_STEP:
        raise OptimizerResearchRefused("unapproved optimizer checkpoint cursor")
    import torch
    snapshot = {
        "schema": "sg-lfm3-optimizer-inmemory-state.v1",
        "optimizer_profile": optimizer,
        "next_step": next_step,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "parameter_names": [
            "hidden.weight", "hidden.bias", "head.weight", "head.bias"
        ],
        "weights": model.state_dict(),
        "optimizer_states": [o.state_dict() for o in opts],
        "rng_state": torch.get_rng_state(),
    }
    memory = io.BytesIO()
    torch.save(snapshot, memory)
    raw = memory.getvalue()
    if not 0 < len(raw) <= 256_000:
        raise OptimizerResearchRefused("optimizer scratch checkpoint too large")
    return raw


def restore_in_memory(raw: bytes, *, optimizer: str, expected_sha256: str):
    if type(raw) is not bytes or not 0 < len(raw) <= 256_000:
        raise OptimizerResearchRefused("optimizer checkpoint size invalid")
    # The caller must hold the expected digest independently of this payload.
    # This is equality verification only, not source-custody authentication.
    if (type(expected_sha256) is not str or len(expected_sha256) != 64
            or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256)):
        raise OptimizerResearchRefused("optimizer checkpoint digest mismatch before load")
    import torch
    # Untrusted external checkpoints are never admitted: this function
    # is used only with the freshly produced in-process buffer.
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (type(saved) is not dict
            or set(saved) != {
                "schema", "optimizer_profile", "next_step", "seed",
                "learning_rate", "parameter_names", "weights",
                "optimizer_states", "rng_state",
            }
            or saved["schema"] != "sg-lfm3-optimizer-inmemory-state.v1"
            or saved["optimizer_profile"] != optimizer
            or type(saved["next_step"]) is not int
            or saved["next_step"] != SPLIT_STEP
            or saved["seed"] != SEED
            or saved["learning_rate"] != LEARNING_RATE
            or saved["parameter_names"] != [
                "hidden.weight", "hidden.bias", "head.weight", "head.bias"
            ]):
        raise OptimizerResearchRefused("resume checkpoint cross-experiment identity mismatch")
    model, opt = build(optimizer)
    if (set(saved["weights"]) != set(model.state_dict())
            or type(saved["optimizer_states"]) is not list
            or len(saved["optimizer_states"]) != len(opt)):
        raise OptimizerResearchRefused("model tensor or optimizer count mismatch")
    expected_weights = model.state_dict()
    for name, weight in saved["weights"].items():
        reference = expected_weights[name]
        if (type(weight) is not torch.Tensor
                or weight.dtype != reference.dtype
                or tuple(weight.shape) != tuple(reference.shape)
                or not bool(torch.isfinite(weight).all())):
            raise OptimizerResearchRefused("resume model tensor dtype/shape/finite mismatch")
    model.load_state_dict(saved["weights"], strict=True)
    for current, state in zip(opt, saved["optimizer_states"]):
        if (type(state) is not dict
                or type(state.get("param_groups")) is not list
                or len(state["param_groups"]) != len(current.param_groups)
                or state["param_groups"] != current.state_dict()["param_groups"]):
            raise OptimizerResearchRefused("optimizer state/group mismatch")
        current.load_state_dict(state)
    if type(saved["rng_state"]) is not torch.Tensor or saved["rng_state"].dtype != torch.uint8:
        raise OptimizerResearchRefused("training RNG state corrupt")
    torch.set_rng_state(saved["rng_state"])
    return model, opt, saved["next_step"]


def parameter_digest(model) -> str:
    import torch
    h = hashlib.sha256(b"sg-optimizer-cpu-exact-param-v1\0")
    for name, p in model.named_parameters():
        h.update(name.encode() + b"\0")
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(mode: str) -> dict:
    import torch
    torch.set_num_threads(2)
    x, y = synthetic_data()
    model, opts = build(mode)
    steps(model, opts, x, y, 0, STEPS)
    uninterrupted = parameter_digest(model)
    unbroken_state = [o.state_dict() for o in opts]
    # Reseed and rebuild the entire experiment, then interrupt at step five.
    fresh, fresh_opt = build(mode)
    steps(fresh, fresh_opt, x, y, 0, SPLIT_STEP)
    raw = capture_in_memory(fresh, fresh_opt, optimizer=mode, next_step=SPLIT_STEP)
    scratch_sha = hashlib.sha256(raw).hexdigest()
    recovered, recovered_opt, cursor = restore_in_memory(
        raw, optimizer=mode, expected_sha256=scratch_sha
    )
    steps(recovered, recovered_opt, x, y, cursor, STEPS)
    resumed = parameter_digest(recovered)
    same_optimizer_state = True
    for before, after in zip(unbroken_state, [o.state_dict() for o in recovered_opt]):
        if set(before["state"]) != set(after["state"]):
            same_optimizer_state = False
        for index in before["state"]:
            for key, value in before["state"][index].items():
                alt = after["state"][index][key]
                if isinstance(value, torch.Tensor):
                    same_optimizer_state &= bool(torch.equal(value, alt))
                else:
                    same_optimizer_state &= value == alt
    if uninterrupted != resumed or not same_optimizer_state:
        raise OptimizerResearchRefused("resume diverged from uninterrupted optimizer")
    return {
        "schema": "sg-lfm3-optimizer-resume-proof.v1",
        "source": "fabricated-cpu-toy-only",
        "optimizer": mode, "seed": SEED,
        "steps": STEPS, "interrupted_after_step": SPLIT_STEP,
        "checkpoint_bytes": len(raw),
        "in_memory_checkpoint_sha256": scratch_sha,
        "uninterrupted_parameters_sha256": uninterrupted,
        "resumed_parameters_sha256": resumed,
        "bit_exact_parameter_resume": True,
        "bit_exact_optimizer_state_resume": same_optimizer_state,
        "checkpoint_written_to_disk": False,
        "untrusted_external_state_loaded": False,
        "real_lfm3_weights_trained": False,
        "production_ready": False,
    }


if __name__ == "__main__":
    print(json.dumps({
        "schema": "sg-lfm3-resume-paired-lab.v1",
        "arms": [run("adamw"), run("muon_hybrid")],
        "production_ready": False,
    }, sort_keys=True))
