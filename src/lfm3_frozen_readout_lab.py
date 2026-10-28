"""Pinned real LFM3 frozen-feature low-rank adapter research (CPU only).

The official BF16 backbone is read-only, fully frozen and evaluated without a
gradient graph. We train only a fresh 2-layer rank-4 *readout residual* from
a few fabricated final hidden states to the fixed LM head. This is NOT
end-to-end LFM3 LoRA/SFT, adapter deployment or recursive self-improvement.
There are no production traces or model-checkpoint writes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

TRAIN = (
    ("Fictional puzzle: option azure means A. What letter represents azure? Letter:", "A"),
    ("Fictional puzzle: option bronze means B. What letter represents bronze? Letter:", "B"),
)
DEV = (
    ("Fictional puzzle: option coral means A. What letter represents coral? Letter:", "A"),
    ("Fictional puzzle: option dune means B. What letter represents dune? Letter:", "B"),
)
RANK = 4
STEPS = 6
LRS = (0.001, 0.002)


def assert_fixed_lab_contract(train=TRAIN, dev=DEV):
    if (type(train) is not tuple or type(dev) is not tuple
            or len(train) != 2 or len(dev) != 2
            or len({x[0] for x in train + dev}) != 4):
        raise ResearchRefused("invalid train/dev synthetic split")
    for text, answer in train + dev:
        if (type(text) is not str or type(answer) is not str
                or not 5 <= len(text) <= 160 or answer not in ("A", "B")
                or not text.startswith("Fictional puzzle:")):
            raise ResearchRefused("unapproved toy feature extraction sample")
    return True


def target_tokens(tokenizer) -> dict[str, int]:
    targets = {}
    for letter in ("A", "B"):
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ResearchRefused("toy label must map to exactly one vocabulary token")
        targets[letter] = ids[0]
    if targets["A"] == targets["B"]:
        raise ResearchRefused("fabricated label collision")
    return targets


def fit_readout_adapter(features, targets, frozen_lm_head, *,
                        lr: float, steps: int = STEPS,
                        optimizer_kind: str = "adamw",
                        explicit_toy_muon_review: bool = False) -> dict:
    if (type(lr) not in (float, int) or lr not in LRS
            or type(steps) is not int or not 1 <= steps <= STEPS):
        raise ResearchRefused("unsupported adapter optimizer budget")
    if optimizer_kind not in ("adamw", "muon_readout_research"):
        raise ResearchRefused("only pinned AdamW or reviewed toy Muon option")
    if optimizer_kind != "adamw" and explicit_toy_muon_review is not True:
        raise ResearchRefused("Muon on low-rank readout prohibited without lab exception")
    import torch
    from torch import nn
    import torch.nn.functional as F

    x_train, x_dev = features
    y_train, y_dev = targets
    if (x_train.ndim != 2 or x_dev.ndim != 2
            or x_train.shape != (2, 2048) or x_dev.shape != (2, 2048)
            or y_train.shape != (2,) or y_dev.shape != (2,)):
        raise ResearchRefused("unexpected LFM3 feature/target shape")
    if (not bool(torch.isfinite(x_train).all())
            or not bool(torch.isfinite(x_dev).all())):
        raise ResearchRefused("invalid frozen model activations")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261028)
        first = nn.Linear(2048, RANK, bias=False, dtype=torch.float32)
        last = nn.Linear(RANK, 2048, bias=False, dtype=torch.float32)
        nn.init.zeros_(last.weight)
    trainables = list(first.parameters()) + list(last.parameters())
    if sum(p.numel() for p in trainables) != 2 * 2048 * RANK:
        raise ResearchRefused("readout residual has unexpected trainable budget")
    for p in frozen_lm_head.parameters():
        if p.requires_grad:
            raise ResearchRefused("official vocabulary readout must be frozen")
    if optimizer_kind == "adamw":
        opt = torch.optim.AdamW(trainables, lr=lr, weight_decay=0.0)
    else:
        # The residual is not an approved LFM3 LoRA or pretrained parameter.
        # This is an explicit experiment on two known 2-D toy matrices only.
        if not hasattr(torch.optim, "Muon"):
            raise ResearchRefused("native Torch Muon unavailable; no substitution")
        if (tuple(first.weight.shape) != (RANK, 2048)
                or tuple(last.weight.shape) != (2048, RANK)):
            raise ResearchRefused("Muon toy matrix layout changed")
        opt = torch.optim.Muon(
            trainables, lr=lr, momentum=0.95, weight_decay=0.0,
            ns_steps=5, adjust_lr_fn="original"
        )

    def loss_on(x, y):
        # Readout residual computed in fp32, official LM head remains in bf16.
        delta = last(first(x))
        logits = frozen_lm_head((x + delta).to(dtype=torch.bfloat16)).float()
        return F.cross_entropy(logits, y)

    with torch.no_grad():
        before_train = float(loss_on(x_train, y_train))
        before_dev = float(loss_on(x_dev, y_dev))
    worst_grad = 0.0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_on(x_train, y_train)
        if not bool(torch.isfinite(loss)):
            raise ResearchRefused("nonfinite adapter training loss")
        loss.backward()
        grad = nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
        if not bool(torch.isfinite(grad)):
            raise ResearchRefused("nonfinite adapter gradients")
        worst_grad = max(worst_grad, float(grad))
        opt.step()
        if not all(bool(torch.isfinite(p).all()) for p in trainables):
            raise ResearchRefused("nonfinite adapter update")
    with torch.no_grad():
        after_train = float(loss_on(x_train, y_train))
        after_dev = float(loss_on(x_dev, y_dev))
    if not all(math.isfinite(x) for x in (
        before_train, before_dev, after_train, after_dev, worst_grad
    )):
        raise ResearchRefused("invalid adapter score")
    return {
        "optimizer": ("adamw_torch" if optimizer_kind == "adamw"
                      else "muon_torch_on_toy_readout_only"),
        "learning_rate": lr,
        "steps": steps,
        "rank": RANK,
        "trainable_residual_parameters": sum(p.numel() for p in trainables),
        "train_loss_before": before_train,
        "train_loss_after": after_train,
        "dev_loss_before": before_dev,
        "dev_loss_after": after_dev,
        "max_preclip_grad_norm": worst_grad,
        "weights_saved": False,
        "muon_on_actual_lfm3_weights": False,
        "production_ready": False,
    }


def run(snapshot: Path, *, allow_toy_muon: bool = False) -> dict:
    verify_restricted_environment()
    assert_fixed_lab_contract()
    assets = verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("unexpected Transformers runtime")
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=True,
        dtype=torch.bfloat16
    )
    if model.__class__.__name__ != "Lfm3ForCausalLM":
        raise ResearchRefused("unexpected pretrained model class")
    model.eval()
    model.requires_grad_(False)
    base_params = tuple(model.parameters())
    base_versions = tuple(p._version for p in base_params)
    if any(p.requires_grad for p in base_params):
        raise ResearchRefused("pretrained LFM3 is not completely frozen")
    vocab = target_tokens(tok)
    features = []
    labels = []
    for split in (TRAIN, DEV):
        inputs = []
        targets = []
        for text, label in split:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors="pt",
            )
            if ids["input_ids"].shape[1] > 80:
                raise ResearchRefused("toy prompt over bounded context window")
            with torch.no_grad():
                output = model(**ids, output_hidden_states=True, use_cache=False)
                last = output.hidden_states[-1][0, -1, :].detach().clone().float()
                del output
            inputs.append(last)
            targets.append(vocab[label])
        features.append(torch.stack(inputs))
        labels.append(torch.tensor(targets, dtype=torch.long))
    if tuple(features[0].shape) != (2, 2048):
        raise ResearchRefused("unexpected pretrained LFM3 hidden size")
    if (id(features[0]) == id(features[1])
            or any(p.requires_grad for p in base_params)):
        raise ResearchRefused("pretrained LFM3 gradient isolation failed")
    arms = [fit_readout_adapter(tuple(features), tuple(labels), model.lm_head,
                                lr=lr) for lr in LRS]
    if allow_toy_muon is True:
        arms.extend(fit_readout_adapter(
            tuple(features), tuple(labels), model.lm_head, lr=lr,
            optimizer_kind="muon_readout_research",
            explicit_toy_muon_review=True,
        ) for lr in LRS)
    elif allow_toy_muon is not False:
        raise ResearchRefused("research Muon flag must be an explicit boolean")
    if tuple(p._version for p in base_params) != base_versions:
        raise ResearchRefused("a pretrained tensor changed unexpectedly")
    return {
        "schema": ("sg-lfm3-frozen-readout-muon-grid.v1" if allow_toy_muon
                   else "sg-lfm3-frozen-readout-adapter.v1"),
        "source": "fabricated-toy-not-actual-agent-traces",
        "checkpoint_sha256": assets["checkpoint_sha256"],
        "model_revision": assets["revision"],
        "model_parameters_frozen": sum(p.numel() for p in base_params),
        "backbone_parameter_versions_unchanged": True,
        "synthetic_train_sha256": hashlib.sha256(repr(TRAIN).encode()).hexdigest(),
        "synthetic_dev_sha256": hashlib.sha256(repr(DEV).encode()).hexdigest(),
        "head": "existing_read_only_full_vocabulary_lm_head",
        "residual": "low_rank_final_hidden_readout_only_not_lora",
        "train_count": 2, "dev_count": 2,
        "arms": arms,
        "no_heldout_evaluation": True,
        "no_finetune_of_pretrained_model": True,
        "no_model_weights_saved": True,
        "no_muon_adapter_optimization_executed": not allow_toy_muon,
        "muon_research_exception_only_for_two_readout_matrices": allow_toy_muon,
        "cannot_promote_fleet": True,
        "production_ready": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", action="store_true", required=True)
    p.add_argument("--ack-sandbox-research", action="store_true", required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--toy-muon-readout-explicit-exception", action="store_true")
    args = p.parse_args()
    try:
        print(json.dumps(run(
            args.snapshot,
            allow_toy_muon=args.toy_muon_readout_explicit_exception,
        ), sort_keys=True))
        return 0
    except (ResearchRefused, OSError, ValueError) as exc:
        raise SystemExit("frozen feature adapter refused: " + str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
