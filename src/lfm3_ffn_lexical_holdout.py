"""Pre-registered one-shot fictional lexical holdout on actual LFM3 FFN-LoRA.

Optimizer selection is fixed from the existing six-arm development receipt
before the holdout is evaluated. No model vote, tuning based on heldout,
checkpoint persistence, autonomous recursion, or promotion is allowed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_frozen_readout_lab import TRAIN, DEV, target_tokens, assert_fixed_lab_contract
from src.lfm3_last_layer_lora_lab import (
    TARGET, LEARNING_RATE, RANK, RESEARCH_SEEDS, OPTIMIZERS, install_adapter)
from src.lfm3_ffn_multiseed_guard import admit_three_seed_training
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

HOLDOUT = (
    ("Fictional codebook: amberline corresponds to A. Letter for amberline:", "A"),
    ("Fictional codebook: birchline corresponds to B. Letter for birchline:", "B"),
    ("Fictional codebook: goldleaf corresponds to A. Letter for goldleaf:", "A"),
    ("Fictional codebook: silverleaf corresponds to B. Letter for silverleaf:", "B"),
)
BUDGET = 3


def validate_protocol(holdout=HOLDOUT):
    assert_fixed_lab_contract()
    if (type(holdout) is not tuple or holdout != HOLDOUT
            or len({prompt for prompt, _ in TRAIN + DEV + holdout}) != 8
            or tuple(label for _, label in holdout) != ("A", "B", "A", "B")
            or any(not prompt.startswith("Fictional codebook:") or len(prompt) > 120
                   for prompt, _ in holdout)):
        raise ResearchRefused("frozen fictional holdout changed or train/dev aliased")
    return True


def fixed_selection(precommitted):
    """Use only the independently checked *dev* receipt. No heldout argument."""
    judge = admit_three_seed_training(precommitted)
    if judge["fixed_objective_best"] != "A" or not judge["may_propose_next_synthetic_review_only"]:
        raise ResearchRefused("development-only optimizer decision not independently admitted")
    if precommitted["arms"][0]["optimizer"] != "adamw":
        raise ResearchRefused("precommitted optimizer labels changed")
    return {
        "optimizer": "adamw",
        "learning_rate": 0.002,
        "seeds": list(RESEARCH_SEEDS),
        "three_training_steps": BUDGET,
        "frozen_dev_receipt_sha256": hashlib.sha256(
            json.dumps(precommitted, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "model_vote_influenced_selection": False,
        "heldout_influenced_selection": False,
        "may_execute_outside_scratch": False,
    }


def run(snapshot: Path, receipt: Path) -> dict:
    verify_restricted_environment()
    validate_protocol()
    precommitted = json.loads(receipt.read_text())
    selection = fixed_selection(precommitted)
    proof = verify_snapshot(snapshot)
    import torch
    import transformers
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__ != "5.21.0":
        raise ResearchRefused("unreviewed transformers runtime")
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16)
    if model.__class__.__name__ != "Lfm3ForCausalLM":
        raise ResearchRefused("unexpected LFM3 architecture")
    model.eval()
    model.requires_grad_(False)
    base = tuple(model.parameters())
    versions = tuple(p._version for p in base)
    target = model.get_submodule(TARGET)
    if (not isinstance(target, torch.nn.Linear)
            or tuple(target.weight.shape) != (2048, 7680)):
        raise ResearchRefused("unexpected final FFN tensor dimensions")
    ids = target_tokens(tok)

    def encode(rows):
        result = []
        for prompt, answer in rows:
            payload = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt")
            if payload["input_ids"].shape[1] > 80:
                raise ResearchRefused("fictional prompt exceeds fixed context budget")
            result.append((payload, ids[answer]))
        return result

    train = encode(TRAIN)
    dev = encode(DEV)
    holdout = encode(HOLDOUT)

    def loss(items):
        scores = []
        for payload, answer in items:
            logits = model(**payload, use_cache=False).logits[0, -1].float()
            scores.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([answer])))
        return torch.stack(scores).mean()

    with torch.no_grad():
        original_dev = float(loss(dev))
        original_holdout = float(loss(holdout))
    if not math.isfinite(original_holdout):
        raise ResearchRefused("invalid frozen-model holdout baseline")

    trials = []
    for index, seed in enumerate(RESEARCH_SEEDS):
        for i, mode in enumerate(OPTIMIZERS):
            A, B, hook = install_adapter(target, seed=seed)
            trainables = list(A.parameters()) + list(B.parameters())
            opt = (torch.optim.AdamW(trainables, lr=LEARNING_RATE, weight_decay=0.0)
                   if i == 0 else torch.optim.Muon(
                       trainables, lr=LEARNING_RATE, momentum=0.95, ns_steps=5,
                       adjust_lr_fn="match_rms_adamw", weight_decay=0.0))
            try:
                with torch.no_grad():
                    first_dev = float(loss(dev))
                if abs(first_dev - original_dev) > 0.0001:
                    raise ResearchRefused("optimizer arms had different base loss")
                for _ in range(BUDGET):
                    opt.zero_grad(set_to_none=True)
                    objective = loss(train)
                    if not bool(torch.isfinite(objective)):
                        raise ResearchRefused("nonfinite real-model objective")
                    objective.backward()
                    if B.weight.grad is None or not bool(torch.isfinite(B.weight.grad).all()):
                        raise ResearchRefused("real FFN adapter lacked valid gradient")
                    norm = torch.nn.utils.clip_grad_norm_(trainables, 1.0)
                    if not bool(torch.isfinite(norm)):
                        raise ResearchRefused("nonfinite adapter gradients")
                    opt.step()
                with torch.no_grad():
                    result_dev = float(loss(dev))
                expected = precommitted["arms"][2*index+i]["dev_loss_after"]
                if abs(result_dev - expected) > 0.01:
                    raise ResearchRefused("dev result not reproduced; holdout not admissible")
                # Only after fixed choice AND reproducibility gate; no updates follow.
                with torch.no_grad():
                    result_holdout = float(loss(holdout))
                if not math.isfinite(result_holdout):
                    raise ResearchRefused("nonfinite one-shot holdout loss")
                trials.append({
                    "seed": seed, "optimizer": mode,
                    "dev_loss_reproduced": result_dev,
                    "precommitted_dev_loss": expected,
                    "one_shot_lexical_holdout_loss": result_holdout,
                    "adapter_checkpoint_written": False,
                })
            finally:
                hook.remove()
    if tuple(p._version for p in base) != versions or any(
            p.grad is not None or p.requires_grad for p in base):
        raise ResearchRefused("pretrained model was modified")
    winners = [
        trials[2*i]["one_shot_lexical_holdout_loss"]
        < trials[2*i+1]["one_shot_lexical_holdout_loss"]
        for i in range(3)
    ]
    return {
        "schema": "sg-lfm3-real-ffn-dev-selected-lexical-holdout.v1",
        "checkpoint_sha256": proof["checkpoint_sha256"],
        "model_revision": proof["revision"],
        "source": "same-family-fictional-codebook-lexical-holdout-only",
        "selection": selection,
        "train_sha256": hashlib.sha256(repr(TRAIN).encode()).hexdigest(),
        "dev_sha256": hashlib.sha256(repr(DEV).encode()).hexdigest(),
        "holdout_sha256": hashlib.sha256(repr(HOLDOUT).encode()).hexdigest(),
        "train_size": 2, "dev_size": 2, "holdout_size": 4,
        "frozen_model_dev_loss": original_dev,
        "frozen_model_holdout_loss": original_holdout,
        "trials": trials,
        "selected_adamw_beats_muon_on_holdout_per_seed": winners,
        "optimizer_selected_using_heldout": False,
        "additional_tuning_after_holdout": False,
        "original_model_parameters_frozen": 1_728_801_536,
        "base_weights_modified": False,
        "adapter_checkpoint_saved": False,
        "source_authenticated": False,
        "independent_task_family": False,
        "may_promote": False,
        "production_ready": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-only", action="store_true", required=True)
    parser.add_argument("--ack-sandbox-research", action="store_true", required=True)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.snapshot, args.receipt), sort_keys=True))
