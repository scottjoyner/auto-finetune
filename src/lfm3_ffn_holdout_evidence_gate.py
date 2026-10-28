"""Independent verification of real-FFN adapter's once-only fictional lexical holdout.

This gate can nominate a NEW *synthetic* experiment for human review. It cannot
change training, optimizer hyperparameters, deployment, or model weights. The
task remains the same lexical family, so no claim of broad generalization.
"""
from __future__ import annotations
import hashlib
import json
import math
from src.lfm3_ffn_multiseed_guard import admit_three_seed_training, PIN, REV, SEEDS, MODES
from src.lfm3_real_checkpoint_smoke import ResearchRefused

FIELDS = {
    "schema", "checkpoint_sha256", "model_revision", "source",
    "selection", "train_sha256", "dev_sha256", "holdout_sha256",
    "train_size", "dev_size", "holdout_size", "frozen_model_dev_loss",
    "frozen_model_holdout_loss", "trials",
    "selected_adamw_beats_muon_on_holdout_per_seed",
    "optimizer_selected_using_heldout", "additional_tuning_after_holdout",
    "original_model_parameters_frozen", "base_weights_modified",
    "adapter_checkpoint_saved", "source_authenticated",
    "independent_task_family", "may_promote", "production_ready",
    "experiment_date", "node", "offline_bwrap_enforced",
    "independent_source_attestation", "heldout_family_independently_curated",
    "model_training_authorized",
}
SELECTION = {
    "optimizer", "learning_rate", "seeds", "three_training_steps",
    "frozen_dev_receipt_sha256", "model_vote_influenced_selection",
    "heldout_influenced_selection", "may_execute_outside_scratch",
}
TRIAL = {
    "seed", "optimizer", "dev_loss_reproduced", "precommitted_dev_loss",
    "one_shot_lexical_holdout_loss", "adapter_checkpoint_written",
}


def verify_holdout(result: object, precommitted: object) -> dict:
    baseline_gate = admit_three_seed_training(precommitted)
    if (baseline_gate["fixed_objective_best"] != "A"
            or not baseline_gate["may_propose_next_synthetic_review_only"]):
        raise ResearchRefused("original dev-only optimizer decision did not pass")
    if type(result) is not dict or set(result) != FIELDS:
        raise ResearchRefused("holdout packet schema changed")
    if (result["schema"] != "sg-lfm3-real-ffn-dev-selected-lexical-holdout.v1"
            or result["checkpoint_sha256"] != PIN
            or result["model_revision"] != REV
            or result["source"] != "same-family-fictional-codebook-lexical-holdout-only"
            or result["train_size"] != 2 or result["dev_size"] != 2
            or result["holdout_size"] != 4
            or result["original_model_parameters_frozen"] != 1_728_801_536
            or result["optimizer_selected_using_heldout"] is not False
            or result["additional_tuning_after_holdout"] is not False
            or result["base_weights_modified"] is not False
            or result["adapter_checkpoint_saved"] is not False
            or result["source_authenticated"] is not False
            or result["independent_task_family"] is not False
            or result["may_promote"] is not False
            or result["production_ready"] is not False
            or result["experiment_date"] != "2026-10-28"
            or result["node"] != "xwing"
            or result["offline_bwrap_enforced"] is not True
            or result["independent_source_attestation"] is not False
            or result["heldout_family_independently_curated"] is not False
            or result["model_training_authorized"] is not False):
        raise ResearchRefused("holdout provenance/authority contract mismatch")
    for key in ("train_sha256", "dev_sha256", "holdout_sha256"):
        v = result[key]
        if type(v) is not str or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ResearchRefused("not canonical synthetic split identity")
    if len({result[k] for k in ("train_sha256", "dev_sha256", "holdout_sha256")}) != 3:
        raise ResearchRefused("source split aliases")
    if (result["train_sha256"] != precommitted["synthetic_train_sha256"]
            or result["dev_sha256"] != precommitted["synthetic_dev_sha256"]):
        raise ResearchRefused("holdout did not preserve precommitted training/dev source")
    sel = result["selection"]
    canonical_dev = hashlib.sha256(json.dumps(
        precommitted, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (type(sel) is not dict or set(sel) != SELECTION
            or sel["optimizer"] != "adamw"
            or type(sel["learning_rate"]) is not float
            or sel["learning_rate"] != 0.002
            or sel["seeds"] != list(SEEDS)
            or type(sel["three_training_steps"]) is not int
            or sel["three_training_steps"] != 3
            or sel["frozen_dev_receipt_sha256"] != canonical_dev
            or sel["model_vote_influenced_selection"] is not False
            or sel["heldout_influenced_selection"] is not False
            or sel["may_execute_outside_scratch"] is not False):
        raise ResearchRefused("dev-only optimizer selection was rewritten")
    baseline = result["frozen_model_holdout_loss"]
    original_dev = result["frozen_model_dev_loss"]
    if (type(baseline) not in (float, int) or not math.isfinite(baseline)
            or baseline <= 0 or type(original_dev) not in (float,int)
            or not math.isfinite(original_dev) or original_dev <= 0):
        raise ResearchRefused("invalid frozen baseline")
    arms = result["trials"]
    if type(arms) is not list or len(arms) != 6:
        raise ResearchRefused("need six paired real-model holdout evaluations")
    pair_wins = []
    gain = []
    dev_max_difference = 0.0
    for index, seed in enumerate(SEEDS):
        pair = []
        for option, mode in enumerate(MODES):
            arm = arms[2*index+option]
            if (type(arm) is not dict or set(arm) != TRIAL
                    or type(arm["seed"]) is not int or arm["seed"] != seed
                    or arm["optimizer"] != mode
                    or arm["adapter_checkpoint_written"] is not False):
                raise ResearchRefused("noncanonical holdout arm")
            for name in ("dev_loss_reproduced", "precommitted_dev_loss",
                         "one_shot_lexical_holdout_loss"):
                x = arm[name]
                if type(x) not in (float,int) or not math.isfinite(x) or x < 0:
                    raise ResearchRefused("nonfinite holdout/dev metric")
            expected = precommitted["arms"][2*index+option]["dev_loss_after"]
            delta = abs(arm["dev_loss_reproduced"]-expected)
            if (abs(arm["precommitted_dev_loss"]-expected) > 1e-9
                    or delta > 0.01):
                raise ResearchRefused("precommitted dev result not reproduced")
            dev_max_difference = max(dev_max_difference, delta)
            pair.append(arm["one_shot_lexical_holdout_loss"])
        pair_wins.append(pair[0] < pair[1])
        gain.append((baseline-pair[0])/baseline)
    if (result["selected_adamw_beats_muon_on_holdout_per_seed"] != pair_wins
            or any(type(x) is not bool for x in result["selected_adamw_beats_muon_on_holdout_per_seed"])):
        raise ResearchRefused("holdout win claim is not independently derived")
    eligible = all(pair_wins) and all(x >= 0.05 for x in gain)
    return {
        "schema": "sg-lfm3-ffn-heldout-independent-evaluation.v1",
        "trials": 6, "seeds": list(SEEDS),
        "dev_max_reproduction_gap": dev_max_difference,
        "holdout_base_loss": baseline,
        "selected_adamw_holdout_relative_gain_by_seed": gain,
        "selected_adamw_lower_holdout_loss_than_muon": pair_wins,
        "related_fictional_family_only": True,
        "independent_task_generalization_verified": False,
        "eligible_for_new_synthetic_review": eligible,
        "execute_optimizer_change": False,
        "save_adapter": False,
        "allow_model_self_modification": False,
        "promotion_authorized": False,
        "production_ready": False,
    }
