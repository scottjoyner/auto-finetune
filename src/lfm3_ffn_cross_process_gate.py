"""Independent cross-process frozen LFM3 FFN optimizer checkpoint assessment.

Reads only two committed research receipts and a precommitted development
comparison; this module cannot train, load model weights, write checkpoint,
change optimizer settings or promote models.
"""
from __future__ import annotations

import math
from src.lfm3_ffn_multiseed_guard import admit_three_seed_training
from src.lfm3_real_checkpoint_smoke import ResearchRefused

PIN = "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330"
SCHEMA = "sg-lfm3-native-ffn-cross-process-verified.v1"
STEP_SCHEMA = "sg-lfm3-ffn-native-optimizer-process-restart.v1"
MODES = ("adamw", "muon_match_rms_adamw")
TOP_KEYS = {
    "schema", "optimizer", "baseline", "save", "resume",
    "exact_cross_process_state", "power_loss_tested", "production_ready",
    "experiment_date", "node", "source_commit",
    "evidence_is_research_only", "independent_power_loss_verification",
}
PHASE_KEYS = {
    "schema", "mode", "optimizer", "snapshot_sha256", "seed",
    "steps_completed", "dev_loss", "adapter_optimizer_canonical_sha256",
    "manifest_payload_sha256", "cross_process_replay_verified",
    "source_authenticated", "fictional_only", "original_backbone_modified",
    "network_disabled_required", "production_ready",
}


def _hex(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def assess_cross_process(reports: object, development: object) -> dict:
    evidence = admit_three_seed_training(development)
    if (not evidence["may_propose_next_synthetic_review_only"]
            or evidence["fixed_objective_best"] != "A"):
        raise ResearchRefused("development-only source cannot authorize this test")
    if type(reports) is not list or len(reports) != 2:
        raise ResearchRefused("need exactly two original optimizer replay packets")
    rows=[]
    for slot,optimizer in enumerate(MODES):
        pkt=reports[slot]
        if (type(pkt) is not dict or set(pkt)!=TOP_KEYS
                or pkt["schema"]!=SCHEMA or pkt["optimizer"]!=optimizer
                or pkt["exact_cross_process_state"] is not True
                or pkt["power_loss_tested"] is not False
                or pkt["production_ready"] is not False
                or pkt["evidence_is_research_only"] is not True
                or pkt["independent_power_loss_verification"] is not False
                or pkt["experiment_date"] != "2026-10-28"
                or pkt["node"] != "xwing"
                or pkt["source_commit"] != "20c5d02651a8868cfca1533ebc664ff90bc9c17e"):
            raise ResearchRefused("forged durable restart provenance/authority")
        stages=[]
        for name,steps in (("baseline",3),("save",1),("resume",3)):
            stage=pkt[name]
            if (type(stage) is not dict or set(stage)!=PHASE_KEYS
                    or stage["schema"]!=STEP_SCHEMA
                    or stage["mode"]!=name or stage["optimizer"]!=optimizer
                    or stage["snapshot_sha256"]!=PIN
                    or type(stage["seed"]) is not int or stage["seed"]!=20261028
                    or type(stage["steps_completed"]) is not int
                    or stage["steps_completed"]!=steps
                    or type(stage["dev_loss"]) not in (float,int)
                    or not math.isfinite(stage["dev_loss"])
                    or stage["dev_loss"] < 0
                    or not _hex(stage["adapter_optimizer_canonical_sha256"])
                    or stage["cross_process_replay_verified"] is not False
                    or stage["source_authenticated"] is not False
                    or stage["fictional_only"] is not True
                    or stage["original_backbone_modified"] is not False
                    or stage["network_disabled_required"] is not True
                    or stage["production_ready"] is not False):
                raise ResearchRefused("wrong restart stage or elevated authority")
            digest=stage["manifest_payload_sha256"]
            if (name=="baseline" and digest is not None
                    or name!="baseline" and not _hex(digest)):
                raise ResearchRefused("checkpoint evidence missing or inconsistent")
            stages.append(stage)
        baseline,saved,resumed=stages
        if (baseline["dev_loss"] != resumed["dev_loss"]
                or baseline["adapter_optimizer_canonical_sha256"] !=
                   resumed["adapter_optimizer_canonical_sha256"]
                or saved["manifest_payload_sha256"] != resumed["manifest_payload_sha256"]):
            raise ResearchRefused("actual independent process optimizer replay diverged")
        expected=development["arms"][slot]["dev_loss_after"]
        if baseline["dev_loss"] != expected:
            raise ResearchRefused("cross-process result disagrees with precommitted development")
        rows.append({
            "optimizer":optimizer,
            "dev_loss":baseline["dev_loss"],
            "final_state_sha256":baseline["adapter_optimizer_canonical_sha256"],
            "scratch_checkpoint_sha256":saved["manifest_payload_sha256"],
            "restarted_process_matches_baseline":True,
            "power_loss_verified":False,
        })
    return {
        "schema":"sg-lfm3-ffn-optimizer-process-restart-gate.v1",
        "real_model_bf16_checkpoint":PIN,
        "real_ffn_adapter_seed":20261028,
        "native_optimizers":list(MODES),
        "restarted_processes_exact_both":True,
        "results":rows,
        "power_loss_verified":False,
        "can_authorize_training":False,
        "can_auto_adjust_optimizer":False,
        "may_promote_model":False,
        "production_ready":False,
    }
