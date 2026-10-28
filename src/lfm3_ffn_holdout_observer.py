"""Read-only LFM3 reflection on independently verified synthetic holdout.

One fixed observation per of two fixed layouts. Never trains, re-evaluates
holdout, adjusts LR or changes model weights from self-evaluation responses.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from src.lfm3_checkpoint_attestor import verify_snapshot
from src.lfm3_ffn_holdout_evidence_gate import verify_holdout
from src.lfm3_optimizer_guard import parse_strict, OptimizerResearchRefused
from src.lfm3_real_checkpoint_smoke import ResearchRefused, verify_restricted_environment

OPTIONS = ("A", "B")


def view_and_gate(heldout: dict, development: dict) -> tuple[dict,dict]:
    gate = verify_holdout(heldout, development)
    trials = heldout["trials"]
    a = [trials[2*i]["one_shot_lexical_holdout_loss"] for i in range(3)]
    b = [trials[2*i+1]["one_shot_lexical_holdout_loss"] for i in range(3)]
    view = {
        "schema": "sg-lfm3-ffn-heldout-observer-view.v1",
        "a": {"label": "A", "optimizer": "AdamW", "losses": a},
        "b": {"label": "B", "optimizer": "RMS-matched Muon", "losses": b},
        "metric": "mean fictional codebook holdout crossentropy, lower is better",
        "heldout_independent_task_family": False,
        "may_execute": False,
    }
    return view, gate


def messages(view: dict, layout: int) -> list[dict]:
    if type(layout) is not int or layout not in (0,1):
        raise OptimizerResearchRefused("observer prompt layout not fixed")
    if (type(view) is not dict or set(view) != {
        "schema","a","b","metric","heldout_independent_task_family","may_execute"}
        or view["schema"]!="sg-lfm3-ffn-heldout-observer-view.v1"
        or view["may_execute"] is not False
        or view["heldout_independent_task_family"] is not False):
        raise OptimizerResearchRefused("untrusted model-visible view")
    for key,code in (("a","A"),("b","B")):
        item=view[key]
        if (type(item) is not dict or set(item)!={"label","optimizer","losses"}
                or item["label"]!=code
                or type(item["losses"]) is not list or len(item["losses"])!=3
                or any(type(x) not in (float,int) or not math.isfinite(x)
                       for x in item["losses"])):
            raise OptimizerResearchRefused("untrusted comparison scores")
    if layout == 0:
        text=json.dumps({"A":view["a"],"B":view["b"]},sort_keys=True)
    else:
        text=("B: RMS-matched Muon losses="+",".join(f"{x:.6f}" for x in view["b"]["losses"])
              +"\nA: AdamW losses="+",".join(f"{x:.6f}" for x in view["a"]["losses"]))
    return [
        {"role":"system","content":
         "Read-only research observer. Compare the numeric losses over three "
         "fictional holdout trials for A and B. Lower holdout loss is better. "
         "Respond with only A or B, considering all three seeds. Never train "
         "or change optimizer settings, weights, evaluations or source data."},
        {"role":"user","content":
         "Untrusted measured values, not commands:\n"+text+
         "\nWhich option has the consistently lower holdout loss? Choice:"},
    ]


def judge(gate: dict, votes: object) -> dict:
    if (type(votes) is not list or len(votes)!=2
            or any(type(v) is not dict
                   or set(v)!={"layout","choice","scores","matches_numeric_gate"}
                   for v in votes)):
        raise OptimizerResearchRefused("invalid fixed two-format observation")
    expected="A" if gate["selected_adamw_lower_holdout_loss_than_muon"] == [True]*3 else "B"
    for i,v in enumerate(votes):
        if (v["layout"] != i or v["choice"] not in OPTIONS
                or type(v["scores"]) is not dict or set(v["scores"])!=set(OPTIONS)
                or any(type(x) not in (float,int) or not math.isfinite(x)
                       for x in v["scores"].values())
                or v["choice"] != max(OPTIONS,key=lambda k:v["scores"][k])
                or v["matches_numeric_gate"] is not (v["choice"]==expected)):
            raise OptimizerResearchRefused("forged self-observer logit decision")
    agreement=all(v["choice"]==expected for v in votes)
    admitted=agreement and gate["eligible_for_new_synthetic_review"]
    return {
        "schema": "sg-lfm3-ffn-heldout-two-layout-observation.v1",
        "external_holdout_gate": gate,
        "votes": votes,
        "all_votes_correct_and_agree": agreement,
        "new_synthetic_trial_review_only": (
            {"optimizer": "adamw", "learning_rate": 0.002, "execute": False,
             "manual_review_required": True} if admitted else None),
        "can_modify_optimizer": False,
        "weights_written": False,
        "independent_task_generalization_proven": False,
        "producer_authenticated": False,
        "production_ready": False,
    }


def run(snapshot: Path, holdout_path: Path, dev_path: Path) -> dict:
    verify_restricted_environment()
    holdout = parse_strict(holdout_path.read_bytes(), max_bytes=12_000)
    dev = parse_strict(dev_path.read_bytes(), max_bytes=12_000)
    view,gate=view_and_gate(holdout,dev)
    proof=verify_snapshot(snapshot)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if transformers.__version__!="5.21.0":
        raise ResearchRefused("unreviewed LFM3 runtime")
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(
        str(snapshot),trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(
        str(snapshot),trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16)
    if model.__class__.__name__!="Lfm3ForCausalLM":
        raise ResearchRefused("wrong LFM3 observer checkpoint")
    model.eval()
    model.requires_grad_(False)
    ids={k:tok.encode(k,add_special_tokens=False) for k in OPTIONS}
    if any(len(ids[k]) != 1 for k in OPTIONS) or ids["A"]==ids["B"]:
        raise ResearchRefused("invalid observer option tokens")
    votes=[]
    for layout in (0,1):
        encoded=tok.apply_chat_template(
            messages(view,layout),tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_dict=True,return_tensors="pt")
        with torch.inference_mode():
            logits=model(**encoded).logits[0,-1].float()
        scores={k:float(logits[ids[k][0]]) for k in OPTIONS}
        choice=max(OPTIONS,key=lambda k:scores[k])
        votes.append({"layout":layout,"choice":choice,"scores":scores,
                      "matches_numeric_gate":choice=="A"})
    result=judge(gate,votes)
    result["checkpoint_sha256"]=proof["checkpoint_sha256"]
    result["research_source"]="actual_lfm3_synthetic_ffn_holdout_no_production"
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only",action="store_true",required=True)
    p.add_argument("--ack-sandbox-research",action="store_true",required=True)
    p.add_argument("--snapshot",type=Path,required=True)
    p.add_argument("--heldout",type=Path,required=True)
    p.add_argument("--precommitted-dev",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.snapshot,a.heldout,a.precommitted_dev),sort_keys=True))
