"""Read-only task-shift detector for the trainable LFM3 Dust direction head.

The trust boundary is a frozen set of normalized 4-scalar LFM3 prompt-only
attention summaries from *TRAIN* examples. The nearest-neighbor cosine
threshold is derived from TRAIN leave-one-out similarities only, with a
preregistered .02 absolute margin. No unseen task contexts, probe gains,
selection/test labels or model training outcomes are used to fit the guard.

The historical head's prequential skill is also checked against always
predicting zero gain on prior training episodes. Failure of EITHER safeguard
means abstain and produce zero output. All methods are SHADOW-ONLY.
"""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA="sensorgraft.lfm3.dust-task-shift-gate.v1"
TRAIN_CONTEXTS=12
CALIBRATION_START_EPISODE=2
SHIFT_MARGIN=.02
MIN_GAIN_MSE_IMPROVEMENT=.05
PARAMETER_SCOPE="last_two_attention_vproj_lora_B_only"


def _unit_context(values):
    if (not isinstance(values,(tuple,list)) or len(values)!=4
            or any(type(x) not in (float,int) or not math.isfinite(x)
                   or abs(x)>1.00001 for x in values)):
        raise ValueError("invalid_lfm3_prompt_only_context")
    norm=math.sqrt(sum(float(x*x) for x in values))
    if not .9999<=norm<=1.0001:
        raise ValueError("prompt_context_not_unit_normalized")
    return tuple(float(x)/norm for x in values)


def _cosine(x,y):
    return sum(a*b for a,b in zip(x,y,strict=True))


def _hash(contexts):
    return hashlib.sha256(json.dumps(contexts,separators=(",",":"),
                                     allow_nan=False).encode()).hexdigest()


class FrozenTaskShiftGate:
    def __init__(self,*,source):
        if (not isinstance(source,dict) or source.get("seed") not in (10,11,12)
                or type(source.get("seed")) is not int
                or source.get("head_parameters")!=195
                or source.get("train_episodes")!=TRAIN_CONTEXTS
                or source.get("transfer_family_feedback_used_for_training") is not False
                or source.get("model_adapter_update_count")!=0
                or source.get("head_frozen_before_transfer_sha256")!=source.get("head_final_sha256")):
            raise ValueError("untrusted_head_training_source")
        rows=source.get("events")
        if not isinstance(rows,list) or len(rows)!=18:
            raise ValueError("incomplete_prior_neural_training")
        train=rows[:TRAIN_CONTEXTS]
        if any(row.get("phase")!="train" for row in train):
            raise ValueError("untrusted_training_split")
        contexts=[_unit_context(row["context"]) for row in train]
        # Each calibration point is compared to *other training examples*,
        # never itself or any already-seen transfer example.
        lno=[max(_cosine(c,other) for j,other in enumerate(contexts) if j!=i)
             for i,c in enumerate(contexts)]
        self.threshold=max(-1.,min(1.,min(lno)-SHIFT_MARGIN))
        self.train_contexts=tuple(contexts)
        self.seed=source["seed"]
        self.training_digest=_hash(contexts)
        self.training_loo_nearest=tuple(lno)
        self.calibration_sample_count=len(lno)
        self.min_prior_train_nearest=min(lno)
        pairs=[
            (entry["predictions"]["neural"],entry["measured_gain"])
            for row in train[CALIBRATION_START_EPISODE:]
            for entry in row["signed_directions"]
        ]
        if len(pairs)!=80:
            raise ValueError("missing_train_only_head_prediction_evidence")
        if any(type(x) not in (float,int) or not math.isfinite(x) or abs(x)>100
               for p,g in pairs for x in (p,g)):
            raise ValueError("bad_training_only_head_gains")
        self.zero_mse=sum(float(g*g) for _,g in pairs)/len(pairs)
        self.head_mse=sum(float((p-g)**2) for p,g in pairs)/len(pairs)
        self.verified_skill=(self.zero_mse>0 and
            self.head_mse<=.95*self.zero_mse)
        self.train_feedback_sample_count=len(pairs)

    def evaluate(self,context,raw_head_prediction,*,candidate_index):
        if (type(candidate_index) is not int or candidate_index not in range(8)
                or type(raw_head_prediction) not in (float,int)
                or not math.isfinite(raw_head_prediction)
                or abs(raw_head_prediction)>100):
            raise ValueError("untrusted_shadow_candidate_prediction")
        query=_unit_context(context)
        nearest=max(_cosine(query,c) for c in self.train_contexts)
        in_distribution=nearest>=self.threshold
        admitted=in_distribution and self.verified_skill
        return {
            "candidate_index":candidate_index,
            "raw_head_prediction":float(raw_head_prediction),
            "guarded_head_prediction":float(raw_head_prediction) if admitted else 0.,
            "nearest_train_context_cosine":nearest,
            "train_only_similarity_threshold":self.threshold,
            "in_training_context_envelope":in_distribution,
            "historical_train_head_beats_zero_by_5pct":self.verified_skill,
            "abstained":not admitted,
            "abstention_reason":(
                "novel_activation_context" if not in_distribution
                else "insufficient_train_only_gain_prediction_skill"
                if not self.verified_skill else None),
            "candidate_measured_gain":None,
            "selected_for_live_probe":False,
            "model_update_permitted":False,
            "optimizer_switch_permitted":False,
            "weight_promotion_permitted":False,
        }

    def summary(self):
        return {
            "schema":SCHEMA,
            "source_seed":self.seed,
            "train_only_context_sha256":self.training_digest,
            "train_contexts":len(self.train_contexts),
            "training_loo_min_nearest":self.min_prior_train_nearest,
            "train_only_similarity_threshold":self.threshold,
            "context_threshold_margin":SHIFT_MARGIN,
            "head_prequential_training_mse":self.head_mse,
            "zero_prequential_training_mse":self.zero_mse,
            "head_training_gain_skill_pass":self.verified_skill,
            "prequential_signed_direction_train_cases":self.train_feedback_sample_count,
            "source_transfer_episodes_used_to_fit_gate":0,
            "optimizer_switch_permitted":False,
            "weight_promotion_permitted":False,
        }
