"""Tiny parametric cross-attention head learned by forward-only coordinate trials.

Two scaled-dot-product attention heads, each with LEARNABLE Q/K projection
matrices (4x12 each) and a learned null-value logit, plus a learned mixing
gate: 195 scalars total. All vectors come from 8-D perturbation coefficients
and 4-D prompt-only frozen LFM3 attention features. Value labels are the
previously measured training-only forward gains. No gradients, autograd,
backpropagation, torch optimizer or live model updates are used.

Head-parameter learning: two deterministic antithetic *head-only* proposals
are scored per episode on previous-episode train targets with strictly
earlier-episode keys/values. Accept only if historical loss improves and
the head's full parameter drift stays in a bounded trust radius. Every
current-episode forecast is then frozen BEFORE model perturbations.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass

from src.lfm3_dust_activation_head import _context
from src.lfm3_dust_direction_attention import _feature

SCHEMA="sensorgraft.lfm3.dust-parametric-cross-attention.v1"
HEADS=2
HEAD_WIDTH=4
INPUT_DIM=12
HEAD_PARAM_COUNT=2*HEADS*HEAD_WIDTH*INPUT_DIM + HEADS + 1
MEMORY_CAP=32
HEAD_SIGMA=.12
HEAD_TRUST=.6
HEAD_DRAWS=2


def _safe(values):
    if (not isinstance(values,(list,tuple)) or len(values)!=HEAD_PARAM_COUNT
            or any(type(x) not in (int,float) or not math.isfinite(x)
                   or abs(x)>10 for x in values)):
        raise ValueError("invalid_parametric_attention_head")
    return tuple(float(x) for x in values)


def _hash(params):
    return hashlib.sha256(json.dumps(params,separators=(",",":"),
                                      allow_nan=False).encode()).hexdigest()


def _input(direction,context):
    return _feature(direction)+_context(context)


@dataclass(frozen=True)
class TrainProbe:
    episode:int
    direction:tuple[float,...]
    context:tuple[float,...]
    gain:float


class ForwardOnlyNeuralAttentionHead:
    def __init__(self,*,seed):
        if type(seed) is not int or seed not in (7,8,9,10,11,12):
            raise ValueError("unregistered_head_seed")
        self.seed=seed
        r=random.Random(718883+seed)
        params=[]
        for _ in range(HEADS):
            params.extend(r.gauss(0,.15) for _ in range(2*HEAD_WIDTH*INPUT_DIM))
            params.append(3.4) # learned zero-valued null key bias
        params.append(0.)       # learned two-head mixture logit
        self.params=_safe(params)
        self.initial=self.params
        self.history=[]
        self.last_episode=-1
        self.model_backward_calls=0
        self.trial_head_forwards=0
        if len(self.params)!=HEAD_PARAM_COUNT:
            raise RuntimeError("wrong_neural_head_parameter_count")

    @property
    def parameter_sha256(self):
        return _hash(self.params)

    @property
    def update_norm(self):
        return math.sqrt(sum((x-y)**2 for x,y in zip(self.params,self.initial,strict=True)))

    @staticmethod
    def _project(weights,x):
        return [sum(w*x0 for w,x0 in zip(
            weights[i*INPUT_DIM:(i+1)*INPUT_DIM],x,strict=True))
            for i in range(HEAD_WIDTH)]

    def predict(self,direction,context,*,episode,params=None,history=None):
        if type(episode) is not int or episode<0:
            raise ValueError("invalid_preprobe_episode")
        q=_input(direction,context)
        weights=self.params if params is None else _safe(params)
        available=(self.history if history is None else history)
        visible=[h for h in available if h.episode<episode][-MEMORY_CAP:]
        if not visible:
            return {"predicted_gain":0.,"historical_keys":0,
                    "status":"cold_start_neural_head"}
        k=[_input(h.direction,h.context) for h in visible]
        offset=0
        outputs=[]
        for _ in range(HEADS):
            matrix_q=weights[offset:offset+HEAD_WIDTH*INPUT_DIM]
            offset+=HEAD_WIDTH*INPUT_DIM
            matrix_k=weights[offset:offset+HEAD_WIDTH*INPUT_DIM]
            offset+=HEAD_WIDTH*INPUT_DIM
            null_logit=weights[offset]
            offset+=1
            query=self._project(matrix_q,q)
            logits=[]
            for entry in k:
                key=self._project(matrix_k,entry)
                logits.append(sum(a*b for a,b in zip(query,key,strict=True))/math.sqrt(HEAD_WIDTH))
            center=max([null_logit]+logits)
            att=[math.exp(v-center) for v in logits]
            none=math.exp(null_logit-center)
            outputs.append(sum(w*h.gain for w,h in zip(att,visible,strict=True))/(sum(att)+none))
        gate=1/(1+math.exp(-weights[-1]))
        estimate=gate*outputs[0]+(1-gate)*outputs[1]
        if not math.isfinite(estimate) or abs(estimate)>100:
            raise RuntimeError("nonfinite_neural_attention_forecast")
        return {"predicted_gain":estimate,"historical_keys":len(visible),
                "status":"two_head_learned_causal_cross_attention"}

    def tune_from_previous(self,*,episode):
        """Only previous-episode labels, with keys from *earlier* episodes."""
        if type(episode) is not int or episode!=self.last_episode+1:
            raise ValueError("out_of_order_training_episode")
        sha_before=self.parameter_sha256
        if episode<2:
            return {"episode":episode,"head_proposals":0,
                    "head_only_forward_evaluations":0,
                    "head_updates_accepted":0,"parameter_sha_before":sha_before,
                    "parameter_sha_after":sha_before,
                    "head_parameter_l2_drift":self.update_norm,
                    "historical_training_cases":0,"model_backward_calls":0}
        training=[r for r in self.history if r.episode==episode-1]
        previous=[r for r in self.history if r.episode<episode-1]
        if len(training)!=8 or len(previous)<8:
            raise RuntimeError("missing_prior_episode_train_probes")
        def loss(params):
            errors=[]
            for item in training:
                forecast=self.predict(item.direction,item.context,
                                      episode=item.episode,params=params,
                                      history=previous)["predicted_gain"]
                errors.append((forecast-item.gain)**2)
            return sum(errors)/len(errors)
        old_loss=loss(self.params)
        forwards=1
        accepted=0
        generator=random.Random(103831+self.seed*100+episode)
        for _ in range(HEAD_DRAWS):
            d=[generator.gauss(0,1) for _ in range(HEAD_PARAM_COUNT)]
            n=math.sqrt(sum(v*v for v in d))
            directions=[HEAD_SIGMA*v/n for v in d]
            proposals=[]
            for sign in (1.,-1.):
                p=tuple(v+sign*dv for v,dv in zip(self.params,directions,strict=True))
                drift=math.sqrt(sum((a-b)**2 for a,b in zip(p,self.initial,strict=True)))
                if drift>HEAD_TRUST:
                    proposals.append((float("inf"),p))
                else:
                    proposals.append((loss(p),p))
                    forwards+=1
            best=min(proposals,key=lambda t:t[0])
            if best[0]+1e-12<old_loss:
                self.params=_safe(best[1])
                old_loss=best[0]
                accepted+=1
        if self.update_norm>HEAD_TRUST+1e-9:
            raise RuntimeError("head_trust_region_exceeded")
        self.trial_head_forwards+=forwards
        return {"episode":episode,"head_proposals":2*HEAD_DRAWS,
                "head_only_forward_evaluations":forwards,
                "head_updates_accepted":accepted,
                "parameter_sha_before":sha_before,
                "parameter_sha_after":self.parameter_sha256,
                "head_parameter_l2_drift":self.update_norm,
                "historical_training_cases":len(training),
                "model_backward_calls":0}

    def observe_episode(self,rows,*,episode,split="train",measured=False):
        if (type(episode) is not int or episode!=self.last_episode+1
                or split!="train" or measured is not True or not isinstance(rows,list)
                or len(rows)!=8):
            raise ValueError("untrusted_neural_head_training_outcomes")
        new=[]
        for entry in rows:
            if (not isinstance(entry,(tuple,list)) or len(entry)!=3
                    or type(entry[2]) not in (int,float) or not math.isfinite(entry[2])
                    or abs(entry[2])>100):
                raise ValueError("nonfinite_or_unmeasured_training_gain")
            new.append(TrainProbe(episode,_feature(entry[0]),_context(entry[1]),
                                  float(entry[2])))
        self.history.extend(new)
        del self.history[:-MEMORY_CAP]
        self.last_episode=episode
        return {"history_size":len(self.history),"head_weights_sha256":self.parameter_sha256,
                "model_backward_calls":0,"weight_promotion_permitted":False}
