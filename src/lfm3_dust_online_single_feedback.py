"""Causal, derivative-free, ONE feedback observation per LFM3 Dust episode.

The preexisting 195-weight LFM3 Q/K prediction-attention head historically
required 8 measured signed-direction gains per training episode. This
controller supports the operationally cheaper one-probe budget: it observes
ONLY the chosen neural-head direction's *train-prompt* measured gain and
uses it to try bounded +/- head-only parameter changes at the NEXT episode.
The previous probe's forecast is always evaluated against the head memory
as it stood BEFORE its outcome was observed. No diagnostics, test labels,
future samples, model gradients or model-adapter training are used.

This trains the small EXTERNAL head in an offline research sandbox; it
NEVER grants authority to choose live/production LFM3 model updates.
"""
from __future__ import annotations

import math
import random

from src.lfm3_dust_trainable_attention import (
    TrainProbe,HEAD_PARAM_COUNT,HEAD_TRUST,_safe,
)
from src.lfm3_dust_direction_attention import _feature
from src.lfm3_dust_activation_head import _context

SCHEMA="sensorgraft.lfm3.dust-causal-one-feedback-head.v1"
SINGLE_HEAD_DRAWS=2
SINGLE_HEAD_SIGMA=.12


class OnlineSingleFeedbackHead:
    def __init__(self,head,*,seed):
        if type(seed) is not int or seed not in (28,29,30):
            raise ValueError("unregistered_online_feedback_seed")
        if len(head.params)!=HEAD_PARAM_COUNT or head.last_episode!=11:
            raise ValueError("invalid_source_pretrained_head")
        self.head=head
        self.seed=seed
        self.pending=None
        self.last_prepared=11
        self.last_observed=11
        self.total_head_only_objective_forwards=0
        self.total_head_accepted_updates=0

    def prepare(self,*,episode,adaptive=True):
        if (type(episode) is not int or episode!=self.last_prepared+1
                or type(adaptive) is not bool
                or self.last_observed!=episode-1):
            raise ValueError("out_of_order_or_unmeasured_episode")
        original_sha=self.head.parameter_sha256
        original_l2=self.head.update_norm
        counted=0
        accepted=0
        old_err=None
        self.last_prepared=episode
        if self.pending is not None:
            p=self.pending
            if p["episode"]!=episode-1:
                raise RuntimeError("noncausal_prior_probe")
            def objective(params):
                out=self.head.predict(
                    p["direction"],p["context"],episode=p["episode"],
                    params=params,history=p["history_before"])["predicted_gain"]
                return (out-p["gain"])**2
            old_err=objective(self.head.params)
            counted+=1
            if adaptive:
                gen=random.Random(20261030+self.seed*100+episode)
                for _ in range(SINGLE_HEAD_DRAWS):
                    raw=[gen.gauss(0,1) for _ in range(HEAD_PARAM_COUNT)]
                    norm=math.sqrt(sum(x*x for x in raw))
                    dv=[SINGLE_HEAD_SIGMA*x/norm for x in raw]
                    choices=[]
                    for sign in (1.,-1.):
                        next_params=tuple(a+sign*b for a,b in zip(
                            self.head.params,dv,strict=True))
                        drift=math.sqrt(sum((a-b)**2 for a,b in zip(
                            next_params,self.head.initial,strict=True)))
                        if drift>HEAD_TRUST:
                            continue
                        err=objective(next_params)
                        choices.append((err,next_params))
                        counted+=1
                    if choices:
                        best=min(choices,key=lambda x:x[0])
                        if best[0]+1e-12<old_err:
                            self.head.params=_safe(best[1])
                            old_err=best[0]
                            accepted+=1
            self.pending=None
        if self.head.update_norm>HEAD_TRUST+1e-9:
            raise RuntimeError("untrusted_neural_head_trust_radius")
        self.total_head_only_objective_forwards+=counted
        self.total_head_accepted_updates+=accepted
        return {
            "schema":SCHEMA,"episode":episode,
            "adaptive":adaptive,
            "previous_train_probes_consumed":0 if old_err is None else 1,
            "head_only_objective_forwards":counted,
            "head_only_accepted_updates":accepted,
            "head_sha_before":original_sha,
            "head_sha_after":self.head.parameter_sha256,
            "head_drift_before":original_l2,
            "head_drift_after":self.head.update_norm,
            "model_backwards":0,
            "model_adapter_updates":0,
            "uses_current_episode_labels":False,
            "uses_diagnostic_labels":False,
            "model_weight_promotion_permitted":False,
        }

    def forecast(self,directions,context,*,episode):
        if (type(episode) is not int or episode!=self.last_prepared
                or self.last_observed>=episode
                or not isinstance(directions,list) or len(directions)!=8):
            raise ValueError("unprepared_or_repeated_neural_forecast")
        x=_context(context)
        result=[]
        for d in directions:
            result.append(self.head.predict(_feature(d),x,episode=episode)[
                "predicted_gain"])
        return result

    def observe(self,direction,context,measured_gain,*,episode):
        if (type(episode) is not int or episode!=self.last_prepared
                or episode!=self.last_observed+1
                or type(measured_gain) not in (int,float)
                or not math.isfinite(measured_gain)
                or abs(measured_gain)>100 or self.pending is not None):
            raise ValueError("noncausal_or_unsafe_single_feedback")
        d=_feature(direction)
        c=_context(context)
        record=TrainProbe(episode,d,c,float(measured_gain))
        self.pending={
            "episode":episode,"direction":d,"context":c,
            "gain":float(measured_gain),
            "history_before":tuple(self.head.history),
        }
        self.head.history.append(record)
        del self.head.history[:-32]
        self.head.last_episode=episode
        self.last_observed=episode
        return {
            "schema":SCHEMA,"episode":episode,
            "new_observations":1,
            "gain":float(measured_gain),
            "head_memory_size":len(self.head.history),
            "head_sha256":self.head.parameter_sha256,
            "model_adapter_updates":0,
            "uses_diagnostic_labels":False,
        }
