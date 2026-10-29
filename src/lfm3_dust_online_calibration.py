"""Causal online calibration between two activation-conditioned Dust heads.

Small parametric two-weight gate: train-on-past-scalar-feedback only, via an
exponential-weights (Hedge) update on each head's *prequential* squared error.
There is no backprop, LFM3 weight training, gradient, or optimizer step here.

Critical scientific limit: a small, nonstationary, same-seed pilot cannot show
that calibration generalizes. This algorithm chooses only which perturbation
directions get forward-tested; it never authorizes an adapter update itself.
"""
from __future__ import annotations
import math

from src.lfm3_dust_activation_head import (
    ActivationDirectionHead, _context,
)
from src.lfm3_dust_direction_attention import _feature

SCHEMA = "sensorgraft.lfm3.dust-online-head-calibration.v1"
ETA = 0.15
MIN_WEIGHT = 0.05


class CalibratedActivationHead(ActivationDirectionHead):
    def __init__(self, *, capacity=32):
        super().__init__(capacity=capacity)
        self.losses = [0., 0.]
        self.calibration_events = 0

    @property
    def weights(self):
        losses = [min(x, 1000.) for x in self.losses]
        shifted = [math.exp(-ETA * (x - min(losses))) for x in losses]
        total = sum(shifted)
        # Prevent a head from being permanently suppressed by sparse evidence.
        raw = [x / total for x in shifted]
        return tuple(MIN_WEIGHT + (1 - 2*MIN_WEIGHT) * x for x in raw)

    def _components(self, direction, context, *, episode):
        d, c = _feature(direction), _context(context)
        if type(episode) is not int or episode < 0:
            raise ValueError("invalid_prediction_episode")
        old = [m for m in self.memory if m.episode < episode]
        if not old:
            return (0., 0., 0)
        d_logits, c_logits, gains = [], [], []
        for m in old:
            ds = sum(x*y for x,y in zip(d,m.direction,strict=True))
            cs = sum(x*y for x,y in zip(c,m.context,strict=True))
            age = -.05*(episode-m.episode)
            d_logits.append(4*ds+age)
            c_logits.append(1.2*ds+3*cs+age)
            gains.append(m.gain)
        return (self._attend(d_logits,gains),
                self._attend(c_logits,gains),len(old))

    def predict(self,direction,context,*,episode):
        dh,ch,n=self._components(direction,context,episode=episode)
        w=self.weights
        return {
            "predicted_gain": w[0]*dh+w[1]*ch,
            "direction_head_gain": dh, "context_head_gain": ch,
            "head_disagreement": abs(dh-ch),
            "head_weights": list(w),
            "calibration_events": self.calibration_events,
            "evidence": n,
            "status": "online_calibrated_prior_train_attention" if n else "cold_start",
        }

    def observe(self,direction,context,gain,*,episode,split="train",measured=False):
        # Fail closed *before* doing a prediction or modifying calibration state.
        _feature(direction)
        _context(context)
        if (type(episode) is not int or episode<0 or episode<self.last_episode
                or split!="train" or measured is not True
                or type(gain) not in (int,float) or not math.isfinite(gain)
                or abs(gain)>1e4):
            raise ValueError("untrusted_online_calibration_label")
        pred=self.predict(direction,context,episode=episode)
        # Prediction made strictly from PREVIOUS episodes. A same-episode gain
        # can adjust future *gate* weights, but never retroactively re-rank
        # candidates already chosen in this episode.
        if pred["evidence"]>=2:
            scale=max(.01,abs(float(gain)))
            errors=[
                min(8., ((pred[name]-gain)/scale)**2)
                for name in ("direction_head_gain","context_head_gain")
            ]
            self.losses=[v+e for v,e in zip(self.losses,errors,strict=True)]
            self.calibration_events+=1
        result=super().observe(direction,context,gain,episode=episode,
                              split=split,measured=measured)
        return {**result,"calibration_events":self.calibration_events,
                "head_weights":list(self.weights),
                "optimizer_update_permitted":False,
                "weight_promotion_permitted":False}
