"""Activation-conditioned, causal two-head Dust predictor (no backprop).

Reads a bounded four-scalar *pre-perturbation, prompt-only* summary from a
frozen LFM3 attention projection. Does not store tokens or raw activations.
The two deterministic attention heads use previous train-only direction gains:
one weights direction similarity, one weights context/direction similarity.
All predictions are advisory; a real forward probe must validate a direction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from src.lfm3_dust_direction_attention import DIM, CAPACITY, _feature

CONTEXT_DIM = 4
MAX_HISTORY = CAPACITY
SCHEMA = "sensorgraft.lfm3.dust-activation-conditioned-head.v1"


def _context(values):
    if (type(values) not in (tuple, list) or len(values) != CONTEXT_DIM
            or any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 10
                   for v in values)):
        raise ValueError("invalid_prompt_only_activation_summary")
    norm = math.sqrt(sum(float(v) ** 2 for v in values))
    if norm <= 1e-8:
        raise ValueError("empty_activation_context")
    return tuple(float(v) / norm for v in values)


@dataclass(frozen=True)
class PastProbe:
    episode: int
    direction: tuple[float, ...]
    context: tuple[float, ...]
    gain: float


class ActivationDirectionHead:
    def __init__(self, *, capacity=MAX_HISTORY):
        if type(capacity) is not int or not 1 <= capacity <= MAX_HISTORY:
            raise ValueError("invalid_attention_capacity")
        self.capacity = capacity
        self.memory: list[PastProbe] = []
        self.last_episode = -1

    @property
    def size(self):
        return len(self.memory)

    @staticmethod
    def _attend(logits, gains):
        maximum = max(logits)
        weights = [math.exp(x - maximum) for x in logits]
        return sum(w * g for w, g in zip(weights, gains, strict=True)) / sum(weights)

    def predict(self, direction, context, *, episode):
        qd, qc = _feature(direction), _context(context)
        if type(episode) is not int or episode < 0:
            raise ValueError("invalid_episode")
        old = [x for x in self.memory if x.episode < episode]
        if not old:
            return {"predicted_gain": 0., "evidence": 0, "head_disagreement": 0.,
                    "status": "cold_start"}
        direction_logits = []
        contextual_logits = []
        gains = []
        for x in old:
            d = sum(a*b for a,b in zip(qd, x.direction, strict=True))
            c = sum(a*b for a,b in zip(qc, x.context, strict=True))
            age = -0.05 * (episode - x.episode)
            direction_logits.append(4.0*d + age)
            contextual_logits.append(1.2*d + 3.0*c + age)
            gains.append(x.gain)
        by_direction = self._attend(direction_logits, gains)
        by_context = self._attend(contextual_logits, gains)
        estimate = (by_direction + by_context) / 2
        if not all(math.isfinite(v) for v in (estimate, by_direction, by_context)):
            raise ValueError("nonfinite_attention_feedback")
        return {"predicted_gain": estimate, "evidence": len(old),
                "head_disagreement": abs(by_direction - by_context),
                "status": "prior_train_probe_two_head_attention"}

    def rank(self, candidates, context, *, episode):
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
            raise ValueError("invalid_candidates")
        scores = [self.predict(x, context, episode=episode) for x in candidates]
        return sorted(range(len(scores)), key=lambda i: (-scores[i]["predicted_gain"], i))

    def observe(self, direction, context, gain, *, episode, split="train", measured=False):
        d, c = _feature(direction), _context(context)
        if (type(episode) is not int or episode < 0
                or episode < self.last_episode or split != "train"
                or measured is not True or type(gain) not in (float, int)
                or not math.isfinite(gain) or abs(gain) > 1e4):
            raise ValueError("untrusted_forward_probe_feedback")
        self.last_episode = episode
        self.memory.append(PastProbe(episode, d, c, float(gain)))
        del self.memory[:-self.capacity]
        return {"status": "train_probe_only_attended_feedback",
                "memory_size": self.size, "backward_calls": 0,
                "optimizer_update_permitted": False, "weight_promotion_permitted": False}


def prompt_only_activation_context(torch, model, tokenizer, prompt):
    """Four aggregate scalars, captured during a prompt-only forward pass.

    The summary is computed in the hook and raw activations are never returned
    to the caller or persisted. This costs one extra full-model forward.
    """
    if type(prompt) is not str or not 1 <= len(prompt) <= 256:
        raise ValueError("invalid_prompt")
    layers = [(name, mod) for name, mod in model.named_modules()
              if name.endswith(".self_attn.v_proj")]
    if len(layers) < 2:
        raise RuntimeError("unavailable_lfm3_attention_projection")
    target = max(layers, key=lambda pair: int(pair[0].split(".layers.")[1].split(".")[0]))[1]
    scalars = []

    def capture(_module, args):
        if not args or not hasattr(args[0], "shape"):
            raise RuntimeError("missing_projection_input")
        x = args[0].detach().float()
        if x.ndim != 3 or x.shape[-1] < 64:
            raise RuntimeError("unexpected_projection_input_shape")
        norm = x.square().mean().sqrt().clamp_min(1e-6)
        chunks = torch.chunk(x, CONTEXT_DIM, dim=-1)
        if len(chunks) != CONTEXT_DIM:
            raise RuntimeError("unexpected_context_chunk_count")
        scalars[:] = [float((part.mean() / norm).clamp(-10,10)) for part in chunks]

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not 1 <= len(token_ids) <= 64:
        raise ValueError("prompt_context_length_out_of_bounds")
    handle = target.register_forward_pre_hook(capture)
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([token_ids], dtype=torch.long), use_cache=False)
    finally:
        handle.remove()
    if len(scalars) != CONTEXT_DIM:
        raise RuntimeError("activation_summary_not_captured")
    return _context(scalars)
