# Draft note to the Dust authors

Subject: Independent K2-Horizon adapter-tuning extension of Dust-inspired activation perturbation

We have been running an independent set of experiments motivated by your Dust
work, focused on a different regime: **forward-only LoRA tuning of a frozen
pretrained 0.9B decoder model** rather than full transformer pretraining.

We want to share the results because several of them appear supportive of the
broader activation-space zeroth-order thesis, while also exposing useful
limitations.

The strongest findings so far:

- On a 1536-wide final attention output projection, increasing a structured
  antithetic perturbation population substantially improves alignment with
  exact local autograd. A complete 1536-direction orthogonal basis reaches
  LoRA-B gradient cosine ~0.981.
- After two actual adapter updates, the complete-basis forward-only trajectory
  reaches approximately **0.998 / 0.991 cosine** to matched backprop for LoRA
  A/B updates.
- A systems optimization that caches the frozen transformer prefix and replays
  only the final downstream tail reduces our K=1024 four-step implementation
  from roughly **347 s to 15 s** on average while preserving the estimator.
  In this tiny matched protocol that is about **2.1x** the elapsed time of
  adapter-only backprop rather than tens of times slower.
- We reproduced the key equivalence checks on the real pretrained
  K2-Horizon-0.9B weights and have also run the estimator on AMD ROCm.

The quality results are not yet positive overall: average heldout CE worsens
slightly in the short protocol for both the forward-only treatment and the
matched backprop control. We are treating that as a useful negative result,
not hiding it.

One important difference from Dust is that our strongest structured estimator
shares a perturbation direction across token positions within a draw, whereas
Dust uses independent per-token activation noise. We have now completed that
parity experiment for the final attention `o_proj`. Independent per-token
Gaussian antithetic noise is viable and improves with population, but at
K=256 and K=1024 it is less aligned with exact local autograd than the shared
orthogonal control at essentially the same tail runtime. At K=1024 and
sigma=.25, mean A/B gradient cosine is approximately **.502/.627** for
tokenwise Gaussian versus **.728/.812** for shared orthogonal directions over
three seeds. We interpret this as a property of this deliberately
position-local final-tail setting, not as a contradiction of Dust's broader
attention/future-credit regime.

We have drafted an arXiv-style independent technical report and can provide a
public-safe reproduction package with synthetic fixtures, estimator code,
manifest schemas, and aggregate result tables. No private training content is
needed for that package.

We would value feedback on whether this extension is useful to your research
agenda, especially on:

1. whether you would expect tokenwise independence to show its main benefit
   only once perturbations participate in attention/future-token credit rather
   than in a position-local final tail;
2. whether you have a preferred way to report population/forward cost for
   adapter-tuning experiments;
3. any implementation details from the full experiments that would affect a
   fair systems comparison.

We are happy to keep this as an independent replication/extension, or discuss
deeper collaboration if the results are useful and there is mutual interest.
