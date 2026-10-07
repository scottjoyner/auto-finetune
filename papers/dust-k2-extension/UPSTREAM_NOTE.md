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

One important difference from Dust is that our current orthogonal estimator
shares a perturbation direction across token positions within a draw, whereas
Dust uses independent per-token activation noise. We are planning that parity
experiment next; because our current trainable site is the final attention
`o_proj`, the downstream tail is position-local and gives us a clean way to
test the token-as-virtual-population idea before attempting earlier-layer
future-token credit.

We have drafted an arXiv-style independent technical report and can provide a
public-safe reproduction package with synthetic fixtures, estimator code,
manifest schemas, and aggregate result tables. No private training content is
needed for that package.

We would value feedback on whether this extension is useful to your research
agenda, especially on:

1. the most faithful tokenwise estimator comparison to prioritize;
2. whether you have a preferred way to report population/forward cost for
   adapter-tuning experiments;
3. any implementation details from the full experiments that would affect a
   fair systems comparison.

We are happy to keep this as an independent replication/extension, or discuss
deeper collaboration if the results are useful and there is mutual interest.
