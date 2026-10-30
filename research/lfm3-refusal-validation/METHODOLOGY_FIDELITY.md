# Mechanical integrity and methodology fidelity — frozen three-checkpoint study

Research evidence dated October 29, 2026. This is a read-only audit
of the existing A/B/C model files, not a training workflow or
proof of behavioral refusal removal.

## The fixed arms
A: original post-trained LFM3-1.7B.
B: A derivative, M1 rank-one residual-output projection.
C: original LFM3-1.7B-Base, which is not a causal control for undoing
post-training. No new checkpoint is created.

## Verified mechanical findings
- All original and edited model / direction SHA-256 identities match.
- Original inventory: 286 weight tensors.
- 48/48 targeted residual-writing matrices match the exact edit formula.
- 238/238 non-targeted tensors are unchanged.
- Direction fit: 2 contrastive calibration pairs only.
- Prior activation audit: 4 disjoint heldout prompt pairs.
- Edited/original directional-projection RMS ratio was below 1 in
  22 of 23 measured layers; median 0.801448, range 0.224911–1.049562.
- These are structural measurements, NOT a validated refusal,
  harmful-actionability, or safety score.

Public sanitized evidence:
evidence/20261029_abliteration_method_fidelity.json

Read-only local reproduction (model files remain private):

    python -m src.abliteration_method_fidelity_audit --output evidence/new_local_method_fidelity.json

The auditor refuses to overwrite previous evidence. Model weights and
the disjoint heldout private prompt measurements are never published.

## Relation to canonical abliteration
Arditi et al., "Refusal in Language Models Is Mediated by a Single
Direction" (NeurIPS 2024, arXiv:2406.11717), derive candidate
directions from mean harmful/harmless activation contrasts and
select them using separate validation. The M1 checkpoint uses a
corresponding rank-one projection but does NOT replicate the
full published selection and heldout validation protocol.

This work has just two calibration pairs, no robust independent
layer/position candidate selection with calibration uncertainty,
and LFM3 sandwich normalization after edited writers may restore
components of the supposedly removed direction. No human-reviewed
behavioral refusal/actionability outcome has been observed yet.
The capability benchmark uses custom likelihood scoring, not the
official ARC scoring harness.

The correct claim is that B is an exact mechanically verified M1-style
derivative of A, not that it is a validated canonical reproduction
or that it reliably suppresses refusal across unseen requests.

## Release gates
Full XSTest 1,350/1,350 generations and full custom ARC likelihood
3,516/3,516 case/checkpoint scores are complete. Two independent
human reviewer packets remain ungraded (zero of 1,350 each).
Standardized capability confirmation, safety conclusions, and
responsible release/publication approval are pending.
No weights are released, and the three-checkpoint lineup is frozen.

Reference: https://arxiv.org/abs/2406.11717.
