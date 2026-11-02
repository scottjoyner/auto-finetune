# LFM3-1.7B M1 direction stability: reproducible frozen-calibration audit (2026-11-01)

**Research only. This is an evaluation design, not a new empirical result.**
The previous M1 B checkpoint used a single layer-10 residual direction fitted on only two prompt pairs, then projected 48 residual-writing matrices. A mean of two examples is not evidence the direction is repeatable, causal, or specific to refusal. The original frozen A/B/C checkpoints and all previous generated response journals remain unchanged.

## New implementation

- `src/capture_calibration_deltas.py`: optional **operator-invoked, CPU forward-pass-only** capture of eight existing frozen paired activation contrasts from original checkpoint **A**. Never runs model generation or weight editing. Verifies exact SHA-256 identities of A model weights, eight-pair input JSONL, and original fitted direction file, both **before and after** the capture. Checks the research write-capacity gate **before loading the model** and again before the private output. It rejects any deviation from the two calibration pairs p01/p02 and six held-out pairs p03–p08.
- `src/calibration_direction_stability.py`: deterministic **offline** auditor of that private capture. Reconstructs the original fitted direction from the **unnormalized** p01/p02 deltas and fails if cosine agreement with frozen source direction is below 0.9995. Computes the p01–p02 directional agreement (two one-example leave-one-out fits), six held-out alignment scores, and matched **absolute** alignment against four seeded sham directions, orthogonal to the frozen fitted direction. The output includes only aggregate statistics, fixed public SHA identifiers, and explicit limitations; no prompts, completions, raw vectors or per-pair results.
- Two synthetic-only regression suites: cancellation, anti-alignment, source hash tampering, mismatched direction, wrong model/layer/pair roster, non-finite tensors, zero vectors, repeatability, private file permissions, blocked root storage, exclusive report creation, and no model loads when the capacity preflight refuses.

## Operation

**Do not execute capture while storage incident #106 remains unresolved.** The existing `research_job_guard.require_research_write_capacity` checks both the local work root and the output filesystem, with a 20 GiB floor plus planned 5 GiB writes and 8 GiB reserve. Earlier ~20.7 GiB root-free readings would fail its conservative gate. Fleet Commander was unreachable while this code was authored; no live root headroom was verified.

After the host is recovered and independently verified, use a dedicated private directory on a verified SSD (0700), and keep both artifacts **local and private**:

```bash
cd research/lfm3-refusal-validation
# Operator only, after root and SSD admission checks:
python -m src.capture_calibration_deltas \
  --private-out /verified-ssd/private-lfm3/contrast_deltas_v1.json

# Offline, model-free aggregate geometry audit:
python -m src.calibration_direction_stability \
  --private-vectors /verified-ssd/private-lfm3/contrast_deltas_v1.json \
  --output /verified-ssd/private-lfm3/contrast_stability_v1.json

# Tests on a disposable CPU-only environment:
python -m pytest -o addopts= -q \
  tests/test_calibration_direction_stability.py \
  tests/test_capture_calibration_deltas.py
```

Both generated reports use exclusive create and 0600 permissions; source directories must be owner-only. The *raw per-pair activation delta file must never be committed or uploaded*. Check the aggregate output for inadvertent private details before any research publication.

## Readout and interpretation

Interpret the metrics as independent questions:

1. **Source fidelity:** the direction reconstructed from the two frozen calibration deltas matches the previously fitted source direction.
2. **Calibration agreement:** cosine(p01 direction, p02 direction) tests whether the two original fitting examples agree at all; opposing directions can cancel.
3. **Held-out alignment:** the fitted direction aligns with the six unused prompt-pair contrast directions; the report includes signed mean/min/max and absolute mean, so anti-alignment cannot be concealed.
4. **Single-calibrator sensitivity:** separately compare held-out absolute alignment against p01-only and p02-only unit directions (leave-one-calibrator-out diagnostic).
5. **Orthogonal sham controls:** four fixed seeds 17/23/41/59 report descriptive alignment and pair counts. Since these sham directions are explicitly orthogonal to fit, they are **not an exchangeable randomization null**; there are no valid causality or p-value claims in this audit.

The capture is an additional 16 last-token CPU forward passes on original checkpoint A, not new response generations or model updates. The original dataset has only six held-out pairs, and contrast direction is *not the same as actual refusal/utility behavior*. Larger independently chosen calibration sets, blinded human ratings, and canonical capability retention still gate any new checkpoint experiment or release.

**Current state:** analyzer and capture commands implemented; only synthetic tests executed locally while Fleet Commander was unavailable. The live LFM3 direction stability numbers are **not known**. No claim of refusal removal, successful abliteration, checkpoint publication or safety validation is made. Keep PR #103 draft and storage incident #106 open.
