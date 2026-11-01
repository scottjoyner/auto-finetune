# LFM3-1.7B: measured M1 layer-10 direction stability — November 1, 2026

**Real x1-370 CPU measurement; no checkpoint edits, generation, human review, or release.**

## Source custody

The frozen source is A_original_posttrained with SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`. Source eight-pair dataset SHA-256 `ebdff35111f03617632a557793122fda05e9a235444c3dd0d6396f925bbf3d81`. Original fitted direction-file SHA-256 `7f31109d5212b4bb1f0839544935a694b9d4f4475e9c3a36926eebe28bd9437f`.

The on-host source-verification code checked these SHA-256 values before CPU model loading and again after the forward passes. Each run performed 16 last-token forward passes (benign and adversarial text for each of the original eight pairs) using the BF16 original A checkpoint on CPU. Both completed successfully, with runtimes of approximately 34.31 and 31.83 seconds. The offline aggregator used the source-pinned and model-free `src.calibration_direction_stability.audit`.

**Private per-pair activation data were not committed or shared.** The two local capture files have mode `0600` in a separate owner-only SSD directory. Both are exactly 417,933 bytes and have identical SHA-256:

`f6d05fd767ada9363dab9afa6076e740998556d5507a97ddf0d177cbc2b14429`

Byte-for-byte capture equality and aggregate equality were verified on the same host with the same settings, twice. No independent-hardware replication or additional sampling was performed.

## Actual measured aggregate

| Measurement | Result |
| --- | ---: |
| Reconstructed direction versus original direction cosine | 1.000000 |
| Unit-direction cosine between the two original calibration contrasts | **0.096468** |
| Mean signed held-out direction cosine | **0.361621** |
| Minimum / maximum signed held-out cosine | 0.095145 / 0.514260 |
| Positive held-out directional alignments | 6 / 6 |
| Mean absolute alignment from calibrator 1 alone | 0.232678 |
| Mean absolute alignment from calibrator 2 alone | 0.322262 |
| Orthogonal sham mean absolute alignment, seed 17 | 0.014478 |
| Orthogonal sham mean absolute alignment, seed 23 | 0.015566 |
| Orthogonal sham mean absolute alignment, seed 41 | 0.008591 |
| Orthogonal sham mean absolute alignment, seed 59 | 0.031386 |

Each of the four sham directions had smaller absolute alignment than the fitted direction on all six held-out examples. This is **descriptive**, not a randomized p-value: the sham directions were constructed orthogonal to the fitted direction, so they do not constitute an exchangeable null for a fitted semantic direction.

## Interpretation

**The calibration directions themselves agree poorly:** cosine approximately 0.096, i.e., nearly orthogonal as unit vectors. Nevertheless, the fitted direction is positively aligned with the held-out adversarial-minus-benign contrast for each of six existing examples. Reconstructing the original fitted direction exactly establishes code/source fidelity, not that the direction is a refusal-specific mechanism. This contrast may reflect general topic, wording, or semantic differences unrelated to refusal. The six held-outs are too few and not an independent population sample. We have not determined the influence of unequal calibration-delta magnitudes on the combined fitted direction.

Neither refusal safety, over-refusal, generation quality, harmful actionability, nor capability preservation was measured here. Independent blinded human grading of frozen 1,350 A/B/C responses remains outstanding. This result **does not authorize zero-refusal claims, weight edits, model deployment or publishing a checkpoint**.

## Storage and reproducibility safety

Fleet Commander connectivity was restored. Three read-only root-space samples initially reported **25.792 GiB** free; root research preflight accepted the existing gate of 20 GiB minimum versus the intended 5 GiB write + 8 GiB reserve. SSD had approximately 416.2 GiB free. After the first capture root was 25.791 GiB; after the second it briefly read 23.806 GiB, then four samples spaced three seconds apart returned to 25.791 GiB. The approximately 2 GiB transient discrepancy remains **unattributed**. No destructive recovery or global writer changes were made. Storage incident #106 remains open for investigation; no further inference was started after the transient observation.

Research code was staged from PR branch commit `951f29b` onto the SSD without changing existing root checkout. The on-host focused synthetic suites reported **26 passed**. The SHA-pinned aggregate is tracked separately in `evidence/20261101_direction_stability_actual.json`; only aggregate scores and source-identity hashes were uploaded.

## Follow-on experiment

Before fitting another direction, compare the original calibration-delta magnitudes, independently verify the cross-pair alignment under larger disjoint contrast sets, and test whether projected activations alter **human-rated** benign over-refusal without increasing unsafe actionability or losing canonical task capability. Keep sham controls distinct from a randomized null and retain the existing open research and storage gates.

