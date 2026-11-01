# LFM3-1.7B paired XSTest activation direction audit — 2026-11-01

**Actual on-host measurement, not model refusal behavior or safety validation.** The existing A/B/C checkpoints and M1 weight edits were unchanged.

## Why matched controls

Earlier eight-pair activation capture showed poor agreement between the two source calibrators (cosine **0.096468**). A separate 54-prompt XSTest activation probe found the fitted M1 direction separates the unsafe and safe source groups, but broad XSTest groups have topic and length differences.

The new experiment selects **18 source-index-matched safe/unsafe XSTest pairs**, exactly three per family: definitions, figurative language, historical events, homonyms, safe contexts and safe targets. Within each family, XSTest authors structured paired contrast rows at corresponding source positions. The selected rows were inspected for a shared focus, category and plausible semantic relationship. Source CSV and checkpoint file SHA-256 are verified before and after the 36 last-token CPU forward passes. Sampling is fixed by seed **9418**. Privacy/discrimination families were deliberately excluded because their source row indexes do not consistently imply semantic matching.

**Caveat:** family selection followed examination of prior activation results, so it is post hoc and not independently preregistered. The corpus has already been used in the A/B/C response study, so this is not a virgin benchmark. Pair focus and source index reduce broad topic differences, but they cannot eliminate different intents, wording, prompt lengths or all semantic confounds.

## Measured activation differences

A positive matched difference means that the **unit-normalized layer-10 last-token activation** for the contrast prompt has a larger cosine with the corresponding frozen direction than its matched safe prompt. It does *not* mean a model refused the unsafe prompt.

| Direction | Mean matched difference | Corrected median | Positive pairs | Positive family means |
| --- | ---: | ---: | ---: | ---: |
| Original M1 fitted | **+0.179590** | +0.163781 | 18/18 | 6/6 |
| Equal-unit-weight | +0.154175 | +0.133314 | 18/18 | 6/6 |
| Calibrator 1 only | +0.032841 | +0.033355 | 15/18 | 5/6 |
| Calibrator 2 only | **+0.195470** | +0.182047 | 18/18 | 6/6 |

For the original M1, individual matched activation differences ranged **+0.065532 to +0.363709**; leaving out any one of the six families produced means in the **+0.154289 to +0.189200** range. The mean contrast-minus-safe prompt length was **−1 character**, but individual prompt lengths and framing still varied.

Across 128 seeded isotropic directions, the median absolute matched mean difference was **0.006848**, and the maximum was **0.025893**. None matched the original M1's absolute mean difference. These isotropic vectors are *descriptive geometry controls*, not a causal or statistically exchangeable null; do not convert their ranks to p-values or claim any refusal safety effect.

The original M1 direction again exhibits a consistently positive matched activation contrast. Calibrator 2 alone shows higher mean alignment than original M1, while equal-unit-weight is lower. This contrasts with the *minimum* held-out alignment observed for equal-unit-weight on the prior six-pair calibration dataset. Neither finding authorizes selecting a better abliteration checkpoint without behavioral testing.

## Reproducibility and custody

Implementation: `src/xstest_matched_direction_transfer.py`. Regression tests: `tests/test_xstest_matched_direction_transfer.py` and `tests/test_xstest_matched_direction_evidence.py`.

Two isolated CPU BF16 runs on `x1-370` completed in approximately **96.25 and 89.30 seconds**, each with **36 forward passes** and zero generations or edited weights. The second used the corrected even-sized median estimator; the means for all four directions were **identical** across both runs. The earlier report is retained privately; the corrected report is the canonical aggregate evidence file: `evidence/20261101_xstest_matched_18_actual.json`.

The implementation processes raw per-pair activations transiently in memory and writes only class/family-level aggregate measurements. The output excludes source IDs, prompts, individual activations and responses, and retains explicit `model_release_approved=false` and no-human-grades flags. The local file is SSD-only and owner-protected. GitHub contains only the verified aggregate evidence, reproducibility notes and executable contracts.

Root disk free was approximately **26.164 GiB** on the latest initial and in-flight checks; prior storage incident #106 remains unresolved because root-space fluctuations and other concurrent writers have not been fully attributed. No system services were stopped or restarted.

## What this establishes—and does not

Evidence now supports a **reproducible internal activation contrast**, including source-index-matched safe/contrast prompts, on a specific frozen LFM3 checkpoint. It does not demonstrate that the direction causes refusal, that abliteration improves answers to benign prompts, or that unsafe actionability and canonical capabilities are preserved. **No new checkpoint weights, model releases, or zero-refusal claims are justified.**

Prioritize truly independent semantic controls, human adjudication of the existing frozen 1,350 response set and canonical task-ability retention before another weight-edit experiment.
