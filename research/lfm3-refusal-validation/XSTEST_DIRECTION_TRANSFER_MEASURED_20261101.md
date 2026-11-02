# XSTest 54-prompt LFM3 activation transfer — measured November 1, 2026

**Status: completed, source-pinned, exploratory geometry only.** This is not a human-rated refusal, safety, or model-utility experiment; no new model or checkpoint was generated.

## Source and procedure

The original LFM3-1.7B post-trained A weights (SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`) were loaded locally on `x1-370` for 54 CPU BF16 forward passes. Source: the independently sourced **XSTest** corpus (SHA-256 `11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a`). Selection was frozen by seed **9417**, three prompts each from all 18 original dataset categories, giving **30 safe** and **24 unsafe** prompts. This corpus differs from the original eight contrast-calibration pairs, but XSTest had previously been used for A/B/C response evaluation, so it is **not a previously unseen benchmark**.

The method captures the final-token activation at layer 10, normalizes it, then computes the difference of mean signed cosine between XSTest *unsafe* and *safe* groups for four directions: the original magnitude-weighted M1, equal-unit-weight post-hoc candidate, and two single-calibrator directions. The code also checks 128 seeded isotropic random unit vectors **not forced orthogonal** to M1 (seeds 31000–31127). These controls are descriptive comparisons, not a model-behavior null distribution or causal randomization test. No test compares model-generated refusals, helpfulness, or actionable harmful content.

Exact source dataset, model, and frozen fitted direction SHA-256s are checked **before and after** the forward-pass run. The existing write-admission policy checks both root and destination prior to model loading and again before writing the aggregate. Input calibration activations remained private on the SSD; private prompt text, source selections, per-prompt vectors and all completions were absent from the output. Only the checked aggregate `evidence/20261101_xstest_direction_transfer_54_actual.json` was committed to GitHub.

## Real measured results

| Direction | Mean safe cosine | Mean unsafe cosine | Unsafe − safe |
| --- | ---: | ---: | ---: |
| Original magnitude-weighted M1 | 0.002473 | 0.146597 | **+0.144124** |
| Equal-unit-weight | 0.032797 | 0.158387 | +0.125590 |
| Calibrator 1 only | 0.102736 | 0.135300 | +0.032564 |
| Calibrator 2 only | −0.054169 | 0.099248 | **+0.153417** |

Of the 128 independently seeded isotropic controls, the median absolute unsafe-minus-safe difference was **0.005333**, and the maximum absolute difference was **0.020010**. Zero of 128 controls matched the original fitted direction's 0.144124 gap. This does **not** imply a hypothesis-test p-value, an independently sampled population, or evidence that the model refuses less. The fitted direction and the diagnostic endpoints were selected after earlier work, and the sample contains only three prompts per category.

## Cross-experiment implication

The earlier **six reused held-out pairs** favored an equal-unit-weight combination in their *minimum* signed alignment (0.171938 versus 0.095145 for original M1), with almost equal mean alignment. In this **different XSTest 54-prompt class-separation** diagnostic, the same equal-unit candidate has *lower* separation than original M1 (0.125590 versus 0.144124); calibrator 2 alone exceeds both.

These outcomes are not contradictory: the two diagnostics measure different quantities on different inputs, and neither optimizes the actual behavioral outcome. They caution against choosing an abliteration edit solely from activation cosine or a favorable sham score. Also, safe and unsafe XSTest categories are unpaired and vary in topic and length, so the observed gap may encode semantic/topic differences rather than a causal refusal mechanism.

## Reproducibility and safety

- Research code: `src/xstest_direction_transfer.py`; synthetic tests: `tests/test_xstest_direction_transfer.py`. Immutable aggregate-contract test: `tests/test_xstest_direction_transfer_evidence.py`.
- Frozen original checkpoint A, eight-pair M1 calibration capture, and XSTest CSV were SHA-verified; **54 forward passes, zero model text generations, zero weight modifications**.
- Local run completed in approximately **123.53 seconds** on x1-370. Private final aggregate: 2,475 bytes, file mode 0600, SHA-256 `60fd45ae8e26973199b12396f23719008ae66fc6eacf429de5d7dedc1dcee8e6`. Final artifact and its private path remain on the SSD. GitHub contains only aggregate statistics and validation code, with no raw prompts or activations.
- Root remained approximately **26.166 GiB free** on the latest read; separate SSD approximately **416.2 GiB** free. Root issue #106 is still open and host-wide memory usage includes other models/services. No production deployment, global service stop, model promotion, or root cleanup was made.

**Research conclusion:** there is reproducible evidence that the frozen M1 direction captures an activation contrast; a separate stratified corpus also shows a nontrivial safe/unsafe class gap relative to isotropic controls. There is **no established behavioral refusal effect, legitimate-use benefit, safety preservation, or new-model superiority**. The next decision must rely on independent matched semantic controls, blinded human review of the existing frozen outputs, and canonical capability retention rather than more post-hoc selection on these 54 prompts.
