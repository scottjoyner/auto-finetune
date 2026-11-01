# LFM3-1.7B: Human-only matched activation-to-behavior bridge (2026-11-01)

**Implemented, tested; behavioral gate remains CLOSED.** This increment adds a descriptive bridge from the completed 18 matched XSTest activation-geometry study to the *existing frozen A/B/C response-review workflow*. It does not generate responses, retrain any model, grade model text automatically, or treat a lexical proxy as a human label.

## Verified live human-review readiness

On `x1-370`, a read-only custody audit against the real private source mapping and reviewer batches returned **0/1,350 graded for reviewer 1**, **0/1,350 graded for reviewer 2**, with 15 original 90-response batches per reviewer still ungraded. Private source columns, model mapping, batch roster and owner-only permissions passed the existing checks; the reviewers' first incomplete batch is batch 001. No human judgments were obtained, inferred, auto-filled, or changed in this turn. No real behavioral comparison exists yet.

## New implementation

- `src/human_matched_activation_bridge.py` computes **only after complete human consensus** across all 450 prompts × A/B/C (1,350 unique source/model pairs), and uses the very same six source-index/focus-matched categories and three pairs per category (seed 9418) as `src/xstest_matched_direction_transfer.py`. It verifies exact triplet completeness, binary human rating domains, 0–3 benign helpfulness and full 250-safe/200-unsafe cohort composition before accessing the matched subset.
- `src/reviewer_adjudication.py` calls this bridge **only after** its verified original reviewer custody checks, full two-human grade completeness, disagreement queue verification, and third-human grading on every disputed case. A real unresolved or missing human rating blocks both the main 12-contrast familywise analysis and the matched bridge. The matched result lives inside the *existing private report*, not a newly public JSON file.
- The matched bridge aggregates per-arm (18 each) benign non-over-refusal count, benign mean helpfulness 0–3, unsafe non-actionability count, unsafe full-refusal count, and the count of pairs satisfying *both* benign non-over-refusal and unsafe non-actionability. It also reports 15 matched cross-arm descriptive comparisons (five metrics × A/B, A/C, B/C), each with positive/equal/negative transition counts.
- There are **no pair-level grades, case IDs, reviewer names, prompt texts, response texts or per-case activations** in this report. The sample is small and post-hoc: no multiplicity-adjusted p-value, equivalence or release recommendation is generated. The full 450-prompt 12-contrast familywise result remains the designated primary human study. The bridge does **not** calculate a per-prompt correlation between activation cosine and human response grades, because prior activation files preserved only aggregates rather than per-case coefficients.
- `tests/test_human_matched_activation_bridge.py` uses strictly labeled synthetic 450×3 grades to prove the output shape, paired effect signs, fail-closed behavior for forged/incomplete/invalid data, nonleakage, and lack of unauthorized inference. `tests/test_reviewer_adjudication.py` now verifies actual integration in synthetic third-human adjudicated and no-disagreement paths.

## How to activate once genuine human grades are complete

Continue the existing private blinded handoff and complete the 15 original review batches **independently for each of two real reviewers**. All source and model identities remain blinded. Resolve all discrepancies with a third human through `src.reviewer_adjudication`; that exact private workflow now embeds `human_behavioral_matched_activation_bridge` next to `human_behavioral_familywise_tradeoff` in the owner-only results. Do not run automatic reviewers or copy private human packets to public repositories. Do not create a score using the current ungraded files.

## Interpretation gates

Activation geometry is already empirically measured: original M1 gave +0.179590 mean matched cosine difference on 18 safe/contrast pairs; 18/18 differences were positive, but the signal could reflect intent, syntax or length and was selected after prior observations. The human bridge is distinct: it will evaluate independently scored response behavior of *frozen A, B, C checkpoints* on that same matched subset. It does not itself test a new intervention, nor prove the activation mechanism causes any behavior. The pretrained C checkpoint differs in training regime, so it remains a descriptive/confounded comparison.

Retain PR #103 as DRAFT, preserve original weights and private review packets, leave storage incident #106 OPEN, and forbid model publication, unsafe actionability claims or zero-refusal declarations until real human consensus and the primary capability/behavior gates are independently supported.
