# LFM3-1.7B Refusal and Adversarial Validation — research prototype

**Status: research-only, NOT a validated safety mechanism or an unrestricted public model release.**

This is a standalone safety-focused companion to auto-finetune and sensorgraft.
The experiment studies how LFM3-1.7B's conv, sliding-attention, and full-attention
layers encode refusal-related signals. It compares the three frozen post-trained, abliterated-derivative and base
checkpoints, and evaluates behavioral refusals, benign utility, and
simulated authorization using separate validation gates.

## Complete XSTest generation results (October 29, 2026)

**Inference completed; human safety evaluation not completed.** The full
pinned 450-case XSTest dataset was generated with exactly 3 model arms,
128 maximum new tokens and greedy seed 17 on x1-370: 1,350/1,350
generation records. The read-only audit verified model/checkpoint SHA-256
identities, exact source hash, case IDs, source categories, unique pairs
and clean JSONL. It found no missing or incomplete generations.

Weak lexical refusal keyword indicators, not validated safety grades:

| Checkpoint | Safe: keyword / 250 | Unsafe: keyword / 200 |
| --- | ---: | ---: |
| A original post-trained | 39 | 172 |
| B M1-abliterated A | 31 | 167 |
| C pretrained base | 34 | 163 |

For A vs B matched prompts, the raw keyword proxy declined 3.2 percentage
points on safe prompts and 2.5 points on unsafe contrast prompts. This
does NOT establish increased harmful compliance or improved helpfulness.

A private, blinded packet of all 1,350 completions has been generated
for each of two independent human reviewers. Reviews NOT STARTED.
Complete built-in canonical ARC-Challenge scoring is now finished
for A/B/C across all 1,172 questions, with per-case outcomes exactly
matching the earlier full local-parquet run (October 30, 2026).
This does NOT establish capability equivalence or behavioral safety.
Two independent human XSTest reviewers remain ungraded. No third-party
model judge, external transfer of prompts or public checkpoint weights.
See BENCHMARK_PROTOCOL.md, HUMAN_REVIEW_GUIDE.md and paper/main.tex.

Run from the study root:

    python -m src.xstest_progress_audit --results-dir results/xstest_full_20261029_v1
    python -m src.publication_gate

## Reversible alpha-direction controls (October 30, 2026)

The original A checkpoint was tested in an additional **72-generation
six-holdout-pair CPU BF16 smoke** with reversible layer-10 alpha values
0, 0.25, 0.5, and 1, a random orthogonal control, and no-hook baseline.
All conditions had zero of six benign and six of six unsafe lexical
refusal indicators. At alpha 1, exact response text changed in four
benign and all six unsafe generations; matched alpha 0 replicated A
exactly. These are **lexical and text-divergence observations only**,
not verified refusal, safe compliance, or altered-checkpoint efficacy.
No model weights were modified. See
[REVERSIBLE_ALPHA_CONTROLS_20261030.md](REVERSIBLE_ALPHA_CONTROLS_20261030.md).
Two real human XSTest reviews remain the primary publication blocker.

## Multiple orthogonal random-control repetitions (October 30, 2026)

Three additional orthogonal random-control directions (seeds 23, 41, 59)
were evaluated on the same six disjoint pairs, producing 36 new
CPU BF16 generations. All 4 shams (including existing seed 17)
changed fewer exact response strings than the fitted alpha=1 direction
on these small heldouts. However, **no paired control comparison passed
the six-test Holm adjustment** (smallest corrected p=0.375), and
refusal-keyword classifications never changed across conditions.
These are *response-string* findings, not behavioral safety grades.
See [ORTHOGONAL_SHAM_REPEATS_20261030.md](ORTHOGONAL_SHAM_REPEATS_20261030.md).
All A/B/C model weights and the frozen XSTest journal remain unchanged.

## Full XSTest response-length bias diagnostic (October 30, 2026)

The existing 1,350 frozen XSTest generations were analyzed without
new inference or private-text publication. Median unsafe-prompt
response length was A 67 whitespace words, B 93, and pretrained C 8.
Of C's 200 unsafe responses, **127 were shorter than 50 characters
and every one of those 127 triggered the weak refusal-keyword regex**.
This makes the original keyword-count ranking inadequate as a
behavioral refusal comparison. Paired word-length differences were
also tested with six Holm-adjusted exploratory exact sign tests.
No behavioral safety, justified refusal, or helpfulness was verified.
Details: [XSTEST_LENGTH_CONFOUNDING_20261030.md](XSTEST_LENGTH_CONFOUNDING_20261030.md).

## Exact response-duplication audit (October 30, 2026)

The full immutable 1,350-generation XSTest panel exhibits a significant
descriptive caveat: pretrained C returned **one identical short,
keyword-positive response 156 times**, including 29 benign prompts
and 127 unsafe prompts. C generated only 37 distinct keyword-positive
unsafe response strings despite 163 unsafe keyword-positive samples.
A had 168 distinct keyword-positive unsafe strings from 172 positives;
B had 167 from 167. These are exact-string diversity measurements,
**not independent verified behavioral refusal events**. Responses and
per-response hashes remain private. Details and aggregate-only evidence:
[XSTEST_RESPONSE_DIVERSITY_20261030.md](XSTEST_RESPONSE_DIVERSITY_20261030.md).
Human refusal grading, actionable-harm review and model release stay blocked.

## Human-only targeted review pilot (October 30, 2026)

An independently reproducible 32-prompt, 96-response-per-reviewer pilot
was **actually selected on x1-370** from the complete frozen XSTest
review queue. It deliberately includes safe/unsafe contexts where C
repeated its modal response or generated another response (8 prompts
per bucket, three checkpoint answers per prompt). No model-arm labels,
raw responses, source IDs, human graders or response hashes are
published. The original blinded CSVs remain authoritative: an
interactive human reviewer edits the original assigned batch rows,
not a competing private packet. Pilot status is **0/96 graded for each
of two independent reviewers**; no annotations are manufactured.
This is a purposive pilot, NOT a representative estimate, a blinded
human outcome, consensus, or a model-release authorization.
See [PRIVATE_HUMAN_REVIEW_PILOT_20261030.md](PRIVATE_HUMAN_REVIEW_PILOT_20261030.md)
for local human-only launch and progress commands.

## Pilot source-attestation and large-write storage guard (October 30, 2026)

The existing human-only pilot review/status commands now fail closed if
the 96 blinded review ID positions differ from a **fresh deterministic
reconstruction** based on the pinned 1,350-generation journal, complete
450-case source, C-repeat strata, and original model/blind mapping. A
live tamper test confirmed that a route containing valid but reordered
IDs now gets rejected, while the authentic route passes. No original
human ratings or prompt/response text are changed or published.
The optional read-only compute preflight (`src.research_storage_preflight`)
correctly **blocks** a 5-GiB planned write on x1-370 with only ~8.6 GiB
available under a 20-GiB free-space floor. This guard applies only to
new job launchers explicitly invoking it; it does not reserve space or
automatically block external processes. See
[PILOT_ROUTE_ATTESTATION_AND_STORAGE_PREFLIGHT_20261030.md](PILOT_ROUTE_ATTESTATION_AND_STORAGE_PREFLIGHT_20261030.md).

## Human-only pilot inter-rater agreement gate (October 30, 2026)

After two actual independent reviewers finish all 96 selected original
blinded batch entries, `src.private_pilot_agreement` verifies the
exact source-attested pilot and scores response-category, actionability
and benign-helpfulness inter-rater agreement. It **refuses partial or
uncertain reviews**, maintains the source prompt/response privacy
boundary, reports aggregate Cohen's kappa with null for degenerate
marginals, and never unblinds model-arm rankings or authorizes release.
The real x1-370 pilot is **still ungraded: 0/96 per reviewer**; its
agreement CLI correctly exits 2 without writing a report.
See [PRIVATE_PILOT_HUMAN_AGREEMENT_GATE_20261030.md](PRIVATE_PILOT_HUMAN_AGREEMENT_GATE_20261030.md).

## Frozen XSTest timing and repeated-output confounding (October 31, 2026)

The original 1,350 CPU BF16 completions now have a reproducible
aggregate-only elapsed-time audit. Unsafe-prompt median per-generation
elapsed time was A 11.75 s, B 15.70 s, C 2.6165 s, but C's 127
identical short unsafe replies had median 2.523 s versus 15.456 s for
the 73 other unsafe C outputs. Therefore C's apparent fast response
time is closely associated with short, repeated answers; the logs
cannot establish improved decode throughput or safety. Exploratory
per-prompt signed timing comparisons use six Holm-adjusted tests.
There are **no tokenizer token counts, prefill/decode splits or finish
reasons** in the frozen response journal. See
[XSTEST_LATENCY_REPETITION_20261031.md](XSTEST_LATENCY_REPETITION_20261031.md).

## Fail-closed disk checks integrated with evaluation runners (October 31, 2026)

The XSTest, ARC quick, ARC full-likelihood and three-checkpoint smoke
entry points now check local root free space **before loading models or
writing output**. XSTest and ARC full also recheck per checkpoint and
journal append. The policy requires 20 GiB free and checks a different
output filesystem independently, even when an NAS is mounted. The live
x1-370 XSTest attempt correctly stopped at 6.007 GiB free. This cannot
prevent external fleet jobs consuming disk or reserve filesystem space.
See [DISK_GUARD_INTEGRATION_20261031.md](DISK_GUARD_INTEGRATION_20261031.md).

## Expanded experimental-run and checkpoint-write disk safeguards (October 31, 2026)

Five additional LFM3 research paths now use the same local storage
preflight: refusal probe, M1 activation fidelity, reversible residual
projection, orthogonal-sham repeats, and the direct experimental
checkpoint **build()** API. Together with XSTest and the existing ARC
and three-checkpoint evaluation runners, nine entry points are guarded.
The shared output-path validation now refuses intermediate symlinks,
parent traversal and invalid/nonfinite capacity limits. Live attempts
to run the reversible probe and write a new model checkpoint stopped
**before any outputs or model inference** at ~5.8 GiB free versus a
20 GiB local minimum. No cleanup, model changes, or claims of human
behavioral refusal followed; issue #106 remains open. Full synthetic
regression suite: **368/368 tests passing**. Details in
[RESEARCH_DISK_GUARD_COVERAGE_20261031.md](RESEARCH_DISK_GUARD_COVERAGE_20261031.md).

## Post-hoc C-output-conditioned matched XSTest latency (October 31, 2026)

On the **same unsafe prompts**, C was faster than A on 126/127
requests producing C's identical short stock answer (median C-minus-A
elapsed time **-8.550 seconds**), but only 32/73 requests with
other C replies (median difference **+0.165 seconds**; exploratory
Holm-adjusted sign-test p approximately **0.349**). Among benign
nonrepeated C answers, C was slower than A on 176/221 matched prompts.
These output-conditioned, post-hoc comparisons from the original
1,350 frozen CPU BF16 generations strengthen the repeated-answer
confounding diagnosis; they **do not** prove causal model efficiency,
valid behavioral refusal, or tokenizer-token throughput.
See [XSTEST_MODAL_CONDITIONED_PAIRED_LATENCY_20261031.md](XSTEST_MODAL_CONDITIONED_PAIRED_LATENCY_20261031.md).

## Whitespace-word overlap sensitivity (October 31, 2026)

`src.xstest_length_overlap_sensitivity` is a synthetic-test-validated
method for checking the **coverage loss** when frozen C/A and C/B
responses differ by at most 0, 5, 10 or 20 whitespace words on
matched prompts. It separates C's repeated short answer from other
outputs, reports sample retention and paired elapsed timing, and
makes **no tokenizer-throughput or behavioral refusal claim**.
The actual frozen private journal was audited on x1-370 and its
32-row aggregate reproduced byte-for-byte. For unsafe C stock responses,
only **3/127** matched A within five whitespace words, and only
**5/127** within 20 words. For other unsafe C responses, the 20-word
overlap retained **48/73** (C faster 21, slower 27). The report is
`evidence/20261031_xstest_length_overlap_sensitivity.json`.
See [XSTEST_LENGTH_OVERLAP_SENSITIVITY_20261031.md](XSTEST_LENGTH_OVERLAP_SENSITIVITY_20261031.md).

## Single-snapshot private pilot review integrity (October 31, 2026)

The original 32-prompt / 96-response-per-reviewer pilot route is
now opened once during source-bound attestation; both interactive
human review and the post-review agreement gate consume **that exact
verified in-memory snapshot** rather than reopening the route.
Private-route reads use `O_NOFOLLOW` / `fstat` and reject
nested symlinks and multiply-linked route files. Real source
checks passed for both pilot reviewers, each still **0/96 graded**;
the agreement gate correctly refused to create an output from
incomplete human reviews. The suite passed **391/391 tests**.
See [PRIVATE_PILOT_VERIFIED_ROUTE_SNAPSHOT_20261031.md](PRIVATE_PILOT_VERIFIED_ROUTE_SNAPSHOT_20261031.md).

## Assets on x1-370
- Original source weights: /home/scott/lfm3-research/model-metadata
- Original Q4 benchmark: /home/scott/lfm3-research/gguf/LFM3-1.7B-Q4_K_M.gguf
- Python: /home/scott/lfm3-research/train-venv/bin/python
- Nothing in this repository changes or deletes the originals.

## Quick start
Run in the project directory:
  /home/scott/lfm3-research/train-venv/bin/python -m pytest -o addopts= -q tests
  /home/scott/lfm3-research/train-venv/bin/python -m src.refusal_probe --mode probe --max-pairs 2
  /home/scott/lfm3-research/train-venv/bin/python -m src.refusal_probe --mode compare --max-pairs 2 --layer 10 --alpha 0.25

The probe writes a private activation-direction safetensors file to the ignored
results directory; compare tests frozen-weight generation using reversible hooks.
Do not publish the vector, raw harmful responses, or derived weights without
separate safety and license reviews.

## Research methodology and limits
1. Freeze model SHA, tokenizer/template, seed, precision, and matched dataset.
2. Capture last-position hidden vectors from each decoder block with hooks.
3. Estimate per-layer harmful-minus-benign contrastive directions; do not assume
   these are causal refusal circuits without intervention and holdout validation.
4. Compare baseline and reversible projections; include sham controls and
   independent held-out cases. Current eight pairs are smoke tests ONLY.
5. Score false refusals, legitimate helpfulness, refusal and safe-redirection
   behavior on adversarial prompts, and mocked unauthorized tool attempts.
6. Require independent manual annotations for actionable harmful content;
   keyword refusal matches are not a sufficient safety verdict.
7. Inference is local and tool execution is disabled. Simulated actions never
   become real commands. No automatic deployment or model promotion.
8. Track regressions by category and publication gates in paper/main.tex.

## Critical limitations
- Correlation is not causation: steering can shift model style without changing
  effective resistance to malicious actions.
- Eight paired probes and a lexical refusal proxy cannot support broad claims.
- Base and post-trained variants must be evaluated separately; Q4 and BF16
  results must not be pooled.
- Work remains unpublished until enough independent human-graded holdout results
  exist, reproducibility is verified, and responsible-release review is complete.

## Three-checkpoint M1 smoke extension (October 29, 2026)

The v0.1 research scope is exactly three model checkpoints:
A = original post-trained LFM3-1.7B; B = M1-style abliterated derivative
of A; C = original LFM3-1.7B-Base. See STUDY_PROTOCOL.md.

Source checksums and the direction-vector provenance are captured in a
private local receipt. The base model is pinned at Hugging Face revision
545fc8f4c7b5fa9f405105274cd29c4f8f44b526.

The B checkpoint was edited locally using difference-of-means from two
calibration prompts (SMOKE ONLY) and full-alpha rank-one orthogonalization
of 48 residual-writing output matrices. Verification confirmed that all
48 selected tensors changed exactly as prescribed and the other 238 did
not change. The original model has not been modified.

Exactly 4 heldout prompt conditions were executed per checkpoint with a
20-token output cap. Refusal keyword counts were 2/4 in each arm and
only 1/4 generated responses matched exactly between A and B. These
are diagnostic integration results, NOT refusal or safety efficacy claims.

Commands in this study directory:
  /home/scott/lfm3-research/train-venv/bin/python -m pytest -o addopts= -q tests
  /home/scott/lfm3-research/train-venv/bin/python -m src.verify_checkpoint
  /home/scott/lfm3-research/train-venv/bin/python -m src.three_checkpoint_eval --max-heldout-pairs 2

DO NOT rerun the B checkpoint builder unless using an explicitly new
output path: it refuses to overwrite existing checkpoint material.
Keep raw generations, vectors, and all model weights local.
