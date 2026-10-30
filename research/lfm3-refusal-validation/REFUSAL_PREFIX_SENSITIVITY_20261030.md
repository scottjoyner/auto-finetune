# LFM3-1.7B frozen XSTest refusal-keyword sensitivity — 2026-10-30

Scope: read-only secondary analysis of the ORIGINAL A/B/C 450-question
XSTest experiment (1,350 generations). The three checkpoints and the
complete private journal remain unchanged. This is not a model release,
behavioral refusal validation, optimizer experiment, or new model arm.

## Real x1-370 results

Counts are *regex keyword matches*, not human-rated refusals.
Each benign denominator is 250; each unsafe denominator is 200.

| Response window | Benign A/B/C | Unsafe A/B/C |
|---|---|---|
| First 64 characters | 37 / 31 / 34 | 170 / 164 / 159 |
| First 128 characters | 37 / 31 / 34 | 171 / 165 / 162 |
| First 256 characters | 38 / 31 / 34 | 172 / 166 / 163 |
| First 32 whitespace words | 37 / 31 / 34 | 172 / 165 / 163 |
| Complete response | 39 / 31 / 34 | 172 / 167 / 163 |

A has the highest indicator count in BOTH strata for every measured
window. On BENIGN prompts, that might indicate *more unwanted refusal*.
On UNSAFE prompts, keyword positivity need not be a true justified refusal:
models can include these phrases and still produce actionable content.
These windows share the same keyword regex, so this is not independent
semantic confirmation or a new inferential hypothesis test.
## Reproduction, data custody, and caveats

- Source: locally pinned XSTest CSV; original A/B/C benchmark dataset,
  manifests and source/model/journal SHA-256 receipts rechecked.
- Greedy seed 17, CPU BF16, 128 max new tokens, pinned chat template.
- For each model response, regex match spans were found in the *uncropped*
  completion; a prefix was positive only if an existing match ended before
  its cutoff. This avoids manufacturing word boundaries by truncating text.
- New pure synthetic tests validate count monotonicity, A/B pairing,
  labeling, original-proxy parity, protocol identity and fail-closed handling.
- Public JSON evidence holds only aggregate window counts, paired A/B
  keyword transitions and digests; no raw prompts, completions or case IDs.
- Private data and model weights remain local on x1-370.
- Existing six-comparison Holm-adjusted p=0.1289 for A/B benign full-response
  indicators remains exploratory; the prefix audit does not change it.
- The two independent blinded human XSTest reviews were 0/1,350 graded
  each at this audit. Publication and release approvals remain blocked.

From the **study package root**, using the pinned Python venv:

```bash
python -m src.refusal_prefix_sensitivity \
  --results-dir /home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_20261029_v1 \
  --out evidence/NEW_UNIQUE_refusal_prefix_sensitivity.json
python -m pytest -o addopts= -q tests
```

Evidence: `evidence/20261030_refusal_prefix_sensitivity.json`.
Do not rerun into the same output filename. Next: two genuine independent
human reviewers score benign utility, justified refusal, and harmful
actionability, then paired scoring can replace these weak proxy rankings.
