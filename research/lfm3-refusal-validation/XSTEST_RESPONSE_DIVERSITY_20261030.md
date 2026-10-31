# LFM3-1.7B full XSTest exact-response diversity audit — 2026-10-30

Status: completed aggregate-only post-hoc analysis of the **original** frozen
450 XSTest prompts × A/B/C = 1,350 local generations on x1-370.
No new inference, training, edits, human labels, checkpoint release or deployment.

## Previously hidden repetition in the lexical counts

| Frozen checkpoint | Exact-distinct responses /450 | Largest identical-response group | Responses with exact text shared across benign and unsafe prompts |
|---|---:|---:|---:|
| A original post-trained | 441 | 10 | 10 |
| B M1 weight-edited | 450 | 1 | 0 |
| C pretrained base | 295 | 156 | 156 |

C emits **the exact same short response 156 times**, including **29 of the
250 benign prompts and 127 of the 200 unsafe contrasts**. This same response
matches the original weak refusal-keyword pattern. A has a smaller repeated
short response spanning five benign and five unsafe prompts. B has no exact
duplicate responses in these 450 completions.
## Impact on interpreting the original weak refusal keyword counts

| Arm | Benign keyword-positive /250 | Distinct benign keyword-positive strings | Unsafe keyword-positive /200 | Distinct unsafe keyword-positive strings |
|---|---:|---:|---:|---:|
| A | 39 | 35 | 172 | 168 |
| B | 31 | 31 | 167 | 167 |
| C | 34 | 6 | 163 | 37 |

Of C's 163 unsafe keyword-positive responses, **127 are the identical
short repeated response**. The remaining 36 keyword-positive responses
have different exact text. Of C's 34 benign keyword-positive responses,
29 are the same repeated response and five have different exact text.

A conventional keyword rate counts each *prompt-completion pair*, even
when the model returns identical text for unrelated requests. Counting
distinct strings is a **diversity diagnostic**, not an alternate refusal
rate: different wording does not guarantee different behavior, and identical
wording is not automatically inappropriate.

The observations support the hypothesis of a prompt-insensitive stock
reply for C, but this remains an inference from outputs, not evidence
of an internal mechanism or a measured model property across tasks.
C is a pretrained base using a chat template despite lacking instruction
tuning; it is not a controlled refusal-specific intervention.
## Provenance, privacy, and remaining gates

- Pinned source dataset, model A/B/C SHA-256, complete journal SHA-256,
  source class/category and complete 1,350-pair coverage are verified.
- Exact response strings are counted in memory only. The public JSON holds
  **no response text, no response hashes, no per-case IDs, and no generated
  prompts**, only aggregate counts and artifact integrity digests.
- Ties for the most common string are not arbitrarily resolved. When
  no unique top mode exists, the mode breakdown is null.
- Lexical-positive classification uses the original frozen regex. A high
  unsafe count may be legitimate or merely formulaic; a high benign count
  may reflect over-refusal. Neither conclusion is established automatically.
- No inferential significance test was pre-registered for diversity; all
  findings are post-hoc, descriptive, and conditioned on 128 maximum tokens.
- The two independent 1,350-response human-review queues remain ungraded.
  A blinded human must assess each prompt-response pair in context even
  when response strings repeat verbatim. Do not auto-copy human grades
  across prompts based on shared text.
- No model weight release, deployment, or safety mechanism approval.

Run from the study package root using the pinned Python interpreter:

```bash
python -m src.xstest_response_diversity \
  --results-dir /home/scott/git/lfm3-refusal-study-20261029/results/xstest_full_20261029_v1 \
  --out evidence/NEW_UNIQUE_xstest_response_diversity.json
python -m pytest -o addopts= -q tests
```

Frozen public aggregate: `evidence/20261030_xstest_response_diversity.json`.
Never overwrite the existing evidence file or share the private response corpus.
