# Frozen LFM3-1.7B architecture vs. refusal-keyword indicators

Status: descriptive, 450-prompt XSTest keyword screening only.
None of the 1,350 private model generations has been human-graded.
A high keyword frequency is not a verified refusal rate. In particular,
refusal on unsafe prompts can be appropriate, whereas refusal on benign
prompts can be unwanted. This document does not rank safe compliance.

## What architecture was actually compared?

The three frozen A/B/C checkpoints have **identical** 24 decoder layers:
13 gated convolution, 5 sliding-window attention, and 6 full-attention
layers, all with 2,048 hidden size and 16 configured attention heads.
A is original post-trained LiquidAI LFM3-1.7B, B is an M1
rank-one edited derivative of A, and C is pretrained LFM3-1.7B-Base.
There was no architectural swap to a new spirited, sparse, or
alternative attention architecture in this experiment.

These three model files were compared using full-file SHA256 receipts,
the exact model config layer-type sequence, and the same 450 XSTest
source prompts. Because the architecture is fixed, observed
differences cannot be attributed to sparse vs dense or convolution
vs attention without a separate, suitably controlled experiment.

## Highest observed keyword signal

| Frozen checkpoint | Benign keyword-positive /250 | Unsafe keyword-positive /200 |
|---|---:|---:|
| A original post-trained | **39 (15.6%)** | **172 (86.0%)** |
| B M1 edited derivative | 31 (12.4%) | 167 (83.5%) |
| C pretrained base | 34 (13.6%) | 163 (81.5%) |

A leads BOTH strata on this weak measure. Wilson 95% intervals,
shown only as descriptive binomial intervals for keyword incidence:
A benign 11.63–20.61%, A unsafe 80.51–90.13%;
B benign 8.87–17.06%, B unsafe 77.73–88.00%;
C benign 9.90–18.40%, C unsafe 75.54–86.27%.

Matched benign A to B positive keyword detections fell by
8 of 250; nominal exact conditional two-sided p=0.021484375,
but Holm-adjusted across six exploratory contrasts p=0.12890625,
not below 0.05. Keyword matches can reflect wording rather
than actual refusal, helpfulness or harmful actionability.

Do not equate the strongest keyword-positive rate with the
strongest safety mechanism. The 24-layer hybrid architecture is
descriptive; no layer was established as a causal refusal circuit.

The public audit evidence/20261030_refusal_architecture_lexical_ranking.json
contains only architecture layer types, aggregate counts,
exploratory intervals and source/checkpoint SHA256 receipts.
No private prompts, response texts, human identities or weights
are included.

## Blocking evidence

Two independent humans must finish their separate private
1,350-item blinded ratings; any disagreement requires actual
human adjudication, not a model judge or keyword classifier.
No universal zero-refusal result, safe-compliance rate, model
release, or public publication is established here.
