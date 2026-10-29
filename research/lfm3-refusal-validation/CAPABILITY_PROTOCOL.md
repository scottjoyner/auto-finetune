# Independent capability-retention axis for the three fixed LFM3 checkpoints

The main refusal study retains exactly three model checkpoints:
A original instruction-tuned, B M1-edited A, and C pretrained base.
Capability evaluations are independent measurements, NOT additional models.

## Public ARC-Challenge source

- Source: `allenai/ai2_arc`, ARC-Challenge test parquet.
- Pinned HF revision: `210d026faf9955653af8916fad021475a3f00453`.
- SHA-256 of test parquet: `62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9`.
- Total questions: 1,172; raw questions and answers remain local.

## Experimental ARC-style probe

The runner `src.arc_capability_eval` selects the same seeded sample for A/B/C,
formats each question with canonical A-E option letters, and compares the next-token
letter logits in a **raw completion prompt** rather than using a chat template. This
is intentional to reduce a major fairness mismatch for the pretrained base.

The metric is **only a one-token letter-completion accuracy proxy**. It is *not*
the benchmark's official choice-sequence conditional likelihood score, does not
use multiple option permutations, and cannot stand alone as a capability-retention
claim. A full publishable study will use the test's complete set, multiple choice
order permutations, proper normalized conditional likelihood, and standardized
benchmarks of instruction following and tool use.

Example local command:

```bash
cd /home/scott/git/lfm3-refusal-study-20261029
/home/scott/lfm3-research/train-venv/bin/python -m src.arc_capability_eval --sample-size 12 --seed 17
```

Private case-level outputs are never committed; only sanitized denominator
counts and limitations may appear in the paper or pull request. Never infer
that successful reasoning implies resistance to malicious or unauthorized actions.


## Preliminary 12-item smoke result (not official accuracy)

A original: 4/12; B M1-edited: 4/12; C pretrained: 8/12.
All twelve A/B raw-completion selected letter predictions were identical.
This cannot validate preservation of capability; report only as an
integration test. No benchmark question or output text is public.
