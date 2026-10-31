# LFM3 research: expanded disk-capacity guard and direct checkpoint protection

**Date:** October 31, 2026. **Host:** x1-370.
**Status:** Implemented and regression tested; root filesystem remains below required free space.

## Scope of protection

The earlier four entry points (XSTest, ARC quick, ARC full likelihood,
and three-arm smoke) retain their existing disk gates. This update
integrates the same shared guard with five additional paths:

- `src.refusal_probe`: before model loading and again before writing
  direction vectors, private generated responses, and final summary.
- `src.abliteration_fidelity`: before source/model loading, before
  each checkpoint load, and before final report write.
- `src.reversible_alpha_controls`: before source and output creation,
  before model load, before every generation journal append, and
  before the final public aggregate.
- `src.orthogonal_sham_repeats`: before source and output creation,
  before model load, before every journal append, and before the
  final public aggregate.
- `src.build_abliterated_checkpoint`: **inside the direct build()
  Python API**, before reading/checkpoint construction and immediately
  before writing the full experimental model.safetensors artifact.

The shared `src.research_job_guard.existing_directory` now rejects
symlink components anywhere along a candidate output path, rather
than checking only its nearest existing ancestor. Paths containing
parent-traversal components are rejected. Numeric thresholds reject
non-finite values, strings and booleans. This is not a sandbox,
security boundary against arbitrary hostile code, or a global quota.

## Actual measured fail-closed checks

- Before this change, the XSTest runner already correctly blocked
  startup at ~6.007 GiB free vs. the 20 GiB floor.
- The reversible projection runner was directly invoked against
  non-existent private/public destinations on x1-370. It blocked
  before output creation at ~5.845 GiB free.
- The experimental checkpoint builder was directly invoked with
  `--allow-smoke` and a fresh destination. It blocked before
  creating that directory at **5.736 GiB free** (20 GiB required).
- Neither actual probe generated any model text, loaded a weight
  checkpoint, deleted files, edited source evidence or wrote a model.

The full study suite passed **368/368 tests**, including synthetic
checks that all five new paths refuse insufficient capacity before
accessing source data; direct build() caller protection; chained
symlink rejection; nonfinite threshold rejection; and conservation
of the explicit --allow-smoke experimental authorization gate.

## Boundaries and next steps

The default policy requires >=20 GiB of *unprivileged* available
space at the local work filesystem, plus the greater of planned
write and reserve requirements. It checks a separately mounted
destination independently. This is a **preflight** rather than an
atomic reservation: external background agents and live writes can
consume space between checks. Unrelated fleet repositories are
outside scope; the shared library is not automatically injected
into arbitrary CLI launchers.

The production root /dev/nvme1n1p6 remains at 100% reported use,
with ~5.8 GiB available during the final regression suite, and
/nas remains a distinct available mount. Recovery issue #106
remains open, with no automatic deletion, migration, reserved-block
adjustment or service restart performed here.

The original XSTest private generation journal and both independent
1,350-response reviewer assignments are unchanged, with **zero
human-reviewed responses**. No behavioral refusal-safety conclusions,
production deployments or checkpoint promotions are authorized by
this engineering hardening.
