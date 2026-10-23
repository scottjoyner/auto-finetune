# Direct trainer admission read failures — October 23, 2026

The canonical src/train.py main entrypoint already performs fail-closed
admission before scanning datasets or loading a model. The xwing active
checkout had an older Git HEAD, which initially made its locally
installed version appear uncommitted; the existing gate must NOT be
duplicated or bypassed.

This small hardening patch wraps the existing read_live() call so
unavailable admission services, exceptions or missing telemetry produce
a deterministic HOLD exit78 rather than an uncaught traceback.
A denied decision also returns exit78 as before; dry-run keeps its
CPU-only validation behavior without claiming training permission.

Eight synthetic CPU tests verify denial for direct calls to train.main
with different source/label/dataset arguments; assert no holdout
scan, tokenization or GPU/backend selection happens before admission;
and verify a failing admission read returns exit78. The tests also
ensure that positive admission cannot skip held-out validation and
that dry-run remains a non-training mode.

This does not enable checkpoint resume, remove the stop-file, change
GPU leases, start a training job, or overwrite the dirty live xwing
src/train.py. PR #48 independently documents the lack of an explicit
resume_from_checkpoint argument in both trainer backends. Restore
custody and physical R9700 power/OCuLink remain NO_GO.
