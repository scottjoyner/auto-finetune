# Model export must be release-approved before writing deployable artifacts

The legacy src/model_export.py could merge adapters into HF weights,
convert through a locally installed llama.cpp script to GGUF, and
export ONNX without source authentication, independent held-out quality
validation or trusted release authority. The previous module did not
distinguish synthetic AdamW/Muon receipts from approved model releases.

This draft replaces all three legacy export routes with an explicit
fail-closed release hold. It preserves the public Python entrypoint
names and ExportResult structure for caller compatibility, but always
returns success=False. The module CLI returns HOLD (exit code 78) with
a structured JSON receipt and no private paths or checkpoint IDs.
It does not import model libraries, read checkpoints, create output
directories or files, access the network, or spawn converter processes.
No bypass environment variable is available. The historical exporter
remains accessible in Git history for a future reviewed design.

This is intentionally breaking: GGUF, ONNX and merged exports for
all model families are stopped, including non-LFM3 models. Do not merge
without independent release-operator and downstream-compatibility
review. It has not modified deployed code or an existing artifact.

A subsequent production-capable implementation needs:
1. A trusted release authority and cryptographically verified,
   replay-resistant approval token.
2. Bound identities for the complete pretrained model, adapter
   weights, exact training code, evaluator and held-out datasets.
3. Explicit quality/regression and resource constraints and an
   authorized requested output digest or output policy.
4. Atomic export to a protected staging path with integrity
   verification, audit trail, rollback and separate publication
   approval.
5. Multi-process concurrency and path/symlink/hardlink defenses,
   resource admission, and independent security review.

The synthetic tests validate API/CLI holds, frozen output directories,
mocked forbidden operations, no import of heavy dependencies and
private argument redaction. They do not exercise conversion kernels,
model fidelity, quantization accuracy or production permission
handling. The full-project 69% coverage gate must remain unchanged.
