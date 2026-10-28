# LFM3 feedback and model-promotion release guardrails — 2026-10-28

**auto-finetune draft PR #61. No production release or autonomous self-modification authorized.**

## Motivation

A self-observing optimizer is unsafe if feedback records can be malformed, repeated analysis inflates issue frequencies, alerts lie about acknowledgments, or simply completing an A/B experiment implicitly activates new model weights. The pinned LFM3 advisor's previous prompt-sensitive disagreements prove that a model suggestion must be evidence, **never** operational authority.

## Changes committed

1. **`src/feedback_loop.py`:** `FeedbackCollector.record()` refuses nonfinite/out-of-range numeric scores (including bool/string), ensuring scores are numeric in [0, 1]. `_load()` now applies the same finite score filter to persisted JSONL rows; malformed/invalid legacy rows do not prevent reading later valid entries. Issue detection removes duplicates when `analyze()` is rerun; flagged-session output is deterministic and `get_flagged_sessions()` returns unique session **IDs**, not FeedbackEntry objects. These are local signal-quality checks, **not** externally attested source evidence.
2. **`src/alerting.py`:** alert IDs use high-resolution time plus random UUID suffix. Loading the append-only alert log uses **last event per alert ID** so acknowledgments remain acknowledged after restart, rather than duplicating outstanding alerts.
3. **`src/auto_promote.py`:** `PromotionConfig.require_manual_approval=True` by default. A candidate is **not** automatically promoted if no active baseline exists or A/B startup fails. `check_promotion()` requires `min_feedback_samples` in **each** arm and strictly valid nonnegative integral counts. Normal early completion requires **twice** that minimum in each arm; otherwise the configured test duration still matters. When evidence suffices, default mode returns `action: review_required` without invoking `ModelRegistry.complete_ab_test()`, which itself promotes as a side effect. The old direct flow is available only if an operator explicitly configures the legacy opt-out `require_manual_approval=False`.

## Verified evidence and remaining limitations

- `tests/test_feedback_loop_stability.py`: scratch-only recording, stable issue frequencies, unique flagged sessions, finite score refusal and repair-pair integrity. An additional test covers malformed/nonfinite persisted rows.
- `tests/test_alerting_replay.py`: scratch-only acknowledgment replay, unique IDs and mocked notification/log error behavior.
- `tests/test_auto_promote_review.py`: default human review, per-arm minimum and duration threshold, no-baseline, A/B startup failure, and mocked inference metric routing.
- Exact xwing scratch checks earlier in this pass confirmed feedback + alerting tests and verified A/B manual gates had no registry promotion side effect.
- **Hosted coverage:** a completed run at commit `28f835e8` passed **1,202 tests, 6 skipped**, with **58.68% coverage** vs previous **56.62%**. The mandatory **69%** gate remains unchanged and still fails. Later additional tests and score-replay changes require their own completed CI validation; do not claim they passed until observed.
- **Direct registry entrypoints remain separately exposed:** `ModelRegistry.promote` and `complete_ab_test` still mutate state when explicitly invoked outside AutoPromoter. There is **no independently signed approval capability**, immutable evaluation witness, producer consent chain or system-wide model promotion authorization boundary. Do not describe these draft changes as complete promotion security.
- SensorGraft parallel journal lock issue #8 remains unresolved. Instrumented scratch xwing test runs reproduced **3 failures in 35** 4-thread runs, all exclusive lock-busy on distinct temporary regular lock files. The debug instrumentation was reverted; it did not prove a kernel/filesystem root cause.
- A separate staged `src/lfm3_last_layer_lora_lab.py` option compares native Muon `original` vs `match_rms_adamw` scaling across **three** fixed synthetic seeds with three steps per arm. That experiment's code and admission checks were committed, but **no new real LFM3 model-run result has been verified** for this option in this pass. Its authority flag remains false; do not mix it with already verified AdamW/Muon result receipts.

## Next gates

Independent train/dev/final-heldout source families, durable real-FFN LoRA optimizer checkpoints and recovery, per-matrix Muon scaling across parameter shapes, full-model regression tests, 69% hosted coverage, independently authenticated source/witness data and an enforced authorization gate at **every** registry mutation entrypoint. Continue keeping PRs draft and production traces/weights untouched.
