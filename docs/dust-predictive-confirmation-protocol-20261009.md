# Frozen synthetic confirmation protocol — predictive direction utility

**Registered before executing the confirmation cohort.** 2026-10-09. Workstream: Draft predictive multi-direction stochastic SGD classifier, stacked on Dust/K2 PR #13.

The earlier 16-episode synthetic test split (seeds 3100–3115) has already been inspected under both an initial full-batch and a corrected mini-batch classifier fit. It is now classified as **development evaluation**, not a final untouched independent holdout. Do not present subsequent improvements on it as confirmatory evidence.

Freeze the current learned model definition exactly: synthetic quadratic task, fixed 8 pre-probe features, episode-level bootstrap ensemble head seeds 7/42/1337, 64 training episodes 1100–1163, 16 threshold-only validation episodes 2100–2115, seeded *mini-batch size 32* logistic training at 100 epochs, learning rate 0.1 and L2 = .005. No adjustment may be made to this fit after the confirmation output is inspected.

**Independent confirmation group:** 48 novel deterministic episode seeds **4100–4147**, each with 16 candidate directions. No overlap with any train, validation or previously inspected development episode IDs. Report exactly: classifier top-four beneficial precision and actual quadratic gain, momentum and curvature-aware momentum, deterministic random baseline, paired episode-bootstrap 1000 draws (fixed seed 90210), Brier score and classification accuracy. Must also report per-episode count and ensemble disagreement.

**Go/no-go (scientific, not production):** for a strong confirmatory advantage, require classifier to outperform **both** momentum controls on both top-four precision and mean gain, with a 95% paired episode-bootstrap lower CI **strictly above zero** relative to curvature-aware momentum. If any condition fails, record **NO_EVIDENCE_OF_SUPERIORITY**. Even a pass would authorize only further shadow research, not sampled-gradient selection, backprop alternatives, pretrained fine-tuning, or deployment.

The candidate generator is a toy quadratic with simulated historical gradient/curvature hints: still **not real K2 token CE**. Sampling all candidate labels makes offline top-four selection testable without missing-not-at-random selective logging. Do not infer unbiased gradient estimates from selection precision.

**Safety:** no tokenizer/model download, GPU seconds, NAS writes, hosted inference, adapter update, production scheduling, or checkpoint.
