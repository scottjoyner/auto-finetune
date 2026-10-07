# Dust/K2 arXiv draft

This directory contains an **independent technical-report draft** describing
the K2-Horizon-0.9B activation-perturbation experiments in this repository.

It is intentionally conservative:

- the work is described as **Dust-inspired**, not a reproduction of the
  complete Q Labs Dust method;
- the current estimator's shared directions are distinguished from Dust's
  independent per-token perturbations;
- negative/neutral heldout results are included;
- no authorship or collaboration with the Dust authors is implied;
- private prompts, responses, token IDs, credentials, and host-specific
  secrets must not be added.

## Files

- `main.tex` — arXiv-compatible article draft using standard packages.
- `references.bib` — provisional bibliography, including the Q Labs Dust
  research report and public code.
- `../../docs/dust-k2-next-slice.md` — experimental plan that should be
  completed before treating the draft as submission-ready.

## Before submission

1. Finalize authors and affiliations.
2. Replace provisional Dust web citation if/when an official arXiv identifier
   is available.
3. Generate tables from immutable public-safe manifests rather than hand
   transcription.
4. Complete tokenwise-parity and expanded quality experiments.
5. Add a public code/data-availability URL.
6. Run an independent technical review for unsupported claims.
7. Contact the Dust authors with the evidence package and draft as an
   independent extension; discuss collaboration/authorship only if mutually
   appropriate.

The draft deliberately does not bundle a custom arXiv style file; it uses
standard LaTeX packages commonly available in arXiv's TeX environment.
