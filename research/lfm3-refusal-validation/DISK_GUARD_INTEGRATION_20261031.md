# Hard-stop capacity guards for LFM3 evaluation — October 31, 2026

**Scope:** This change wires the previously advisory free-space check into
four actual research entry points. It does not perform model inference, clean
up disk, delete files, change filesystem reserves, or alter original results.

## Protected entry points

- `src.xstest_450_eval`: before source/manifest/output creation, before
  each checkpoint load, and immediately before appending each response.
- `src.arc_full_likelihood_eval`: before the ARC source, new manifest
  and journal, before each checkpoint load, and before each append.
- `src.arc_capability_eval`: before source / model loading.
- `src.three_checkpoint_eval`: before hashing/loading the first checkpoint
  and before each subsequent checkpoint load.

The `src.research_job_guard.require_research_write_capacity` helper
resolves the nearest existing parent of a prospective output directory.
It verifies the local working filesystem **even when the output is on a
separate mounted NAS**, and checks that destination separately if the
devices differ. Policy is a minimum **20 GiB of unprivileged free space**,
with an assumed **5 GiB planned write + 8 GiB reserve**; effective
requirement is the maximum of the two checks. A failed check raises
`ResearchWriteBlocked` before the guarded stage.

### Actual host and regression evidence

On `x1-370`, an attempted XSTest CLI start against the existing pinned
full-run directory failed before starting inference:

```text
ResearchWriteBlocked: LFM3 research write BLOCKED on local filesystem:
6.007 GiB free, 20.000 GiB required.
Recover and verify storage capacity; no model run started.
```

No private benchmark record was added or edited by this attempt.
Full study test suite: **356/356 passing**, including synthetic
tests of rejection before loading input, accepting sufficient simulated
capacity, invalid paths, and rejection across all four entry points.

**Limits:** The guard is not a filesystem reservation, global writer
restriction, or daemon. Independent Rust jobs, other repositories and
running processes are outside its control. Filesystem conditions can
change between checks. Operators must still restore stable headroom
on root and validate the NAS before large research jobs resume.

Frozen XSTest output remains human-ungraded: two independent reviewer
assignments, each 0/1,350. No checkpoint promotion or behavioral-safety
claim is authorized by adding these operational checks.
