# Opt-in Cargo artifact placement on the separate x1-370 SSD

**Date:** October 31, 2026. **Incident:** auto-finetune #106.
**Scope:** future explicit Cargo commands; not an automatic policy switch.

## Problem and verified mount layout

Following the controlled reclaim of ~21.134 GiB of old root-backed
Cargo incremental caches, `x1-370` again had ~26 GiB ordinary-user
root capacity. The existing `sensorgraft-path-topk-20261030`
compiler/linker process was deliberately left running. Its
`/proc/<pid>/environ` explicitly specified:
`CARGO_TARGET_DIR=/home/scott/git/sensorgraft-path-topk-20261030/target`
and `CARGO_INCREMENTAL=0`. Disabling incremental compilation
does **not** move compiled binaries or debug artifacts off root.

The **separate mounted SSD** is `/media/scott/SSD_4TB`
(`/dev/nvme0n1p1`, `st_dev=66305`); the exhausted root is
`/dev/nvme1n1p6` (`st_dev=66312`). At planning time,
ordinary-user space was roughly **25.446 GiB on root**
and **427.402 GiB on the SSD**.

## New command, read-only by default

`src.ssd_cargo_runner` plans one deterministic, per-repository
SSD target directory using the repository's absolute path hash.
It validates the project manifest, real SSD mount identity,
non-symlink destination and device separation, and checks both
filesystems before any build.

From `research/lfm3-refusal-validation`:

```bash
PY=/home/scott/lfm3-research/train-venv/bin/python
REPO=/home/scott/git/sensorgraft-path-topk-20261030

# Read-only: prints the exact target directory, command and headroom.
$PY -m src.ssd_cargo_runner --repo "$REPO" --command check

# Explicit opt-in for a NEW run ONLY; does not alter an existing job.
$PY -m src.ssd_cargo_runner --repo "$REPO" --command check --execute
```

The execution mode requires at least **20 GiB ordinary-user root
space** and **80 GiB on the SSD** by default. Planned 5 GiB
SSD write plus 32 GiB SSD reserve must also fit. Those are
preflight checks, **not filesystem reservations**.

When explicitly executed, only the new subprocess receives:

- `CARGO_TARGET_DIR=/media/scott/SSD_4TB/lfm3-cargo-targets/<repo-hash>/target`
- `CARGO_INCREMENTAL=0`
- `TMPDIR=/media/scott/SSD_4TB/lfm3-cargo-targets/<repo-hash>/tmp`
- `RUSTC=/home/scott/.cargo/bin/rustc` (required because the Fleet
  Commander shell lacks rustc on PATH)

The command uses an explicit installed Cargo executable,
`--locked --offline` to avoid lockfile updates and new network
dependency resolution, and one of `check`, `test`, `build`
or `bench`. No general shell command is accepted as an action.
Other worktrees and running processes retain their original
environment and files. Existing `target` directories are neither
moved nor deleted.

## Actual end-to-end test

A tiny dependency-free synthetic Rust library was created in a
temporary source directory and compiled with the explicit
`--execute --command check` path. The first attempt exposed
an environment issue: `cargo` ran but could not find `rustc`
on Fleet Commander's PATH. The runner was amended to supply the
installed Rust compiler explicitly.

The subsequent actual `cargo check --locked --offline` **passed
(exit 0)** in ~0.07 seconds. The compiler emitted one
`liblfm3_ssd_smoke-*.rmeta` file on **device 66305**, while
root remained **device 66312** and the synthetic source directory
never acquired a `target` subdirectory. The synthetic temporary
source was removed after the check; small isolated smoke build
artifacts are left on the SSD.

The full synthetic regression includes low-space fail-closed
behavior, SSD mount identity changes, per-repository namespace
separation, symlink/manifest checks, explicit compiler path,
no-write planning, and actual environment propagation to the
opted-in Cargo child process.

## Limitations and next actions

This is an **opt-in launcher**, not a global Cargo or Rustup
configuration change. Existing builds that specify a root-backed
`CARGO_TARGET_DIR` continue writing there. Cargo home registry
caches, independent build scripts, outputs explicitly written by
benchmarks, and processes outside the launcher may still use root.
The wrapper is not a security sandbox against malicious Cargo
build scripts or adversarial filesystem races.

To sustain the recovery, new SensorGraft build harnesses should
adopt the opt-in wrapper or an equivalent pinned-target policy
after their owners verify dependency availability and expected
artifact paths. Do not migrate or purge old targets automatically.
Issue #106 should remain open under capacity observation until
those launches are consistently redirected. No LFM3 model
checkpoint or human annotation was created or changed by this work.
