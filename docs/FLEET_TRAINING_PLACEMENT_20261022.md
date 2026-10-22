# Fleet training placement — read-only qualification (2026-10-22)

This script determines whether x1-370 satisfies conservative conditions to
be considered for Qwen3-8B fine-tuning. It neither runs nor authorizes
training; even a PREFLIGHT_PASS has may_launch_training: false. Run from
xwing or another Tailnet host with trusted BatchMode SSH access:

    python3 scripts/fleet_training_placement.py --target x1-370

The remote command gathers only sysfs/proc counters, PCI and ROCm inventory,
the existing lease owner (not its secret fields), and the local read-only
GPU lease observer status. SSH requires host-key verification. It never
installs drivers, rescans PCI devices, changes power state, unmounts NAS,
restarts models, dispatches training, or clears archive/security holds.

Mandatory predicates for x1:
- Host identity x1-370, >=32 GiB available RAM, >=2 GiB free swap,
  memory PSI some <=2% and full <=1% (ten-second averages).
- ROCm sees gfx1201 and at least two AMD PCI display devices, distinguishing
  the absent R9700 from the still-working integrated Radeon 890M.
- The exclusive gfx1201 lease belongs to predictive-finetuning, and
  the append-only observer reports verified ledger AND healthy observation.
- At least 200 GiB free on the physically mounted x1 SSD. No NAS writes.

Observed x1-370, 2026-10-22: physical gfx1201 missing, observer critical,
and available swap below 2 GiB. Training stays NO_GO regardless of apparent
free RAM. These are independent blocking conditions. October 3 inventory
reported R9700 online; October 16 and later boot records show only iGPU.
TBGAA is a disconnected USB4 peripheral; its relationship to GPU unproven.
Do not blindly authorize an enclosure or rescan PCIe.

Other fleet nodes were checked for free memory and resident model servers;
none was admitted. Script supports only x1-370 R9700, not blanket permission.

If R9700 returns, verify identity and custody, inspect safe swap relief
without swapoff, and reconcile leases, source provenance and rollout gates.
Leave xwing training cron and stop-file HOLD in place until separate approval.
