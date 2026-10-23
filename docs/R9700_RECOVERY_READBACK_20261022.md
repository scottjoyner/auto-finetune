# R9700 physical recovery readback

The original x1-370 R9700 used OCuLink (PCI root 00:03.1,
switch 1002:1478/1479, GPU 1002:7551). This is unrelated to the
MINISFORUM USB4 TBGAA / RTX3090 experiment from 2026-10-16.

## Read-only report from xwing

    cd ~/git/auto-finetune
    python3 -m scripts.r9700_recovery_report

Uses existing local Govee API credential on xwing only with safe key
permissions. Issues only HTTPS GET requests to retrieve the Plug C
relay state, then uses trusted, batch-mode SSH to read x1 PCI ancestry,
ROCm, GPU lease, observer health/freshness, RAM, swap, PSI and SSD.
Does not reveal the API key or other device identities, and contains
no power-control operation.

Plug C relay ON/online proves the shared relay, not downstream
Corsair SF1000 power, ATX-on, auxiliary PCIe connectors, card seating,
or OCuLink continuity. Plug C also powers production tower hosts.
Never toggle it as a GPU diagnostic.

NO_GO (exit 78) means hardware or telemetry preflight blocked.
PHYSICAL_REVIEW_REQUIRED (exit 0) means the software witnesses passed,
not that training is authorized. The emitted may_launch_training field
is always false. Hardware custody and operational release remain separate.

## Observed 2026-10-22

Plug C relay ON/online; original OCuLink root 00:03.1, Navi switches and
R9700 endpoint absent. GPU observer ledger verified but current
hardware observation CRITICAL. Free swap less than 2 GiB, although
Neo4j Java accounts for ~7 GiB cold swapped pages and RAM/PSI remain
healthy. Do not perform swapoff or restart Neo4j for cosmetic free swap.

Archived kern.log.2.gz shows the R9700 at c7:00.0 through October 6
14:01 EDT. October 10 and later boot captures lack the original PCI
route; exact removal or power-loss time between those points unknown.

Physical actions still necessary: identify actual card/dock location,
check SF1000 rocker/ATX-on/auxiliary power and OCuLink cable while
appropriately de-energized. Never blindly rescan PCIe, authorize USB4,
cycle shared Plug B/C, reload drivers, restart training or release
GPU leases.

See scottjoyner/knowledge#179 and the canonical x1-370 eGPU runbook.
