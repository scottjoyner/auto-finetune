# Xwing Qwen trainer Python environment — post-Ubuntu upgrade hold

October 24, 2026 — RUNTIME_RESTORE_HOLD; no repair or training.

The existing training environment is located at
/media/scott/data/finetune-venv.
Its pyvenv.cfg was created for Python **3.12.3** and identifies the
original executable /usr/bin/python3.12. After the Ubuntu 26.04
upgrade, that original binary is **missing**. The venv's
bin/python3 symlink resolves through /usr/bin/python3 to **3.14.4**.

The original /lib/python3.12/site-packages directory is still
present with **134** dist-info metadata directories and package
folders for Torch, Transformers, PEFT, TRL, Accelerate, Datasets,
Safetensors and Tokenizers. But the active Python 3.14 site-packages
directory inside the venv is absent, and the venv interpreter reports
these training packages as unavailable.

Do NOT try to make the old environment work by overriding sys.path,
changing Python symlinks in place, copying binary extension packages
between interpreter versions or installing into the system interpreter.
PyTorch, NumPy and other compiled wheels may have a Python ABI and
kernel/ROCm driver compatibility requirement; static package presence
is insufficient to qualify a restore.

## Read-only repeatable audit

    python3 -m scripts.qwen_python_runtime_audit

The script checks the bounded pyvenv.cfg and site-packages paths.
It probes interpreter version only through the executable with
-I -S, disabling user environment and site-packages, and never imports
Torch, PEFT or any model, modifies an environment, or runs pip.
It always returns RUNTIME_RESTORE_HOLD exit78 with all
environment-modification, training, resume and pruning permissions
false, even if its structural checks happen to pass.

## Recovery and future qualification

1. Preserve the old environment and its 134 distribution records
   read-only, together with existing model, dataset and checkpoint
   hashes. The physical source and all four training/pruning holds
   remain unchanged.
2. Obtain trusted, verified Python 3.12 binaries or choose an
   independently supported new Python version and compatible
   pinned Torch, Transformers, TRL, PEFT, bitsandbytes/ROCm packages.
3. Prepare a separately named *disposable* venv, not an in-place
   replacement, with package source and wheel checksum receipts.
   This step requires its own reviewed environment-build plan.
4. Run CPU-only synthetic imports and model-free tests, then
   hardware/driver qualification on a separate leased execution path.
5. Complete independent 22-file physical checkpoint backup,
   NFS publication/durability verification and isolated optimizer/
   scheduler/RNG/trainer-state restore before enabling training.
6. Add a real guarded checkpoint resume entrypoint and maintain
   the stop-file, admission HOLD and rollback controls throughout.

The old venv's current ability to run src.training_admission and
produce HOLD exit78 does NOT prove it can execute a trainer.

Track restoration under scottjoyner/auto-finetune issue #37.
