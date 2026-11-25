# LFM3-1.7B: combined pinned checkpoint + native chat CPU smoke

This is an optional, explicitly opted-in **functional** CPU smoke test.
It is not a training run, throughput claim, optimizer comparison, source
publisher signature, or permission to promote weights.

The combined PR #68 loader and PR #69 native tokenizer code was exercised
read-only on x1-370 during October 29, 2026 EDT, using the locally cached
LiquidAI/LFM3-1.7B snapshot at pinned revision
40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd, Transformers 5.17.0
isolated import overlay and PyTorch 2.9.1+rocm6.4 with visible GPUs disabled.

Observed:
- Official snapshot full file-digest and 286-tensor SafeTensors checks passed.
- Exact native tokenizer, tool-capable Jinja and BOS/EOS/PAD checks passed.
- Verified loader accepted the full BF16 checkpoint.
- Fixed public test phrase "Hello world" generated two tokenizer tokens.
- One CPU-only forward pass emitted shape (1, 2, 128000); every logit
  was finite. No scores, benchmark repetitions, throughput, train losses,
  optimizer steps, adapter writes or production mutations were generated.
- The first isolated attempt failed because the test harness applied
  the LFM3 RoPE config patch after loading the tokenizer. Correcting the
  order to patch before configuration and tokenizer construction resolved
  the failure. The production train.main() already patches before load.

A standalone, opt-in reproducibility script is provided at
research/lfm3_cpu_checkpoint_smoke.py. Run only on an admitted research
node with the required approved offline Python dependencies, enough RAM,
and an external timeout/process resource budget:

    timeout 100s python3 research/lfm3_cpu_checkpoint_smoke.py \
      --checkpoint /local/reviewed/model-snapshot --run

Without --run the script returns a blocked JSON receipt and does not
import the model loader. It forces offline mode, masks GPUs, sets PyTorch
CPU threads to 2, validates pinned source and native template and
evaluates a constant public prompt. It never transmits file paths,
inference text, or exception payloads in its JSON receipt.

**Cautions:** The script relies on trust_remote_code only after the
pretrained source bytes have been pinned and validated. Local SHA-256
matching does not independently establish publisher signatures or
prove remote execution source custody. A successful functional smoke
is NOT a validated GPU forward/backward pass or a held-out gain. At least
three independent paired training seeds, quality evaluation, and actual
per-node resource benchmarks are still needed.
