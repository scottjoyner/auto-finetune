# Preserve the native LFM3 tool/chat protocol in auto-finetune

**Draft research: stacked on PR #68 (pinned pretrained LFM3 loader).**

Inspection of the current LFM3 PEFT path found two independent bugs:

1. The trainer unconditionally replaces the native 12,129-character LFM3
   chat template with a simplified template that drops native tool calls,
   reasoning modes, role rendering and generation handling.
2. Its tokenizer fallback uses an incorrect end-of-message token.

The pinned LFM3 path now refuses any failed native AutoTokenizer load;
it cannot silently fall back to a generic text tokenizer. The native
chat_template.jinja file must be exactly SHA-256
f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818,
must match the loaded tokenizer's chat_template, and must preserve
official bos/eos/pad token names and numeric IDs. The guard only
validates the template; it never replaces the tokenizer template.

The pinned tokenizer.json SHA-256 was also corrected to
7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701.
This is the SHA-256 of **actual downloaded bytes**, independently fetched
using the official Hugging Face revision URL for
LiquidAI/LFM3-1.7B@40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd.
The local 17,908,635-byte tokenizer.json matched those bytes exactly.
The verified config.json, modeling_lfm3.py and chat_template.jinja
also matched their official revision downloads byte-for-byte.

Note: Hugging Face response ETag values must not be assumed to be
content SHA-256 hashes. Confirm actual bytes when deriving SHA-256 pins.

## Verification performed
- 27 focused CPU-only tests passed in the independent review staging copy,
  including original PR #68 loader/tensor tests and new negative tests for
  missing/tampered templates, wrong EOS/IDs and a rewritten serializer.
- Offline, local AutoTokenizer.from_pretrained loaded the 1.7B tokenizer
  successfully without allocating model weights, and the strict native
  template guard accepted its source bytes and token IDs.

## Remaining holds
No full 3.46 GB model checkpoint load, training step, optimizer comparison,
adapter promotion or task-level benchmark was executed by this change.
A full source + weight digest verification, Transformers/ROCm compatibility
check and independent real-model inference/held-out results remain required.
No production routing, training, private-trace ingestion or service mutation.
