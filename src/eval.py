"""Evaluation harness for the tool-call finetunes.

Two complementary signals:
  1. Held-out token loss / perplexity — robust, adapter-vs-base comparison.
  2. Tool-call correctness — parse the assistant turn's <tool_call> blocks and
     compare name + args against the gold held-out target.

Held-out splits are carved from the training datasets *now* (while training
runs) so they are ready the moment each adapter finishes. The curated probe
set (eval/probe.jsonl) is a small qualitative spot-check that does not depend
on the training data.
"""
from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── tool_call parsing ──────────────────────────────────────────────────────────
# Format seen in the datasets:
#   <tool_call name="read" call_id="...">
#   { "filePath": "..." }
#   \u276E\u276E\u276E            <- literal backslash-escaped "⋮⋮⋮" (NOT the real char)
#   <tool_result>...</tool_result>
# NOTE: there is NO closing </tool_call> tag; the args block ends at the
# backslash-u separator. So we capture name + the JSON up to that separator.
_SEP = "\\u276E\\u276E\\u276E"
_TOOL_CALL_RE = re.compile(
    r"<tool_call\s+name=\"([^\"]+)\"\s+call_id=\"[^\"]*\">(.*?)" + re.escape(_SEP),
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from an assistant message.

    Returns a list of {"name": str, "args": dict}. Malformed args fall back to
    {"name": str, "args": None} so the name can still be scored.
    """
    calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        name = m.group(1)
        raw = m.group(2).strip()
        args: Optional[dict] = None
        try:
            args = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            try:
                args = json.loads(raw.strip().strip("`").strip())
            except Exception:
                args = None
        calls.append({"name": name, "args": args})
    return calls


def _args_match(pred: Optional[dict], gold: Optional[dict]) -> bool:
    """Partial-arg match: every gold key/value must be present & equal in pred.

    If args are unparseable on either side, treat as match-only on name.
    """
    if gold is None or pred is None:
        return True  # can't distinguish; caller scores name separately
    if not isinstance(gold, dict) or not isinstance(pred, dict):
        return gold == pred
    for k, v in gold.items():
        if k not in pred or pred[k] != v:
            return False
    return True


@dataclass
class ToolCallScore:
    total: int = 0
    name_correct: int = 0
    exact: int = 0          # name + all args equal
    partial: int = 0        # name correct + gold args subset of pred args
    pred_calls: int = 0

    @property
    def name_accuracy(self) -> float:
        return self.name_correct / self.total if self.total else 0.0

    @property
    def partial_match(self) -> float:
        return self.partial / self.total if self.total else 0.0

    @property
    def exact_match(self) -> float:
        return self.exact / self.total if self.total else 0.0


def score_tool_calls(pred_text: str, gold_text: str) -> ToolCallScore:
    """Score predicted assistant text against gold assistant text."""
    gold = parse_tool_calls(gold_text)
    pred = parse_tool_calls(pred_text)
    sc = ToolCallScore(total=max(len(gold), 1), pred_calls=len(pred))
    # match gold calls to pred calls positionally, falling back to name search
    for i, g in enumerate(gold):
        p = pred[i] if i < len(pred) else None
        if p is not None and p["name"] == g["name"]:
            sc.name_correct += 1
            if p["args"] == g["args"]:
                sc.exact += 1
            if _args_match(p["args"], g["args"]):
                sc.partial += 1
        elif p is not None and p["name"] != g["name"]:
            # try to find a later pred with the right name
            hit = next((x for x in pred[i:] if x["name"] == g["name"]), None)
            if hit is not None:
                sc.name_correct += 1
                if _args_match(hit["args"], g["args"]):
                    sc.partial += 1
    return sc


# ── held-out split builder ─────────────────────────────────────────────────────
def build_held_out(dataset_dir: str, label: str, frac: float = 0.1, seed: int = 42) -> Path:
    """Carve a deterministic held-out split from train.<label>.jsonl.

    Writes eval/held-out-<label>.jsonl (same messages schema) and returns its
    path. Idempotent: re-running overwrites with the identical split.
    """
    src = Path(dataset_dir) / f"train.{label}.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"dataset not found: {src}")
    out_dir = Path(dataset_dir).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"held-out-{label}.jsonl"

    import random
    rng = random.Random(seed)
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    rng.shuffle(rows)
    n = max(1, int(len(rows) * frac))
    held = rows[:n]
    out.write_text("\n".join(json.dumps(r) for r in held) + "\n")
    print(f"[eval] held-out split for '{label}': {n}/{len(rows)} -> {out}")
    return out


# ── model loading + loss ───────────────────────────────────────────────────────
def load_model_and_tokenizer(model_path: str, base_model: str, rocm: bool = False):
    """Load a base model + optional PEFT adapter at model_path.

    If model_path contains an adapter_config.json, load it as a LoRA adapter
    on top of base_model; otherwise load base_model directly (for baseline).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    is_adapter = (Path(model_path) / "adapter_config.json").exists()
    base = AutoModelForCausalLM.from_pretrained(
        base_model if not is_adapter else base_model,
        torch_dtype=torch.bfloat16 if rocm else torch.float16,
        attn_implementation="sdpa",
        device_map=None,
    )
    if is_adapter:
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, model_path)
    base.eval()
    return base, tok


def eval_loss(model, tokenizer, examples: list[dict], max_seq: int = 8192) -> float:
    """Mean token cross-entropy loss over the held-out examples.

    The whole conversation is scored (assistant tokens only contribute to the
    loss via causal LMing); we report mean per-token loss.
    """
    import torch
    total_tokens = 0
    total_loss = 0.0
    with torch.no_grad():
        for ex in examples:
            text = _messages_to_text(tokenizer, ex["messages"], max_seq)
            ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_loss += float(out.loss.item()) * n
            total_tokens += n
    return total_loss / total_tokens if total_tokens else float("nan")


def _messages_to_text(tokenizer, messages: list[dict], max_seq: int) -> str:
    """Render messages using the model's chat template (no generation)."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ── end-to-end evaluation ──────────────────────────────────────────────────────
@dataclass
class EvalResult:
    label: str
    adapter: str
    n_held_out: int = 0
    loss: float = float("nan")
    perplexity: float = float("nan")
    tool: ToolCallScore = field(default_factory=ToolCallScore)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "adapter": self.adapter,
            "n_held_out": self.n_held_out,
            "loss": self.loss,
            "perplexity": self.perplexity,
            "tool_name_acc": self.tool.name_accuracy,
            "tool_partial": self.tool.partial_match,
            "tool_exact": self.tool.exact_match,
        }


def evaluate(
    adapter_path: str,
    base_model: str,
    held_out_path: str,
    rocm: bool = False,
    max_seq: int = 8192,
    gen_max_tokens: int = 512,
    loss_only: bool = False,
) -> EvalResult:
    """Run loss (+ optional tool-call generation) eval for one adapter.

    Set loss_only=True to skip model.generate() — this keeps eval from
    competing with an in-flight training run for the GPU, and is enough to
    compare perplexity across adapters. Tool-call correctness (which needs
    generation) is filled in only when loss_only=False.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model, tok = load_model_and_tokenizer(adapter_path, base_model, rocm=rocm)
    examples = [json.loads(l) for l in Path(held_out_path).read_text().splitlines() if l.strip()]

    loss = eval_loss(model, tok, examples, max_seq)
    ppl = math.exp(loss) if loss == loss else float("nan")

    tc = ToolCallScore()
    if not loss_only:
        # tool-call correctness: generate from the prefix, compare to gold assistant
        tc.total = 0
        with torch.no_grad():
            for ex in examples:
                msgs = ex["messages"]
                prefix = [m for m in msgs if m["role"] != "assistant"]
                gold_assistant = " ".join(
                    m["content"] for m in msgs if m["role"] == "assistant"
                )
                prompt = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)
                ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
                out_ids = model.generate(
                    ids, max_new_tokens=gen_max_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                gen = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
                s = score_tool_calls(gen, gold_assistant)
                tc.total += s.total
                tc.name_correct += s.name_correct
                tc.exact += s.exact
                tc.partial += s.partial
                tc.pred_calls += s.pred_calls

    return EvalResult(
        label=Path(held_out_path).stem.replace("held-out-", ""),
        adapter=adapter_path,
        n_held_out=len(examples),
        loss=loss,
        perplexity=ppl,
        tool=tc,
    )


def evaluate_baseline(
    base_model: str, held_out_path: str, rocm: bool = False, max_seq: int = 8192
) -> EvalResult:
    """Loss + tool-call eval of the un-finetuned base model (control)."""
    res = evaluate(base_model, base_model, held_out_path, rocm=rocm, max_seq=max_seq)
    res.adapter = "base"
    return res


# Labels whose adapters are produced by the bucket training queue, plus the
# merged "all together" run. Each maps to its checkpoint dir + held-out split.
EVAL_MATRIX = (
    "ssd",
    "nas5-main",
    "nas5-20260717",
    "opencode-all",
    "opencode-portfolio",
    "hermes-reasoning",
    "combined",
)


def evaluate_all(
    base_model: str,
    out_base: str,
    eval_dir: str,
    rocm: bool = False,
    max_seq: int = 8192,
    loss_only: bool = False,
) -> list[EvalResult]:
    """Score the un-finetuned base model plus every available adapter.

    An adapter is "available" if its checkpoint dir contains adapter_config.json
    (i.e. training finished). Returns a list of EvalResult, base first, then
    each present adapter, ready to print as a comparison table.
    """
    results: list[EvalResult] = []
    # Base model baseline — evaluated against each held-out set; we report it
    # once per label so the table is comparable row-by-row.
    for label in EVAL_MATRIX:
        held = Path(eval_dir) / f"held-out-{label}.jsonl"
        if not held.exists():
            continue
        adapter_dir = Path(out_base) / f"toolcall-v5-3b-{label}"
        if not (adapter_dir / "adapter_config.json").exists():
            # adapter not trained yet — skip (still list base baseline below)
            pass
        else:
            results.append(
                evaluate(str(adapter_dir), base_model, str(held), rocm=rocm,
                         max_seq=max_seq, loss_only=loss_only)
            )
        # attach the baseline (base model) for this label for comparison
        results.append(
            evaluate_baseline(base_model, str(held), rocm=rocm, max_seq=max_seq)
        )
    return results


def format_report(results: list[EvalResult]) -> str:
    """Render an EvalResult list as a markdown-ish comparison table."""
    hdr = f"{'adapter':<28} {'n':>5} {'loss':>8} {'ppl':>9} {'name%':>7} {'part%':>7} {'exact%':>8}"
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        name = r.adapter.replace(str(Path("/media/scott/data/finetune-staging/outputs/checkpoints")), "")
        lines.append(
            f"{name:<28} {r.n_held_out:>5} {r.loss:>8.4f} {r.perplexity:>9.2f} "
            f"{r.tool.name_accuracy*100:>6.1f} {r.tool.partial_match*100:>6.1f} "
            f"{r.tool.exact_match*100:>7.1f}"
        )
    return "\n".join(lines)


# ── curated qualitative probe (spot-check, not from training data) ────────────
def load_probe_set(path: str, label: "str | None" = None) -> list[dict]:
    """Load eval/probe.jsonl. Each row: {messages, checks:[{kind,expect}], label?}.

    If `label` is given, only rows whose `label` field matches (or rows with no
    label field at all, treated as generic) are returned. This lets `probe
    --label=X` scope the qualitative check to a bucket's domain probes while
    still running the generic ones.
    """
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    if label is None:
        return rows
    scoped = [r for r in rows if r.get("label") == label]
    generic = [r for r in rows if "label" not in r]
    return scoped + generic


def _check_evidence(gen_text: str, checks: list[dict]) -> tuple[int, int, list[dict]]:
    """Return (passed, total, per-check detail) for a probe's checks.

    Each check: kind in {path, tool, substring}. `tool` is satisfied if the
    expected tool name appears inside a <tool_call name="...">; path/substring
    are satisfied by a case-insensitive substring match anywhere in the
    generated text. This is a qualitative spot-check, not an exact-match metric.
    """
    gen_lower = gen_text.lower()
    passed = 0
    total = len(checks)
    detail = []
    calls = parse_tool_calls(gen_text)
    call_names = {c["name"].lower() for c in calls}
    for c in checks:
        kind = c.get("kind", "substring")
        expect = c["expect"]
        ok = False
        if kind == "tool":
            ok = expect.lower() in call_names
        else:  # path or substring
            ok = expect.lower() in gen_lower
        if ok:
            passed += 1
        detail.append({"kind": kind, "expect": expect, "ok": ok})
    return passed, total, detail


@dataclass
class ProbeResult:
    adapter: str
    n_probes: int = 0
    checks_passed: int = 0
    checks_total: int = 0
    details: list = field(default_factory=list)

    @property
    def check_rate(self) -> float:
        return self.checks_passed / self.checks_total if self.checks_total else 0.0

    def as_dict(self) -> dict:
        return {
            "adapter": self.adapter,
            "n_probes": self.n_probes,
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "check_rate": self.check_rate,
        }


def grade_probe(
    adapter_path: str,
    base_model: str,
    probe_path: str,
    rocm: bool = False,
    max_seq: int = 8192,
    gen_max_tokens: int = 512,
    label: "str | None" = None,
) -> ProbeResult:
    """Generate answers for the curated probe set and grade expected evidence.

    Requires model generation (GPU). Run when training is idle, or accept that
    it competes with an in-flight run for the iGPU. If `label` is given, only
    probes tagged with that label (plus generic, unlabeled ones) are run.
    """
    import torch
    model, tok = load_model_and_tokenizer(adapter_path, base_model, rocm=rocm)
    probes = load_probe_set(probe_path, label=label)
    res = ProbeResult(adapter=adapter_path)
    res.n_probes = len(probes)
    with torch.no_grad():
        for p in probes:
            msgs = p["messages"]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            out_ids = model.generate(
                ids, max_new_tokens=gen_max_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            gen = tok.decode(out_ids[0][ids.shape[1]:], skip_special_tokens=True)
            passed, total, detail = _check_evidence(gen, p.get("checks", []))
            res.checks_passed += passed
            res.checks_total += total
            res.details.append({"prompt": msgs[-1]["content"], "passed": passed,
                                "total": total, "checks": detail})
    return res


def grade_probe_baseline(base_model: str, probe_path: str, rocm: bool = False,
                         max_seq: int = 8192, gen_max_tokens: int = 512,
                         label: "str | None" = None) -> ProbeResult:
    res = grade_probe(base_model, base_model, probe_path, rocm=rocm,
                      max_seq=max_seq, gen_max_tokens=gen_max_tokens, label=label)
    res.adapter = "base"
    return res


def write_report(
    results: list[EvalResult],
    probe_base: "ProbeResult | None" = None,
    probe_adapter: "ProbeResult | None" = None,
    out_dir: str = "/media/scott/data/finetune-staging/eval-reports",
    tag: str = "",
) -> str:
    """Persist an eval comparison (loss table + probe scores) to local disk.

    Writes both a human-readable .md and a machine-readable .json. Returns the
    markdown path. Local disk only (avoids NFS vault flakiness).
    """
    import datetime
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = f"{tag}-{stamp}" if tag else stamp

    md = [f"# Finetune Eval Report — {stamp}", ""]
    md.append("## Held-out loss / tool-call (base vs adapters)")
    md.append("")
    md.append(format_report(results))
    md.append("")
    if probe_base is not None or probe_adapter is not None:
        md.append("## Curated probe (qualitative evidence check)")
        md.append("")
        if probe_base is not None:
            md.append(f"- base: {probe_base.checks_passed}/{probe_base.checks_total} "
                      f"checks passed ({probe_base.check_rate*100:.1f}%)")
        if probe_adapter is not None:
            md.append(f"- adapter: {probe_adapter.checks_passed}/{probe_adapter.checks_total} "
                      f"checks passed ({probe_adapter.check_rate*100:.1f}%)")
        md.append("")

    md_path = os.path.join(out_dir, f"report-{slug}.md")
    json_path = os.path.join(out_dir, f"report-{slug}.json")
    md_text = "\n".join(md)
    with open(md_path, "w") as f:
        f.write(md_text)
    payload = {
        "stamp": stamp,
        "held_out": [r.as_dict() for r in results],
        "probe_base": probe_base.as_dict() if probe_base else None,
        "probe_adapter": probe_adapter.as_dict() if probe_adapter else None,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    return md_path


def best_adapter(results: list[EvalResult], metric: str = "loss") -> "EvalResult | None":
    """Pick the adapter (excludes 'base') with the best held-out metric.

    metric='loss' -> lowest loss; metric='tool_exact' -> highest exact match.
    Returns None if no adapters present.
    """
    adapters = [r for r in results if r.adapter != "base"]
    if not adapters:
        return None
    if metric == "loss":
        return min(adapters, key=lambda r: r.loss if r.loss == r.loss else float("inf"))
    if metric in ("tool_exact", "tool_partial", "tool_name"):
        key = {"tool_exact": "exact_match", "tool_partial": "partial_match",
               "tool_name": "name_accuracy"}[metric]
        return max(adapters, key=lambda r: getattr(r.tool, key))
    return None


def sanity_check_adapters(
    base_model: str,
    out_base: str,
    labels: list[str],
    rocm: bool = False,
) -> dict[str, str]:
    """Confirm each finished adapter actually loads + runs a forward pass.

    Returns {label: "ok" | "missing" | "error: <msg>"}. Catches wedged /
    empty / corrupt checkpoints immediately after training, before eval time.
    Uses a tiny dummy input so it's fast and doesn't need the dataset.
    """
    import torch
    status: dict[str, str] = {}
    for label in labels:
        adapter_dir = Path(out_base) / f"toolcall-v5-3b-{label}"
        if not (adapter_dir / "adapter_config.json").exists():
            status[label] = "missing"
            continue
        try:
            model, tok = load_model_and_tokenizer(str(adapter_dir), base_model, rocm=rocm)
            ids = tok("hello", return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                out = model(ids)
            assert out.logits is not None
            status[label] = "ok"
        except Exception as e:  # noqa: BLE001
            status[label] = f"error: {type(e).__name__}: {e}"
    return status
