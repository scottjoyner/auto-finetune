"""Validate the heuristic classifier against a hand-labeled sample (CPU-only).

`analyze` classifies every session with a deterministic heuristic
(`classify_bucket` / `classify_difficulty`) and flags failures via
`_is_error` on tool outputs. This module lets us MEASURE how
trustworthy that is: it builds a label sheet (predictions + a compact,
human-readable view of each sampled session), you hand-label the truth,
and `score` reports bucket accuracy + per-bucket precision/recall and
failure-detection precision/recall.

This is the one check that could invalidate strata weighting, the verify
pass-rate, and the contrastive pairs -- if the heuristics are wrong, we are
weighting / mining the wrong thing. Pure CPU; reads only cleaned/.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

from src.analyze import (
    CODE_TOOLS,
    EDIT_TOOLS,
    SHELL_TOOLS,
    classify_bucket,
    classify_difficulty,
    extract_features,
)
from src.verify import load_session_map


def _view(rec: dict) -> tuple[str, list[str]]:
    """First user request + a compact tool-action trace, for hand-labeling."""
    msgs = rec.get("messages", [])
    actions: list[str] = []
    first_user = ""
    for m in msgs:
        role = m.get("role")
        for p in m.get("parts", []):
            if p.get("tool"):
                tool = p.get("tool")
                inp = p.get("input") or {}
                if tool in EDIT_TOOLS:
                    tgt = os.path.basename(inp.get("filePath") or inp.get("path")
                                            or inp.get("file") or "?")
                    actions.append(f"{tool}({tgt})")
                elif tool in SHELL_TOOLS or tool in CODE_TOOLS:
                    cmd = str(inp.get("command") or inp.get("code") or "")[:50]
                    actions.append(f"{tool}:{cmd}")
                else:
                    actions.append(str(tool))
            elif p.get("type") == "text" and role == "user" and not first_user:
                first_user = str(p.get("text") or "")[:200]
    return first_user, actions


def build_sheet(cleaned_dir: str, failures_path: str, out_path: str,
                n_fail: int = 15, n_ok: int = 15, seed: int = 7) -> int:
    """Write an ndjson label sheet (prediction + view) for a stratified sample."""
    sessions = load_session_map(cleaned_dir)
    failure_ids = set()
    if os.path.exists(failures_path):
        with open(failures_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    failure_ids.add(json.loads(line).get("session_id"))

    rng = random.Random(seed)
    fail_ids = sorted(failure_ids)
    ok_ids = sorted(s for s in sessions if s not in failure_ids)
    sample_fail = rng.sample(fail_ids, min(n_fail, len(fail_ids)))
    sample_ok = rng.sample(ok_ids, min(n_ok, len(ok_ids)))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for sid in sample_fail + sample_ok:
            rec = sessions[sid]
            f0 = extract_features(rec)
            first_user, actions = _view(rec)
            sheet = {
                "session_id": sid,
                "source": rec.get("source", ""),
                "pred_bucket": classify_bucket(f0),
                "pred_is_error": f0["has_error"],
                "pred_difficulty": classify_difficulty(f0),
                "n_turns": f0["n_turns"],
                "tool_hist": f0["tool_hist"],
                "intent": f0["intent"],
                "first_user_text": (f0["first_user_text"] or first_user)[:200],
                "actions": actions[:25],
                "error_snippet": f0["error_snippet"][:160],
                # two error axes are recorded separately:
                #  - true_cmd_error   : did a TOOL COMMAND actually fail?
                #                       (what `pred_is_error` = has_error detects)
                #  - true_is_error    : did the SESSION ultimately fail?
                #                       (original hand-label meaning)
                "true_bucket": "", "true_difficulty": "",
                "true_cmd_error": "", "true_is_error": "",
                "true_session_failed": "",
            }
            f.write(json.dumps(sheet) + "\n")
            n += 1
    return n


def score(sheet_path: str) -> dict:
    """Score the sheet once `true_*` fields are filled in.

    Reads every record; records without a `true_bucket` are skipped.
    Returns a metrics dict (also prints a readable report).
    """
    rows = [json.loads(l) for l in open(sheet_path) if l.strip()]
    rows = [r for r in rows if r.get("true_bucket")]

    bucket_tp = Counter()
    bucket_pred = Counter()
    bucket_true = Counter()
    correct = 0
    for r in rows:
        p, t = r["pred_bucket"], r["true_bucket"]
        bucket_pred[p] += 1
        bucket_true[t] += 1
        if p == t:
            correct += 1
            bucket_tp[t] += 1
    n = len(rows)
    bucket_report = {}
    for b in sorted(set(list(bucket_pred) + list(bucket_true))):
        tp = bucket_tp[b]
        prec = tp / bucket_pred[b] if bucket_pred[b] else 0.0
        rec = tp / bucket_true[b] if bucket_true[b] else 0.0
        bucket_report[b] = {"precision": round(prec, 2),
                            "recall": round(rec, 2),
                            "support": bucket_true[b]}

    def _err_axis(rows, pred_key, true_key):
        """Confusion matrix for one (pred, true) error axis.

        Sessions missing the ``true_*`` label are skipped on that axis
        (so the two axes can be scored independently as the sheet grows).
        ``pred`` is always ``pred_is_error`` (the command-level signal
        ``has_error``); the axes differ only in what truth they are
        checked against.
        """
        tp = fp = fn = tn = 0
        for r in rows:
            if true_key not in r or r[true_key] in ("", None):
                continue
            p = bool(r[pred_key])
            t = bool(r[true_key])
            if t and p:
                tp += 1
            elif t and not p:
                fn += 1
            elif (not t) and p:
                fp += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return {"precision": round(prec, 3), "recall": round(rec, 3),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    # Axis 1: command-level error (what has_error actually detects).
    cmd = _err_axis(rows, "pred_is_error", "true_cmd_error")
    # Axis 2: session-level failure (the original hand-label meaning).
    sess = _err_axis(rows, "pred_is_error", "true_is_error")

    diff_ok = sum(1 for r in rows if r["pred_difficulty"] == r["true_difficulty"])

    metrics = {
        "n_labeled": n,
        "bucket_accuracy": round(correct / n, 3) if n else 0.0,
        "bucket_per_bucket": bucket_report,
        "cmd_error": cmd,
        "session_failure": sess,
        "difficulty_accuracy": round(diff_ok / n, 3) if n else 0.0,
    }
    _print_report(metrics)
    return metrics


def _print_report(m: dict) -> None:
    print(f"labeled sessions : {m['n_labeled']}")
    print(f"bucket accuracy : {m['bucket_accuracy']}")
    print("per-bucket (precision / recall / support):")
    for b, v in m["bucket_per_bucket"].items():
        print(f"  {b:18s} P={v['precision']:.2f} R={v['recall']:.2f} n={v['support']}")
    c = m["cmd_error"]
    print(f"COMMAND-error   : precision={c['precision']} recall={c['recall']} "
          f"(tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']})  [pred=has_error]")
    s = m["session_failure"]
    print(f"SESSION-failure : precision={s['precision']} recall={s['recall']} "
          f"(tp={s['tp']} fp={s['fp']} fn={s['fn']} tn={s['tn']})  "
          f"[pred=has_error, mis-scoped]")
    e = m["error_matrix"]
    print(f"error detection  : precision={m['error_precision']} recall={m['error_recall']} "
          f"(tp={e['tp']} fp={e['fp']} fn={e['fn']} tn={e['tn']})")
    print(f"difficulty acc   : {m['difficulty_accuracy']}")
