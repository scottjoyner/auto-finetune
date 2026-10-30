#!/usr/bin/env python3
"""Evaluate final-answer compliance separately from key appearing somewhere in output.

This is a *synthetic* three-case smoke test, not general reasoning accuracy.
"""
import json
import pathlib
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent


def normalize_final(text):
    """Handle an optional reasoning terminator, Unicode dashes, and whitespace.

    Do not strip prose, markdown or embedded explanatory text: those are genuine
    failures of an exact-answer requirement.
    """
    last = text.rsplit("</think>", 1)[-1].strip()
    last = unicodedata.normalize("NFKC", last)
    return last.replace("\N{NON-BREAKING HYPHEN}", "-").strip(' \"\'.,')


def score(path):
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "completed":
        raise ValueError(f"not completed: {path.name}")
    cases = receipt.get("results")
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError(f"expected exactly three fixtures: {path.name}")
    results = []
    for case in cases:
        actual = case["generated"]
        expected = case["expected"]
        if not actual or not expected:
            raise ValueError(f"empty result: {path.name}")
        results.append({
            "row": case["row"],
            "final_exact": normalize_final(actual) == expected,
            "substring_present": expected in actual,
            "final_answer_excerpt": normalize_final(actual)[:120],
        })
    return {"receipt": path.name, "cache_type": receipt["cache_type"],
            "case_count": len(results), "exact_after_reasoning_count": sum(r["final_exact"] for r in results),
            "substring_count": sum(r["substring_present"] for r in results), "cases": results}


def main():
    sources = sorted(p for p in ROOT.glob("quality-*.json") if p.name != "quality-scored.json")
    if not sources:
        raise ValueError("missing quality receipts")
    results = [score(path) for path in sources]
    target = ROOT / "quality-scored.json"
    target.write_text(json.dumps({"schema": "lfm3.synthetic-retrieval-score.v1",
                                  "notice": "Three fixed retrieval fixtures per cache type; exact final answer is not broad accuracy.",
                                  "scores": results}, indent=2) + "\n")
    for result in results:
        print(result["cache_type"], result["exact_after_reasoning_count"], "/", result["case_count"], "strict final answers")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        sys.exit(1)
