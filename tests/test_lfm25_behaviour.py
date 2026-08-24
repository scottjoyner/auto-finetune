"""Behavioural tests for LFM25Driver: retries, stalls, clipping, compaction.

All tests monkeypatch `_chat`, so no live endpoint is needed — the driver's
control flow is exercised deterministically against scripted responses.
"""
from __future__ import annotations

import pytest

from src.drivers_lfm25 import (LFM25Driver, _clip, _is_wrapped_tool_result,
                               estimate_tokens)


def _driver(**kw) -> LFM25Driver:
    d = LFM25Driver(base_url="http://127.0.0.1:1", **kw)
    return d


def _script(driver, responses):
    """Replace _chat with a scripted responder; returns the call log."""
    log = []
    seq = list(responses)

    def fake_chat(messages, max_tokens):
        log.append([dict(m) for m in messages])
        if len(seq) > 1:
            return seq.pop(0)
        return seq[0]
    driver._chat = fake_chat
    return log


# ── clipping ─────────────────────────────────────────────────────────────
def test_clip_short_text_untouched():
    assert _clip("hello", 2000) == "hello"


def test_clip_long_text_head_tail_and_marker():
    text = "A" * 3000 + "MIDDLE" + "B" * 3000
    out = _clip(text, 2000)
    assert len(out) < len(text)
    assert "clipped" in out
    assert out.startswith("A")
    assert out.endswith("B")


def test_wrap_result_clips():
    wrapped = LFM25Driver.wrap_result("x" * 9000)
    assert len(wrapped) < 4500
    # clipping happens inside the JSON string, so the wrapper stays parseable
    assert _is_wrapped_tool_result(wrapped) is True
    assert "clipped" in wrapped


# ── token estimation ─────────────────────────────────────────────────────
def test_estimate_tokens_scales_with_content():
    short = [{"role": "user", "content": "hi"}]
    long = [{"role": "user", "content": "x" * 36000}]
    assert estimate_tokens(short) < estimate_tokens(long)
    assert estimate_tokens(long) > 10000


# ── compaction ───────────────────────────────────────────────────────────
def test_no_compaction_below_threshold():
    d = _driver(num_ctx=8192)
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "u"}]
    log = _script(d, ["ok"])
    out = d._maybe_compact(msgs)
    assert out is msgs  # untouched, same object
    assert not log


def test_compaction_splices_memory_and_keeps_last_turn():
    d = _driver(num_ctx=1024)  # tiny ctx so threshold is crossed
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"user turn {i} " + "y" * 1200})
        msgs.append({"role": "assistant",
                     "content": f"assistant reply {i}"})
    log = _script(d, ["COMPRESSED-BRIEF"])
    out = d._maybe_compact(msgs)
    assert any("[compacted memory" in str(m.get("content"))
               and "COMPRESSED-BRIEF" in str(m.get("content"))
               for m in out)
    # last real user turn survives verbatim (followed by its reply)
    assert out[-2]["content"].startswith("user turn 5")
    assert out[-1]["content"] == "assistant reply 5"


def test_compaction_never_splits_tool_loop():
    d = _driver(num_ctx=1024)
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "old task " + "z" * 1200},
            {"role": "assistant", "content": "<|tool_call_start|>[bash(command=\"ls\")]<|tool_call_end|>"},
            {"role": "user", "content": '[{"output":"files"}]'},
            {"role": "user", "content": "newest user turn"}]
    log = _script(d, ["BRIEF"])
    out = d._maybe_compact(msgs)
    # newest user turn + its preceding tool exchange stay intact and in order
    tail = [m["role"] for m in out[-3:]]
    assert tail == ["assistant", "user", "user"]
    assert out[-1]["content"] == "newest user turn"


def test_compaction_failure_falls_back_to_hard_drop():
    d = _driver(num_ctx=1024)
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(6):
        msgs.append({"role": "user", "content": f"turn {i} " + "q" * 600})
        msgs.append({"role": "assistant", "content": f"reply {i}"})

    def failing_chat(messages, max_tokens):
        raise RuntimeError("endpoint down")
    d._chat = failing_chat
    out = d._maybe_compact(msgs)
    assert estimate_tokens(out) < estimate_tokens(msgs)
    assert out[0]["role"] == "system"
    assert out[-1]["content"].startswith("turn 5") or \
        out[-1]["role"] == "assistant"


# ── generate() control flow ──────────────────────────────────────────────
def test_generate_happy_path_single_call():
    d = _driver()
    good = ('<|tool_call_start|>[write(filePath="a.txt", '
            'content="hi")]<|tool_call_end|>Doing it.')
    log = _script(d, [good])
    text = d.generate([{"role": "user", "content": "make a.txt"}])
    assert text == good
    assert len(log) == 1


def test_generate_malformed_call_triggers_retry():
    d = _driver()
    bad = '<|tool_call_start|>[bash("ls")]<|tool_call_end|>'  # positional arg
    good = '<|tool_call_start|>[bash(command="ls")]<|tool_call_end|>'
    log = _script(d, [bad, good])
    text = d.generate([{"role": "user", "content": "list files"}])
    calls = d.parse_tool_calls(text)
    assert calls[0]["args"] == {"command": "ls"}
    assert len(log) == 2  # original + corrective retry
    # correction nudge appended to retried payload
    assert any("malformed tool call" in str(m.get("content"))
               for m in log[1])


def test_generate_refusal_retries_then_acts():
    d = _driver()
    refusal = ("I don't have the ability to directly write files. "
               "You can run this yourself locally.")
    acting = ('<|tool_call_start|>[write(filePath="a.txt", '
              'content="hi")]<|tool_call_end|>')
    log = _script(d, [refusal, acting])
    text = d.generate([{"role": "user", "content": "make a.txt"}])
    assert d.parse_tool_calls(text), "should end with an actionable call"
    assert len(log) == 2
    assert any("tools are real" in str(m.get("content")) for m in log[1])


def test_generate_refusal_gives_up_after_two_retries():
    d = _driver()
    refusal = "As an AI model I cannot help with that."
    log = _script(d, [refusal, refusal, refusal])
    text = d.generate([{"role": "user", "content": "do it"}])
    assert text == refusal
    assert len(log) == 3  # original + exactly 2 corrective turns


def test_genuine_final_answer_is_not_retried():
    d = _driver()
    answer = "The file contains 39 entries per your earlier tool result."
    log = _script(d, [answer])
    text = d.generate([{"role": "user", "content": "summarize"}])
    assert text == answer
    assert len(log) == 1


# ── translate ────────────────────────────────────────────────────────────
def test_translate_does_not_duplicate_tools_doc():
    d = _driver()
    msgs = [{"role": "system",
             "content": "List of tools: [{\"name\": \"bash\"}] be brief"}]
    out = d._translate(msgs, advertise_tools=True)
    joined = " ".join(str(m.get("content")) for m in out[:1])
    assert joined.count("List of tools") == 1
