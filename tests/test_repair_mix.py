import json

from src.repair_mix import build_dpo_mix, validate_messages_mix


def _repair(name, args):
    return {"name": name, "arguments": args}


def test_build_dpo_mix_roundtrip(tmp_path):
    repairs = tmp_path / "repairs.jsonl"
    repairs.write_text(
        json.dumps(
            {
                "session": "s1",
                "bucket": "multi-file-refactor",
                "error_tool": "write_file",
                "error_marker": "error:",
                "target": "x.py",
                "prompt_messages": [
                    {"role": "user", "content": "write x.py"},
                    {"role": "assistant", "content": "ok"},
                ],
                "chosen_call": _repair("write_file", {"path": "x.py", "content": "good"}),
                "rejected_call": _repair("write_file", {"path": "x.py", "content": "bad"}),
            }
        )
        + "\n"
    )
    out = tmp_path / "repairs.dpo.jsonl"
    n = build_dpo_mix(repairs, out)
    assert n == 1
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"prompt", "chosen", "rejected"}
    assert row["chosen"][0]["tool_calls"][0]["function"]["name"] == "write_file"
    assert (
        json.loads(row["chosen"][0]["tool_calls"][0]["function"]["arguments"])["content"]
        == "good"
    )
    assert (
        json.loads(row["rejected"][0]["tool_calls"][0]["function"]["arguments"])["content"]
        == "bad"
    )


def test_build_dpo_mix_normalizes_raw_prompt(tmp_path):
    # repair pairs store raw live-store messages (parts-based), not the
    # OpenAI `messages` shape the chat template expects.
    repairs = tmp_path / "repairs.jsonl"
    repairs.write_text(
        json.dumps(
            {
                "session": "s2",
                "bucket": "file",
                "error_tool": "execute_code",
                "error_marker": "error:",
                "target": "run.py",
                "prompt_messages": [
                    {"id": "h1", "role": "user", "agent": "cli",
                     "parts": [{"type": "text", "text": "run the script"}]},
                    {"id": "h2", "role": "assistant",
                     "parts": [{"type": "tool", "tool": "execute_code",
                                "input": {"code": "x=1"},
                                "call_id": "c1"}]},
                    {"id": "h3", "role": "tool", "agent": "cli",
                     "parts": [{"type": "tool_result", "output": "1",
                                "call_id": "c1"}]},
                ],
                "chosen_call": _repair("execute_code", {"code": "print(1)"}),
                "rejected_call": _repair("execute_code", {"code": "print(2)"}),
            }
        )
        + "\n"
    )
    out = tmp_path / "repairs.dpo.jsonl"
    build_dpo_mix(repairs, out)
    row = json.loads(out.read_text().splitlines()[0])
    roles = [m["role"] for m in row["prompt"]]
    # system prepend + user + assistant(tool_call) + tool result
    assert roles[0] == "system"
    assert "user" in roles
    assert "assistant" in roles
    assert "tool" in roles
    # every message the chat template sees must carry `content`
    assert all("content" in m for m in row["prompt"])
    # assistant tool_call preserved in normalized form
    asst = next(m for m in row["prompt"] if m["role"] == "assistant")
    assert asst["tool_calls"][0]["function"]["name"] == "execute_code"


def test_build_dpo_mix_empty(tmp_path):
    repairs = tmp_path / "repairs.jsonl"
    repairs.write_text("")
    out = tmp_path / "out.dpo.jsonl"
    assert build_dpo_mix(repairs, out) == 0
    assert out.read_text() == ""


def test_validate_messages_mix(tmp_path):
    f = tmp_path / "mix.jsonl"
    f.write_text(
        json.dumps({"messages": [{"role": "system", "content": "s"}]})
        + "\n"
        + json.dumps({"foo": "bar"})
        + "\n"
    )
    rep = validate_messages_mix(f)
    assert rep["total"] == 2
    assert rep["malformed"] == 1
