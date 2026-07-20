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
