"""Tests for src.format_dataset."""
from __future__ import annotations

import json

from conftest import make_cfg

from src.format_dataset import (
    _format_window,
    _render_message,
    _render_part,
    _window_messages,
    emit_strata,
    main,
    to_alpaca,
    to_chatml,
    to_sharegpt,
)


def _msg(role, parts):
    return {"role": role, "parts": parts}


def _tool(name, inp=None, out=None):
    p = {"type": "tool", "tool": name}
    if inp is not None:
        p["input"] = inp
    if out is not None:
        p["output"] = out
    return p


def _text(role, text):
    return {"role": role, "parts": [{"type": "text", "text": text}]}


def test_render_text():
    assert _render_part({"type": "text", "text": "hi"}) == "hi"


def test_render_reasoning():
    assert _render_part({"type": "reasoning", "text": "think"}) == "think"


def test_render_tool_with_output():
    p = {"type": "tool", "tool": "bash", "call_id": "c1",
         "input": {"cmd": "ls"}, "output": "file"}
    out = _render_part(p)
    assert "<tool_call" in out and "bash" in out
    assert "<tool_result>" in out and "file" in out


def test_render_tool_no_output():
    p = {"type": "tool", "tool": "x", "call_id": "c", "input": "i"}
    out = _render_part(p)
    assert "<tool_result>" not in out


def test_render_patch():
    out = _render_part({"type": "patch", "files": ["a.py", "b.py"]})
    assert "a.py" in out and "b.py" in out


def test_render_unknown_is_empty():
    assert _render_part({"type": "step-start"}) == ""


def test_render_message_joins():
    m = _msg("user", [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert _render_message(m) == "a\nb"


def test_to_chatml():
    msgs = [_msg("user", [{"type": "text", "text": "q"}]),
            _msg("assistant", [{"type": "text", "text": "a"}])]
    out = to_chatml(msgs, "SYS")
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1]["role"] == "user" and out[2]["role"] == "assistant"


def test_to_chatml_skips_empty():
    msgs = [_msg("assistant", [{"type": "step-start"}])]
    out = to_chatml(msgs, "")
    # empty system is preserved; only the content-less assistant is dropped
    assert out == [{"role": "system", "content": ""}]


def test_to_sharegpt():
    msgs = [_msg("user", [{"type": "text", "text": "q"}]),
            _msg("assistant", [{"type": "text", "text": "a"}])]
    out = to_sharegpt(msgs, "SYS")
    assert out[0]["from"] == "system"
    assert out[1]["from"] == "human" and out[2]["from"] == "gpt"


def test_to_alpaca():
    msgs = [_msg("user", [{"type": "text", "text": "q1"}]),
            _msg("assistant", [{"type": "text", "text": "a1"}]),
            _msg("user", [{"type": "text", "text": "q2"}]),
            _msg("assistant", [{"type": "text", "text": "a2"}])]
    out = to_alpaca(msgs, "SYS")
    assert out["instruction"].startswith("SYS")
    assert out["input"] == "q2"
    assert out["output"] == "a2"


def test_window_by_turns():
    msgs = [_msg("user", [{"type": "text", "text": str(i)}]) for i in range(10)]
    wins = _window_messages(msgs, max_turns=4, max_chars=0)
    # step = max_turns//2 = 2 -> windows start at 0,2,4,6,8 => 5 windows
    assert len(wins) == 5
    assert all(len(w) <= 4 for w in wins)


def test_window_by_chars():
    # each message ~100 chars; budget 250 -> ~2-3 per window
    msgs = [_msg("user", [{"type": "text", "text": "x" * 100}]) for _ in range(10)]
    wins = _window_messages(msgs, max_turns=0, max_chars=250)
    assert len(wins) > 1
    for w in wins:
        total = sum(len(_render_part(p)) for m in w for p in m["parts"])
        assert total <= 250


def test_window_whole_when_no_budget():
    msgs = [_msg("user", [{"type": "text", "text": "x"}]) for _ in range(5)]
    wins = _window_messages(msgs, max_turns=0, max_chars=0)
    assert len(wins) == 1 and len(wins[0]) == 5


def test_format_window_dispatch():
    msgs = [_msg("user", [{"type": "text", "text": "q"}])]
    assert "messages" in _format_window(msgs, "chatml", "s")
    assert "conversations" in _format_window(msgs, "sharegpt", "s")
    assert "instruction" in _format_window(msgs, "alpaca", "s")


def test_main_writes_jsonl(tmp_root, sample_session):
    import json
    cleaned = tmp_root / "data" / "cleaned"
    datasets = tmp_root / "data" / "datasets"
    (cleaned / "a.json").write_text(json.dumps(sample_session))
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(datasets)})
    n = main(cfg)
    assert n >= 1
    lines = (datasets / "train.jsonl").read_text().strip().splitlines()
    # main writes each output file exactly once; the merged file's line count
    # equals the count returned for that file (the aggregate `total` also
    # includes per-source subdir files, which is why we assert on train.jsonl).
    assert len(lines) == n
    obj = json.loads(lines[0])
    assert "messages" in obj


def _write_session(path, sid, bucket, messages):
    path.write_text(json.dumps(
        {"source": "opencode", "session_id": sid, "messages": messages}))
    return {sid: {"bucket": bucket}}


def test_emit_strata_writes_per_bucket(tmp_root):
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = {}
    bm.update(_write_session(
        cleaned / "s.json", "s1", "shell",
        [_text("user", "run the tests"),
         {"role": "assistant", "parts": [_tool("bash", {"command": "pytest"}, "ok")] * 3}]))
    bm.update(_write_session(
        cleaned / "e.json", "e1", "file-edit",
        [_text("user", "create an add module"),
         {"role": "assistant", "parts": [
             _tool("write", {"filePath": "/repo/add.py", "content": "def add(a,b):\n    return a+b"}, "ok")]},
         _text("assistant", "Done.")]))
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(tmp_root / "data" / "datasets")})
    counts = emit_strata(cfg, bm, str(out))
    assert counts["shell"] >= 1 and counts["file-edit"] >= 1
    assert (out / "train.shell.jsonl").exists()
    assert (out / "train.file-edit.jsonl").exists()
    lines = (out / "train.shell.jsonl").read_text().strip().splitlines()
    assert len(lines) == counts["shell"]
    import json as _json
    _json.loads(lines[0])


def test_emit_strata_balance_upsamples(tmp_root):
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = {}
    bm.update(_write_session(
        cleaned / "s.json", "s1", "shell",
        [_text("user", "run it"),
         {"role": "assistant", "parts": [_tool("bash", {"command": "ls"}, "ok")]},
         _text("assistant", "done")]))
    for i in range(2):
        bm.update(_write_session(
            cleaned / f"e{i}.json", f"e{i}", "file-edit",
            [_text("user", "create module"),
             {"role": "assistant", "parts": [
                 _tool("write", {"filePath": f"/repo/{i}.py", "content": "x=1"}, "ok")]},
             _text("assistant", "ok")]))
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(tmp_root / "data" / "datasets")})
    counts = emit_strata(cfg, bm, str(out), balance=True)
    # largest bucket is file-edit (2); shell upsampled to match
    assert counts["file-edit"] == 2
    assert counts["shell"] == 2
    assert counts["balanced"] == 4
    assert (out / "train.balanced.jsonl").exists()


def test_emit_strata_balance_downsamples(tmp_root):
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = {}
    # a long shell session -> many windows (max_turns=2), downsampled by balance
    shell_msgs = []
    for i in range(6):
        shell_msgs.append(_text("user", f"step {i}"))
        shell_msgs.append({"role": "assistant", "parts": [
            _tool("bash", {"command": f"echo {i}"}, "ok")]})
    bm.update(_write_session(cleaned / "s.json", "s1", "shell", shell_msgs))
    bm.update(_write_session(
        cleaned / "e.json", "e1", "file-edit",
        [_text("user", "create module"),
         {"role": "assistant", "parts": [
             _tool("write", {"filePath": "/repo/x.py", "content": "x=1"}, "ok")]},
         _text("assistant", "ok")]))
    cfg = make_cfg(
        paths={"raw_dir": str(tmp_root / "data" / "raw"),
               "cleaned_dir": str(cleaned),
               "dataset_dir": str(tmp_root / "data" / "datasets")},
        format={"max_turns_per_example": 2, "template": "chatml",
                "max_chars_per_example": 0})
    counts = emit_strata(cfg, bm, str(out), balance=True, cap=2)
    # shell had >2 windows but is capped down to 2; file-edit upsampled to 2
    assert counts["shell"] == 2
    assert counts["file-edit"] == 2
    assert counts["balanced"] == 4


def test_emit_strata_excludes_holdout(tmp_root):
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = {}
    bm.update(_write_session(
        cleaned / "s.json", "s1", "shell",
        [_text("user", "run the tests"),
         {"role": "assistant", "parts": [_tool("bash", {"command": "pytest"}, "ok")] * 3}]))
    bm.update(_write_session(
        cleaned / "e.json", "e1", "file-edit",
        [_text("user", "create an add module"),
         {"role": "assistant", "parts": [
             _tool("write", {"filePath": "/repo/add.py", "content": "def add(a,b):\n    return a+b"}, "ok")]},
         _text("assistant", "Done.")]))
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(tmp_root / "data" / "datasets")})
    counts = emit_strata(cfg, bm, str(out), exclude={"s1"})
    # s1 is held out: no shell examples, file-edit still emitted
    assert counts.get("excluded") == 1
    assert counts.get("shell", 0) == 0
    assert counts["file-edit"] >= 1
    assert not (out / "train.shell.jsonl").exists()
    assert (out / "train.file-edit.jsonl").exists()


# ── provenance sidecars + holdout verification ───────────────────────────────

def _write_stratum(out, bucket, sessions, rows_per_session=1):
    """Mimic emit_strata's output: dataset file plus provenance sidecar."""
    (out / f"train.{bucket}.jsonl").write_text(
        "".join(json.dumps({"messages": [{"role": "user", "content": f"{bucket} {i}"}]}) + "\n"
                for i in range(len(sessions) * rows_per_session)))
    (out / f"train.{bucket}.provenance.jsonl").write_text(
        "".join(json.dumps({"row": i, "session_id": sid}) + "\n"
                for i, sid in enumerate(sessions)))


def test_verify_holdout_clean(tmp_path):
    from src.format_dataset import verify_holdout
    _write_stratum(tmp_path, "debug", ["s1", "s2"])
    r = verify_holdout(str(tmp_path), {"held_out"})
    assert r["status"] == "clean"
    assert r["n_rows"] == 2 and r["n_sessions"] == 2


def test_verify_holdout_detects_leaked_session(tmp_path):
    """The point of the sidecar: catch a held-out session after the fact."""
    from src.format_dataset import verify_holdout
    _write_stratum(tmp_path, "debug", ["s1", "held_out"])
    r = verify_holdout(str(tmp_path), {"held_out"})
    assert r["status"] == "contaminated"
    assert r["leaked"][0]["session_id"] == "held_out"
    assert r["leaked"][0]["row"] == 1


def test_verify_holdout_flags_row_count_mismatch(tmp_path):
    """A sidecar that drifts from its dataset must not read as clean."""
    from src.format_dataset import verify_holdout
    _write_stratum(tmp_path, "debug", ["s1", "s2"])
    with (tmp_path / "train.debug.provenance.jsonl").open("a") as f:
        f.write(json.dumps({"row": 9, "session_id": "s3"}) + "\n")
    r = verify_holdout(str(tmp_path), set())
    assert r["status"] == "contaminated"
    assert any("reason" in h for h in r["leaked"])


def test_verify_holdout_unverifiable_without_sidecars(tmp_path):
    """Pre-existing strata have no sidecar; say so rather than claim clean."""
    from src.format_dataset import verify_holdout
    (tmp_path / "train.debug.jsonl").write_text('{"messages": []}\n')
    r = verify_holdout(str(tmp_path), {"x"})
    assert r["status"] == "unverifiable"


def test_verify_holdout_orphaned_provenance(tmp_path):
    from src.format_dataset import verify_holdout
    (tmp_path / "train.debug.provenance.jsonl").write_text(
        json.dumps({"row": 0, "session_id": "s1"}) + "\n")
    r = verify_holdout(str(tmp_path), set())
    assert r["status"] == "contaminated"


def test_emit_strata_writes_provenance_and_manifest(tmp_root):
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = _write_session(
        cleaned / "s.json", "s1", "shell",
        [_text("user", "run the tests"),
         {"role": "assistant", "parts": [_tool("bash", {"command": "pytest"}, "ok")] * 3}])
    bm.update(_write_session(
        cleaned / "b.json", "s_bench", "shell",
        [_text("user", "bench only"),
         {"role": "assistant", "parts": [_tool("bash", {"command": "true"}, "ok")]}]))
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(tmp_root / "data" / "datasets")})
    counts = emit_strata(cfg, bm, str(out), exclude={"s_bench"})

    m = json.loads((out / "strata-manifest.json").read_text())
    assert m["excluded_sessions"] == ["s_bench"]
    assert m["counts"] == counts
    assert counts["excluded"] == 1
    for b in counts:
        if b == "excluded":
            continue
        ds = out / f"train.{b}.jsonl"
        pf = out / f"train.{b}.provenance.jsonl"
        assert pf.is_file(), f"no provenance sidecar for {b}"
        assert sum(1 for _ in ds.open()) == sum(1 for _ in pf.open())

    from src.format_dataset import verify_holdout
    r = verify_holdout(str(out), {"s_bench"})
    assert r["status"] == "clean"
    assert "s_bench" not in {s for st in m["strata"].values() for s in st["sessions"]}


def test_emit_strata_provenance_stays_out_of_training_rows(tmp_root):
    """Provenance belongs in the sidecar; the trainer reads only `messages`."""
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = _write_session(
        cleaned / "s.json", "s1", "shell",
        [_text("user", "run the tests"),
         {"role": "assistant", "parts": [_tool("bash", {"command": "pytest"}, "ok")] * 3}])
    cfg = make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(tmp_root / "data" / "datasets")})
    emit_strata(cfg, bm, str(out), exclude=set())
    for ds in out.glob("train.*.jsonl"):
        if ds.name == "train.balanced.jsonl" or ds.name.endswith(".provenance.jsonl"):
            continue
        row = json.loads(ds.open().readline())
        assert sorted(row) == ["messages"], f"{ds.name} gained keys: {sorted(row)}"


def test_emit_strata_provenance_survives_balancing(tmp_root):
    """Upsample/downsample reorders and repeats rows; sidecars must stay aligned."""
    from src.format_dataset import verify_holdout
    cleaned = tmp_root / "data" / "cleaned"
    out = tmp_root / "data" / "analysis"
    bm = {}
    shell_msgs = []
    for i in range(6):
        shell_msgs.append(_text("user", f"step {i}"))
        shell_msgs.append({"role": "assistant", "parts": [
            _tool("bash", {"command": f"echo {i}"}, "ok")]})
    bm.update(_write_session(cleaned / "s.json", "s1", "shell", shell_msgs))
    bm.update(_write_session(
        cleaned / "e.json", "e1", "file-edit",
        [_text("user", "create module"),
         {"role": "assistant", "parts": [
             _tool("write", {"filePath": "/repo/x.py", "content": "x=1"}, "ok")]},
         _text("assistant", "ok")]))
    cfg = make_cfg(
        paths={"raw_dir": str(tmp_root / "data" / "raw"),
               "cleaned_dir": str(cleaned),
               "dataset_dir": str(tmp_root / "data" / "datasets")},
        format={"max_turns_per_example": 2, "template": "chatml",
                "max_chars_per_example": 0})
    counts = emit_strata(cfg, bm, str(out), balance=True, cap=2)
    for b in counts:
        ds = out / f"train.{b}.jsonl"
        pf = out / f"train.{b}.provenance.jsonl"
        if ds.name == "train.balanced.jsonl" or not pf.is_file():
            continue
        assert sum(1 for _ in ds.open()) == sum(1 for _ in pf.open()), b
    assert verify_holdout(str(out), {"nothing"})["status"] == "clean"


# ── format_main holdout (this path previously had none) ──────────────────────

def _cfg_for(tmp_root, cleaned):
    return make_cfg(paths={
        "raw_dir": str(tmp_root / "data" / "raw"),
        "cleaned_dir": str(cleaned),
        "dataset_dir": str(tmp_root / "data" / "datasets")})


def test_format_main_holds_out_benchmark_sessions(tmp_root):
    cleaned = tmp_root / "data" / "cleaned" / "hermes"
    cleaned.mkdir(parents=True)
    _write_session(cleaned / "a.json", "s_keep", "shell",
                   [_text("user", "keep me"),
                    {"role": "assistant", "parts": [_tool("bash", {"command": "true"}, "ok")]}])
    _write_session(cleaned / "b.json", "s_bench", "shell",
                   [_text("user", "benchmark only"),
                    {"role": "assistant", "parts": [_tool("bash", {"command": "true"}, "ok")]}])
    cfg = _cfg_for(tmp_root, tmp_root / "data" / "cleaned")
    n = main(cfg, label="hermes", exclude={"s_bench"})
    out = tmp_root / "data" / "datasets" / "train.hermes.jsonl"
    assert n == 1 and out.is_file()
    assert "benchmark only" not in out.read_text()

    from src.format_dataset import verify_holdout
    ds_dir = tmp_root / "data" / "datasets"
    assert verify_holdout(str(ds_dir), {"s_bench"})["status"] == "clean"


def test_format_main_writes_provenance_aligned_with_rows(tmp_root):
    cleaned = tmp_root / "data" / "cleaned" / "hermes"
    cleaned.mkdir(parents=True)
    # Two windows from one session: provenance must repeat, not collapse.
    msgs = []
    for i in range(4):
        msgs.append(_text("user", f"step {i}"))
        msgs.append({"role": "assistant", "parts": [
            _tool("bash", {"command": f"echo {i}"}, "ok")]})
    _write_session(cleaned / "a.json", "s1", "shell", msgs)
    cfg = _cfg_for(tmp_root, tmp_root / "data" / "cleaned")
    cfg.raw["format"]["max_turns_per_example"] = 2
    main(cfg, label="hermes", exclude=set())
    ds = tmp_root / "data" / "datasets" / "train.hermes.jsonl"
    pf = tmp_root / "data" / "datasets" / "train.hermes.provenance.jsonl"
    assert pf.is_file()
    ds_rows = [json.loads(l) for l in ds.open() if l.strip()]
    pf_rows = [json.loads(l) for l in pf.open() if l.strip()]
    assert len(ds_rows) == len(pf_rows) > 1, "windows must stay distinct rows"
    assert all(r["session_id"] == "s1" for r in pf_rows)
    assert [r["row"] for r in pf_rows] == list(range(len(pf_rows)))


def test_format_main_training_rows_stay_messages_only(tmp_root):
    cleaned = tmp_root / "data" / "cleaned" / "hermes"
    cleaned.mkdir(parents=True)
    _write_session(cleaned / "a.json", "s1", "shell",
                   [_text("user", "hi"),
                    {"role": "assistant", "parts": [_tool("bash", {"command": "x"}, "ok")]}])
    cfg = _cfg_for(tmp_root, tmp_root / "data" / "cleaned")
    main(cfg, label="hermes", exclude=set())
    ds = tmp_root / "data" / "datasets" / "train.hermes.jsonl"
    for line in ds.open():
        if line.strip():
            assert sorted(json.loads(line)) == ["messages"]


def test_format_main_falls_back_to_filename_for_session_id(tmp_path):
    """Records without session_id still get provenance from the filename."""
    cleaned = tmp_path / "data" / "cleaned" / "hermes"
    cleaned.mkdir(parents=True)
    (cleaned / "s_from_name.json").write_text(json.dumps({
        "source": "hermes", "messages": [
            {"role": "user", "parts": [_text("user", "hello")]},
            {"role": "assistant", "parts": [_tool("bash", {"command": "x"}, "ok")]}]}))
    cfg = make_cfg(paths={"raw_dir": str(tmp_path / "data" / "raw"),
                          "cleaned_dir": str(tmp_path / "data" / "cleaned"),
                          "dataset_dir": str(tmp_path / "data" / "datasets")})
    assert main(cfg, label="hermes", exclude={"s_from_name"}) == 0
    assert main(cfg, label="hermes", exclude=set()) == 1


def test_format_main_merged_output_is_verifiable(tmp_root):
    """The merged train.jsonl must be covered too, not just per-label files."""
    cleaned = tmp_root / "data" / "cleaned"
    (cleaned / "hermes").mkdir(parents=True)
    (cleaned / "opencode").mkdir(parents=True)
    _write_session(cleaned / "hermes" / "a.json", "h1", "shell",
                   [_text("user", "h"), {"role": "assistant", "parts": [
                       _tool("bash", {"command": "x"}, "ok")]}])
    _write_session(cleaned / "opencode" / "b.json", "o1", "shell",
                   [_text("user", "o"), {"role": "assistant", "parts": [
                       _tool("bash", {"command": "y"}, "ok")]}])
    cfg = _cfg_for(tmp_root, cleaned)
    main(cfg, exclude={"h1"})
    ds_dir = tmp_root / "data" / "datasets"
    assert (ds_dir / "train.provenance.jsonl").is_file(), "merged output needs provenance"
    from src.format_dataset import verify_holdout
    r = verify_holdout(str(ds_dir), {"h1"})
    assert r["status"] == "clean", r["leaked"]
    assert "h1" not in (ds_dir / "train.jsonl").read_text()
