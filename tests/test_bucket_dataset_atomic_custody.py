"""Safe bucketed-dataset publication: only synthetic temporary JSONL records."""
from __future__ import annotations

import json
import sys

import pytest

from src import bucket_dataset as buckets


def example(sid, tools=None):
    msg = {"role": "assistant", "content": "synthetic", "tool_calls": [
        {"function": {"name": name}} for name in (tools or [])
    ]}
    return {"session_id": sid, "messages": [msg]}


def invoke(monkeypatch, source, target, *flags):
    monkeypatch.setattr(sys, "argv", [
        "bucket_dataset", "--input", str(source), "--out", str(target), *flags,
    ])
    return buckets.main()


def write_source(tmp_path, rows):
    source = tmp_path / "train.decontaminated.jsonl"
    source.write_text("".join(json.dumps(x) + "\n" for x in rows))
    return source


def test_no_tools_is_general_bucket():
    assert buckets.bucket_example(example("one")) == ["general"]


def test_specific_tool_bucket_without_overlap():
    names = buckets.bucket_example(example("one", ["read_file", "terminal"]))
    assert names == ["tool:read_file"]


def test_overlap_includes_group_and_specific_tools():
    labels = buckets.bucket_example(
        example("one", ["terminal", "read_file"]), overlap=True)
    assert set(labels) == {
        "code_execution", "filesystem", "tool:terminal", "tool:read_file",
    }


def test_unknown_tool_name_is_isolated_tool_bucket():
    assert buckets.bucket_example(
        example("one", ["custom_tool"])) == ["tool:custom_tool"]


def test_missing_input_refuses_creation(tmp_path, monkeypatch):
    target = tmp_path / "output"
    assert invoke(monkeypatch, tmp_path / "not-there", target) == 78
    assert not target.exists()


def test_empty_file_refuses_output(tmp_path, monkeypatch):
    source = tmp_path / "empty.jsonl"
    source.write_text("\n")
    target = tmp_path / "out"
    assert invoke(monkeypatch, source, target) == 78
    assert not target.exists()


@pytest.mark.parametrize("content", [
    "{bad json\n", '["not-object"]\n',
    '{"session_id":"x","messages":"not-list"}\n',
    '{"session_id":"x","messages":[]}\n',
    '{"session_id":"x","messages":["not-object"]}\n',
])
def test_invalid_examples_fail_before_output(tmp_path, monkeypatch, content):
    source = tmp_path / "bad.jsonl"
    source.write_text(content)
    target = tmp_path / "out"
    assert invoke(monkeypatch, source, target) == 78
    assert not target.exists()


@pytest.mark.parametrize("unsafe_name", [
    "../outside", "../../../outside", "danger/subdir", "too many spaces",
])
def test_untrusted_tool_bucket_names_cannot_write_outside_target(
        tmp_path, monkeypatch, unsafe_name):
    source = write_source(tmp_path, [example("one", [unsafe_name])])
    target = tmp_path / "out"
    assert invoke(monkeypatch, source, target) == 78
    assert not target.exists()
    assert not (tmp_path / "outside").exists()


def test_normal_bucket_output_has_manifest_with_final_paths(tmp_path, monkeypatch):
    source = write_source(tmp_path, [
        example("a", ["read_file"]), example("b", ["read_file"]),
        example("c", ["read_file"]),
    ])
    target = tmp_path / "final"
    assert invoke(monkeypatch, source, target, "--min-per-bucket", "2") == 0
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["total_examples"] == 3
    assert manifest["buckets"]["tool:read_file"]["count"] == 3
    actual = target / "tool:read_file.jsonl"
    assert manifest["buckets"]["tool:read_file"]["file"] == str(actual)
    assert len(actual.read_text().splitlines()) == 3


def test_small_buckets_merged_into_misc(tmp_path, monkeypatch):
    source = write_source(tmp_path, [
        example("a", ["read_file"]), example("b", ["terminal"]),
    ])
    target = tmp_path / "final"
    assert invoke(monkeypatch, source, target, "--min-per-bucket", "2") == 0
    m = json.loads((target / "manifest.json").read_text())
    assert m["buckets"]["misc"]["count"] == 2
    ids = [json.loads(x)["session_id"] for x in
           (target / "misc.jsonl").read_text().splitlines()]
    assert ids == ["a", "b"]


def test_overlap_writes_multiple_readable_bucket_views(tmp_path, monkeypatch):
    source = write_source(tmp_path, [example("one", ["read_file", "terminal"])])
    target = tmp_path / "final"
    assert invoke(monkeypatch, source, target,
                  "--overlap", "--min-per-bucket", "1") == 0
    m = json.loads((target / "manifest.json").read_text())
    assert m["total_examples"] == 1
    assert len(m["buckets"]) == 4
    assert all((target / f"{label}.jsonl").exists() for label in m["buckets"])


def test_existing_output_never_overwritten(tmp_path, monkeypatch):
    source = write_source(tmp_path, [example("a")])
    out = tmp_path / "existing"
    out.mkdir()
    (out / "original").write_text("KEEP")
    assert invoke(monkeypatch, source, out) == 78
    assert (out / "original").read_text() == "KEEP"


def test_preexisting_output_symlink_not_replaced(tmp_path, monkeypatch):
    source = write_source(tmp_path, [example("a")])
    existing = tmp_path / "real"
    existing.mkdir()
    target = tmp_path / "link"
    target.symlink_to(existing, target_is_directory=True)
    assert invoke(monkeypatch, source, target) == 78
    assert target.is_symlink()


def test_bad_minimum_refuses_prior_to_out_creation(tmp_path, monkeypatch):
    source = write_source(tmp_path, [example("a")])
    out = tmp_path / "out"
    assert invoke(monkeypatch, source, out, "--min-per-bucket", "0") == 78
    assert not out.exists()


def test_staging_manifest_failure_does_not_publish_partial_data(
        tmp_path, monkeypatch):
    source = write_source(tmp_path, [example("a")])
    out = tmp_path / "out"
    monkeypatch.setattr(buckets, "atomic_write_json",
                        lambda *a: (_ for _ in ()).throw(OSError("synthetic fail")))
    assert invoke(monkeypatch, source, out, "--min-per-bucket", "1") == 78
    assert not out.exists()
    assert not list(tmp_path.glob(".bucket-stage-*"))
