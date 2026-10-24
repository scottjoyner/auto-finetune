"""Disposable checkpoint metadata only. Never load model or optimizer pickles."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile
import pytest

from scripts import qwen_static_restore_inspect as s


def make_archive(path: Path, prefix: str, *, members=None):
    with zipfile.ZipFile(path,"w",zipfile.ZIP_STORED) as z:
        for key,val in (members or {
            f"{prefix}/data.pkl":b"not a pickle: never deserialize",
            f"{prefix}/version":b"3",
            f"{prefix}/byteorder":b"little",
        }).items():
            z.writestr(key,val)


def fixture(tmp_path):
    root=tmp_path/"qwen3-8b-sft-r1"
    for step in (100,150):
        p=root/f"checkpoint-{step}"
        p.mkdir(parents=True)
        (p/"trainer_state.json").write_text(json.dumps({
            "global_step":step,"max_steps":2522,"num_train_epochs":2,
            "epoch":step/1261,
        }))
        (p/"adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path":s.EXPECTED_BASE,
            "peft_type":"LORA","task_type":"CAUSAL_LM","r":16,
            "lora_alpha":32,"lora_dropout":0.05,
            "target_modules":["q_proj","v_proj"],
        }))
        (p/"tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class":"Qwen2Tokenizer",
            "model_max_length":131072,
        }))
        for name in s.MANDATORY:
            f=p/name
            if name in s.ZIP_ARTIFACTS:
                make_archive(f,s.ZIP_ARTIFACTS[name])
            elif not f.exists():
                f.write_bytes(b"synthetic fixture only")
    return root


def test_valid_structure_still_no_verified_restore(tmp_path):
    root=fixture(tmp_path)
    d=s.inspect(root)
    assert d["metadata_structure_consistent"]
    assert d["issues"]==[]
    assert all(x["torch_zip_structure"]
               for row in d["checkpoints"] for x in row["torch_archives"].values())
    assert not d["untrusted_torch_pickle_deserialized"]
    assert d["status"]=="NO_VERIFIED_RESTORE"
    assert all(d[k] is False for k in (
        "optimizer_restored","scheduler_restored","rng_restored",
        "trainer_restored","may_resume","may_train","may_prune","may_release_source",
    ))


def test_bad_global_step_detected(tmp_path):
    root=fixture(tmp_path)
    p=root/"checkpoint-150/trainer_state.json"
    v=json.loads(p.read_text());v["global_step"]=99;p.write_text(json.dumps(v))
    d=s.inspect(root)
    assert "checkpoint_global_step_mismatch" in d["issues"]


def test_invalid_max_steps_rejected(tmp_path):
    root=fixture(tmp_path)
    p=root/"checkpoint-100/trainer_state.json"
    v=json.loads(p.read_text());v["max_steps"]=25;p.write_text(json.dumps(v))
    assert "trainer_max_steps_invalid" in s.inspect(root)["issues"]


def test_conflicting_max_steps_disqualified(tmp_path):
    root=fixture(tmp_path)
    p=root/"checkpoint-150/trainer_state.json"
    v=json.loads(p.read_text());v["max_steps"]=2600;p.write_text(json.dumps(v))
    assert "trainer_schedule_differs_across_checkpoints" in s.inspect(root)["issues"]


def test_adapter_base_identity_drift_detected(tmp_path):
    root=fixture(tmp_path)
    p=root/"checkpoint-150/adapter_config.json"
    v=json.loads(p.read_text());v["base_model_name_or_path"]="/unverified/base"
    p.write_text(json.dumps(v))
    d=s.inspect(root)
    assert "adapter_base_model_mismatch" in d["issues"]
    assert "adapter_identity_differs_across_checkpoints" in d["issues"]


def test_lora_rank_mismatch_detected(tmp_path):
    root=fixture(tmp_path)
    p=root/"checkpoint-150/adapter_config.json"
    v=json.loads(p.read_text());v["r"]=32;p.write_text(json.dumps(v))
    assert "adapter_identity_differs_across_checkpoints" in s.inspect(root)["issues"]


def test_tokenizer_class_mismatch_detected(tmp_path):
    root=fixture(tmp_path)
    p=root/"checkpoint-100/tokenizer_config.json"
    v=json.loads(p.read_text());v["tokenizer_class"]="DifferentTokenizer"
    p.write_text(json.dumps(v))
    assert "tokenizer_identity_differs_across_checkpoints" in s.inspect(root)["issues"]


def test_missing_optimizer_zip_rejected(tmp_path):
    root=fixture(tmp_path)
    (root/"checkpoint-150/optimizer.pt").unlink()
    d=s.inspect(root)
    assert "state_artifact_missing_or_untrusted" in d["issues"]


def test_corrupt_optimizer_archive_rejected(tmp_path):
    root=fixture(tmp_path)
    (root/"checkpoint-100/optimizer.pt").write_bytes(b"not zip archive")
    assert "zip_archive_unreadable" in s.inspect(root)["issues"]


def test_zip_traversal_rejected(tmp_path):
    root=fixture(tmp_path)
    path=root/"checkpoint-100/scheduler.pt"
    make_archive(path,"scheduler",members={
        "scheduler/data.pkl":b"x","scheduler/version":b"3",
        "scheduler/../../evil":b"no",
    })
    assert "untrusted_archive_member" in s.inspect(root)["issues"]


def test_duplicate_zip_entries_rejected(tmp_path):
    root=fixture(tmp_path)
    path=root/"checkpoint-100/rng_state.pth"
    with zipfile.ZipFile(path,"w") as z:
        z.writestr("rng_state/data.pkl","x")
        z.writestr("rng_state/version","3")
        z.writestr("rng_state/data.pkl","duplicate")
    assert "duplicate_archive_member" in s.inspect(root)["issues"]


def test_missing_zip_version_and_data_pickle_rejected(tmp_path):
    root=fixture(tmp_path)
    path=root/"checkpoint-100/training_args.bin"
    make_archive(path,"training_args",members={"training_args/byteorder":b"little"})
    assert "torch_zip_metadata_incomplete" in s.inspect(root)["issues"]


def test_zip_uncompressed_limit_enforced(tmp_path,monkeypatch):
    root=fixture(tmp_path)
    monkeypatch.setattr(s,"MAX_ZIP_UNCOMPRESSED",1)
    assert "archive_uncompressed_limit_exceeded" in s.inspect(root)["issues"]


def test_zip_entry_limit_enforced(tmp_path,monkeypatch):
    root=fixture(tmp_path)
    monkeypatch.setattr(s,"MAX_ZIP_ENTRIES",1)
    assert "archive_member_count_unbounded" in s.inspect(root)["issues"]


def test_symlinked_artifact_refused(tmp_path):
    root=fixture(tmp_path)
    candidate=root/"checkpoint-100/optimizer.pt"
    external=tmp_path/"not-approved.pt"
    candidate.rename(external)
    candidate.symlink_to(external)
    assert "state_artifact_missing_or_untrusted" in s.inspect(root)["issues"]


def test_symlinked_checkpoint_refused(tmp_path):
    root=fixture(tmp_path)
    old=root/"checkpoint-150"
    moved=tmp_path/"moved"
    old.rename(moved)
    old.symlink_to(moved,target_is_directory=True)
    assert "checkpoint_directory_missing_or_linked" in s.inspect(root)["issues"]


def test_invalid_adapter_target_modules_refused(tmp_path):
    root=fixture(tmp_path)
    p=root/"checkpoint-150/adapter_config.json"
    v=json.loads(p.read_text());v["target_modules"]=["../../unsafe"]
    p.write_text(json.dumps(v))
    assert "adapter_target_modules_invalid" in s.inspect(root)["issues"]


def test_no_fixture_data_written_by_inspector(tmp_path):
    root=fixture(tmp_path)
    before={str(p):p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    s.inspect(root)
    after={str(p):p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}
    assert before==after


def test_cli_always_exits_78_even_if_structure_consistent(tmp_path,monkeypatch,capsys):
    root=fixture(tmp_path)
    monkeypatch.setattr("sys.argv",["qwen_static_restore_inspect.py","--run",str(root)])
    assert s.main()==78
    d=json.loads(capsys.readouterr().out)
    assert d["metadata_structure_consistent"]
    assert d["trainer_restored"] is False
