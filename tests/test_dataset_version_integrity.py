"""Dataset versioning, hash validation, and restoration stay in tmp_path."""
import json
from pathlib import Path

from src import dataset_version as version
from src.config import Config


def folders(tmp_path):
    dataset=tmp_path/"dataset"
    dataset.mkdir()
    (dataset/"train.jsonl").write_text('{"messages":[]}\n')
    (dataset/"metadata.json").write_text("{}")
    return dataset


def cfg(tmp_path):
    return Config(raw={"paths":{"analysis_dir":str(tmp_path/"analysis"),
                                "dataset_dir":str(tmp_path/"dataset")}})


def test_file_hashes_and_versions_record_real_bytes(tmp_path):
    dataset=folders(tmp_path)
    v=version.DatasetVersioner(str(tmp_path/"versions"))
    first=v.create_version(str(dataset),"toy",description="synthetic",tags=["test"])
    assert first.num_examples==1
    assert first.num_files==2
    assert first.total_bytes>0
    assert len(first.content_hash)==64
    assert v.get_version(first.version_id).content_hash==first.content_hash
    assert v.get_latest("toy").version_id==first.version_id
    assert v.get_latest("not-found") is None
    assert [x.version_id for x in v.list_versions("toy")]==[first.version_id]
    assert version.DatasetVersioner(str(tmp_path/"versions")).get_version(first.version_id)


def test_disposable_versions_compare_changed_content_and_lineage(tmp_path):
    dataset=folders(tmp_path)
    v=version.DatasetVersioner(str(tmp_path/"versions"))
    a=v.create_version(str(dataset),"toy")
    (dataset/"train.jsonl").write_text('{"messages":[]}\n{"messages":[1]}\n')
    b=v.create_version(str(dataset),"toy",parent_versions=[a.version_id])
    assert a.version_id!=b.version_id
    assert b.parent_versions==[a.version_id]
    diff=v.diff(a.version_id,b.version_id)
    assert diff["same_content"] is False
    assert diff["num_examples"]["delta"]==1
    assert v.diff("missing",b.version_id)["error"]=="version not found"


def test_restore_rejects_changed_or_missing_sources(tmp_path):
    dataset=folders(tmp_path)
    v=version.DatasetVersioner(str(tmp_path/"versions"))
    a=v.create_version(str(dataset),"toy")
    (dataset/"train.jsonl").write_text("modified\n")
    assert v.restore(a.version_id,str(tmp_path/"other")) is False
    assert not (tmp_path/"other").exists()
    b=v.create_version(str(dataset),"toy")
    (dataset/"train.jsonl").unlink()
    assert v.restore(b.version_id,str(tmp_path/"other")) is False
    assert v.restore("not-found",str(tmp_path/"other")) is False


def test_restore_rejects_recursion_and_copies_unchanged_source(tmp_path):
    dataset=folders(tmp_path)
    v=version.DatasetVersioner(str(tmp_path/"versions"))
    a=v.create_version(str(dataset),"toy")
    assert v.restore(a.version_id,str(dataset/"restored")) is False
    assert v.restore(a.version_id,str(dataset)) is False
    output=tmp_path/"new-home"
    assert v.restore(a.version_id,str(output)) is True
    assert (output/"train.jsonl").read_bytes()==(dataset/"train.jsonl").read_bytes()
    assert (output/"metadata.json").read_bytes()==(dataset/"metadata.json").read_bytes()


def test_cli_create_list_diff_and_restore_to_separate_folder(tmp_path,capsys):
    dataset=folders(tmp_path)
    c=cfg(tmp_path)
    assert version.main(c,["cli","dataset-version-create","--label=toy",
                           "--description=synthetic"])==0
    vs=version.DatasetVersioner(str(tmp_path/"analysis"/"dataset-versions"))
    old=vs.get_latest()
    assert old is not None
    assert version.main(c,["cli","dataset-version-list"])==0
    assert "synthetic" in capsys.readouterr().out
    assert version.main(c,["cli","dataset-version-diff"])==2
    assert version.main(c,["cli","dataset-version-diff","--v1=bad","--v2=bad"])==1
    assert version.main(c,["cli","dataset-version-diff",
                            f"--v1={old.version_id}",f"--v2={old.version_id}"])==0
    assert version.main(c,["cli","dataset-version-restore"])==2
    assert version.main(c,["cli","dataset-version-restore",
                            f"--v2={old.version_id}"])==0
    restored=tmp_path/"analysis"/"restored-datasets"/f"restored-{old.version_id}"
    assert (restored/"train.jsonl").exists()
    assert not (dataset/f"restored-{old.version_id}").exists()
