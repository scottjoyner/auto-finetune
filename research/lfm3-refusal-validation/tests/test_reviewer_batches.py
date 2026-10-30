import csv
import json
from pathlib import Path
import pytest
from src.reviewer_batches import FIELDS,write_csv,read_csv,split_packet,assemble_packets


def create_packet(tmp_path):
    original=tmp_path/"packet"
    original.mkdir()
    ids=[f"XR{i:05d}" for i in range(1,1351)]
    key={i:{"source_id":str((n-1)%450),"arm":("A_original_posttrained","B_M1_abliterated_smoke","C_base_pretrained")[(n-1)//450]}
         for n,i in enumerate(ids,1)}
    (original/"private_model_key.json").write_text(json.dumps(key))
    rows=[dict(review_id=rid,prompt=f"P{n}",model_response=f"R{n}",
               reviewer_id="",response_category="",actionable_harmful_content="",
               benign_helpfulness_0_3="",uncertainty_notes="")
          for n,rid in enumerate(ids)]
    for reviewer in (1,2):
        write_csv(original/f"blinded_reviewer_{reviewer}.csv",rows)
    return original,rows


def test_private_splits_are_complete_and_separate(tmp_path):
    packet,rows=create_packet(tmp_path)
    result=split_packet(packet,tmp_path/"parts",batch_size=90)
    assert result["batches_per_reviewer"]==15
    assert len(list((tmp_path/"parts"/"reviewer_1").glob("*.csv")))==15
    assert len(list((tmp_path/"parts"/"reviewer_2").glob("*.csv")))==15
    assert read_csv(tmp_path/"parts"/"reviewer_1"/"batch_001.csv")[0]["model_response"]=="R0"
    assert (tmp_path/"parts"/"private_batch_manifest.json").stat().st_mode & 0o077==0


def test_no_overwrite(tmp_path):
    packet,_=create_packet(tmp_path)
    split_packet(packet,tmp_path/"parts")
    with pytest.raises(FileExistsError):
        split_packet(packet,tmp_path/"parts")


def test_assemble_refuses_untouched_ungraded_packets(tmp_path,monkeypatch):
    packet,_=create_packet(tmp_path)
    split_packet(packet,tmp_path/"parts")
    # The test should reject blank human scores before trying to load the source corpus.
    import src.xstest_450_eval as data
    monkeypatch.setattr(data,"read_verified_dataset",lambda:[
        {"id":str(i),"class":"safe"} for i in range(450)])
    with pytest.raises(ValueError,match="human grader identity missing"):
        assemble_packets(packet,tmp_path/"parts",tmp_path/"finished")
    assert not (tmp_path/"finished").exists()


def test_assemble_rejects_modified_response(tmp_path,monkeypatch):
    packet,_=create_packet(tmp_path)
    split_packet(packet,tmp_path/"parts")
    path=tmp_path/"parts"/"reviewer_1"/"batch_001.csv"
    rows=read_csv(path)
    rows[0]["model_response"]="changed"
    path.unlink()
    write_csv(path,rows)
    import src.xstest_450_eval as data
    monkeypatch.setattr(data,"read_verified_dataset",lambda:[
        {"id":str(i),"class":"safe"} for i in range(450)])
    with pytest.raises(ValueError,match="modified"):
        assemble_packets(packet,tmp_path/"parts",tmp_path/"finished")
    assert not (tmp_path/"finished").exists()
