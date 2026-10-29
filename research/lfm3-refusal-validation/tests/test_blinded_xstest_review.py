import pytest
from src.blinded_xstest_review import make_blinded_rows, validate_completed


def cases():
    records = [dict(id="01",arm=a,response="Test only") for a in
               ("A_original_posttrained", "B_M1_abliterated_smoke",
                "C_base_pretrained")]
    benchmark = [dict(id="01",prompt="A benign evaluation question")]
    return records,benchmark


def test_model_identity_is_secret():
    records, benchmark=cases()
    blinded, mapping = make_blinded_rows(records, benchmark, 41)
    assert len(blinded)==3
    assert len(mapping)==3
    assert not any("arm" in item or "source_id" in item for item in blinded)
    assert set(x["arm"] for x in mapping.values()) == set(r["arm"] for r in records)


def test_deterministic_review_shuffle():
    records, benchmark=cases()
    a=make_blinded_rows(records,benchmark,17)
    b=make_blinded_rows(records,benchmark,17)
    assert a==b


def test_incomplete_triple_rejected():
    records, benchmark=cases()
    with pytest.raises(ValueError, match="triplet"):
        make_blinded_rows(records[:2],benchmark,17)


def test_unknown_case_rejected():
    records,benchmark=cases()
    with pytest.raises(ValueError, match="unknown"):
        validate_completed(records,{"other"})
