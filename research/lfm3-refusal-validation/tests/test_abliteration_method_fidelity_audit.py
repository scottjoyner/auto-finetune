import pytest
from src.abliteration_method_fidelity_audit import summarize


def fixture():
    prov={
        "method":"M1-style row-space weight projection",
        "release_authorized":False,
        "publication_approved":False,
        "edited_matrices":48,"direction_layer":10,"alpha":1.0,
        "source_sha256":"base","checkpoint_sha256":"edited",
        "direction_sha256":"direction",
    }
    private={
        "original_checkpoint_sha256":"base",
        "edited_checkpoint_sha256":"edited",
        "calibration_case_ids":["p1","p2"],
        "holdout_case_ids":["p3","p4","p5","p6"],
        "ratios_edited_to_original_by_layer":{str(i):0.5 for i in range(1,24)},
    }
    live={
        "source_sha256":"base","target_sha256":"edited",
        "exact_projection_verified":True,
        "source_tensor_count":286,
        "selected_residual_writers":48,
        "actual_changed_tensors":48,
        "verified_non_target_unchanged":238,
    }
    return prov,private,live


def test_structural_only_fidelity_output():
    data=summarize(*fixture())
    assert data["calibration_pairs"]==2
    assert data["disjoint_holdout_prompt_pairs"]==4
    assert data["measured_layers_with_reduced_projection"]==23
    assert data["refusal_effectiveness_validated"] is False
    assert data["human_safety_validated"] is False
    assert data["exact_projection_verified"] is True


def test_fail_closed_tampered_tensor_identity():
    a,b,c=fixture()
    c["actual_changed_tensors"]=47
    with pytest.raises(ValueError,match="tensor-by-tensor"):
        summarize(a,b,c)


def test_fail_closed_wrong_hash():
    a,b,c=fixture()
    b["edited_checkpoint_sha256"]="different"
    with pytest.raises(ValueError,match="different model"):
        summarize(a,b,c)


def test_fail_closed_non_disjoint_holdout():
    a,b,c=fixture()
    b["holdout_case_ids"][0]="p1"
    with pytest.raises(ValueError,match="disjoint"):
        summarize(a,b,c)


def test_fail_closed_missing_layer_or_invalid_ratio():
    a,b,c=fixture()
    b["ratios_edited_to_original_by_layer"].pop("23")
    with pytest.raises(ValueError,match="23"):
        summarize(a,b,c)
    a,b,c=fixture()
    b["ratios_edited_to_original_by_layer"]["1"]=float("nan")
    with pytest.raises(ValueError,match="invalid"):
        summarize(a,b,c)
