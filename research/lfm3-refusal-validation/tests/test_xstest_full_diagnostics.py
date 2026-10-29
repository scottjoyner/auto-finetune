from copy import deepcopy
import pytest
from src.xstest_full_diagnostics import diagnostics
from src.xstest_450_eval import ARM_ORDER


def mock():
    cases={}
    rows=[]
    for i in range(450):
        name=str(i)
        cases[name]={"id":name,"class":"safe" if i<250 else "unsafe"}
        for arm in ARM_ORDER:
            rows.append({"id":name,"arm":arm,"refusal_proxy":arm==ARM_ORDER[0]})
    return rows,cases


def test_paired_proxy_differences_do_not_implicitly_claim_safety():
    rows,cases=mock()
    v=diagnostics(rows,cases,iterations=200)
    assert v["human_safety_validated"] is False
    assert v["per_stratum"]["safe"]["arms"][ARM_ORDER[0]]["lexical_refusal_count"]==250
    flip=v["per_stratum"]["safe"]["paired_diagnostic_only"][f"{ARM_ORDER[1]}_minus_{ARM_ORDER[0]}"]
    assert flip["previously_yes_now_no"]==250
    assert flip["difference"]["mean_difference"]==-1


def test_incomplete_journal_rejected():
    rows,cases=mock()
    with pytest.raises(ValueError,match="exact 1350"):
        diagnostics(rows[:-1],cases,iterations=200)


def test_duplicate_output_rejected():
    rows,cases=mock()
    with pytest.raises(ValueError,match="duplicate"):
        diagnostics([rows[0]]+rows,cases,iterations=200)


def test_invalid_proxy_rejected():
    rows,cases=mock()
    rows[0]["refusal_proxy"]="yes"
    with pytest.raises(ValueError,match="boolean"):
        diagnostics(rows,cases,iterations=200)
