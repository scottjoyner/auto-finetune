"""Synthetic-only tests; no simulated human judgments."""
import copy

import pytest

from src.official_arc_readiness import ARMS_EXPECTED
from src.refusal_architecture_rank import summarize,wilson

LAYERS=(["conv"]*13+["full_attention"]*6+["sliding_attention"]*5)

def fixtures():
    hashes={arm:f"synthetichash{i}" for i,arm in enumerate(ARMS_EXPECTED)}
    counts={"safe":[39,31,34],"unsafe":[172,167,163]}
    proxy={"evidence_class":"descriptive, weak lexical keyword proxy",
           "human_safety_validated":False,
           "n_cases":450,"n_generations":1350,
           "private_journal_sha256":"private-receipt",
           "benchmark_sha256":"benchmark-receipt",
           "checkpoint_sha256":copy.deepcopy(hashes),
           "per_stratum":{}}
    for name,nums in counts.items():
        n=250 if name=="safe" else 200
        proxy["per_stratum"][name]={
            "number_of_prompts":n,
            "arms":{
                arm:{"n":n,"lexical_refusal_count":k,"lexical_refusal_rate":k/n}
                for arm,k in zip(ARMS_EXPECTED,nums)
            }
        }
    progress={"recorded_generations":1350,"journal_integrity":"passed",
              "manual_review":"NOT STARTED","ignored_unterminated_tail":False,
              "journal_complete_lines_sha256":"private-receipt",
              "benchmark_sha256":"benchmark-receipt",
              "checkpoint_sha256":copy.deepcopy(hashes)}
    configs={arm:{"layer_types":LAYERS[:],"hidden_size":2048,"num_attention_heads":16}
             for arm in ARMS_EXPECTED}
    return proxy,progress,configs,hashes


def test_highest_signal_is_a_but_not_human_verified():
    d=summarize(*fixtures())
    assert d["strata"]["safe"]["highest_keyword_positive_arm"]==[ARMS_EXPECTED[0]]
    assert d["strata"]["unsafe"]["highest_keyword_positive_arm"]==[ARMS_EXPECTED[0]]
    assert d["strata"]["unsafe"]["arms"][ARMS_EXPECTED[0]]["fraction"]==.86
    assert d["architecture_summary"]["layer_types"]=={
        "conv":13,"full_attention":6,"sliding_attention":5
    }
    assert d["human_rated_refusal_rate_available"] is False
    assert d["claim_any_specific_layer_causes_refusal"] is False
    assert d["model_release_approved"] is False


def test_wilson_interval_bounded():
    for k,n in ((0,200),(200,200),(39,250),(172,200)):
        a,b=wilson(k,n)
        assert 0<=a<=k/n<=b<=1


@pytest.mark.parametrize("tamper",[
    "changed_checkpoint","changed_journal","review_started","wrong_architecture",
    "different_arm_architecture","wrong_count","wrong_rate","missing_stratum"
])
def test_fail_closed_for_untrusted_sources(tamper):
    proxy,progress,configs,hashes=fixtures()
    if tamper=="changed_checkpoint":hashes[ARMS_EXPECTED[0]]="different"
    elif tamper=="changed_journal":progress["journal_complete_lines_sha256"]="changed"
    elif tamper=="review_started":progress["manual_review"]="COMPLETE"
    elif tamper=="wrong_architecture":configs[ARMS_EXPECTED[0]]["layer_types"]=[]
    elif tamper=="different_arm_architecture":configs[ARMS_EXPECTED[1]]["layer_types"][:]=["conv"]*24
    elif tamper=="wrong_count":proxy["per_stratum"]["safe"]["arms"][ARMS_EXPECTED[0]]["n"]=249
    elif tamper=="wrong_rate":proxy["per_stratum"]["unsafe"]["arms"][ARMS_EXPECTED[0]]["lexical_refusal_rate"]=.3
    elif tamper=="missing_stratum":proxy["per_stratum"].pop("safe")
    with pytest.raises(ValueError):
        summarize(proxy,progress,configs,hashes)
