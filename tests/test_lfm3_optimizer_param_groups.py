"""Conservative offline Muon/AdamW trainable-parameter classification."""

from src.lfm3_optimizer_param_groups import plan_parameter_groups


def row(name, shape, trainable=True):
    return {"name": name, "shape": shape, "trainable": trainable}


def sample():
    return [
        row("model.layers.2.self_attn.q_proj.weight", [2048, 2048]),
        row("model.layers.0.conv.in_proj.weight", [6144, 2048]),
        row("model.layers.2.self_attn.gate_proj.weight", [16, 2048]),
        row("model.layers.0.conv.conv.weight", [2048, 1, 7]),
        row("model.embed_tokens.weight", [128000, 2048]),
        row("base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight", [16, 2048]),
        row("base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight", [2048, 16]),
        row("model.layers.2.self_attn.k_proj.weight", [256, 2048], trainable=False),
    ]


def test_safe_default_is_adamw_only():
    result = plan_parameter_groups(sample())
    assert result["status"] == "observed"
    assert result["proposal"]["muon_matrix_names"] == []
    assert len(result["proposal"]["adamw_names"]) == 7
    assert result["proposal"]["frozen_parameter_count"] == 1
    assert result["weight_promotion_permitted"] is False


def test_explicit_research_mode_only_selects_hidden_matrices():
    result = plan_parameter_groups(
        sample(), mode="full_finetune_research", allow_muon_proposal=True
    )
    assert result["status"] == "review_required"
    assert result["proposal"]["muon_matrix_names"] == [
        "model.layers.0.conv.in_proj.weight",
        "model.layers.2.self_attn.q_proj.weight",
    ]
    assert result["proposal"]["lora_trainable_count"] == 2
    assert len(result["proposal"]["adamw_names"]) == 5
    assert result["weight_promotion_permitted"] is False


def test_research_mode_still_requires_explicit_opt_in():
    assert (
        plan_parameter_groups(sample(), mode="full_finetune_research")["proposal"][
            "muon_matrix_names"
        ]
        == []
    )


def test_frozen_hidden_matrix_never_admitted_to_muon():
    p = [row("model.layers.4.feed_forward.w1.weight", [7680, 2048], False)]
    result = plan_parameter_groups(p, mode="full_finetune_research", allow_muon_proposal=True)
    assert result["status"] == "observed"
    assert result["proposal"]["muon_matrix_names"] == []


def test_outside_known_layer_range_stays_adamw():
    p = [row("model.layers.24.feed_forward.w1.weight", [7680, 2048])]
    assert not plan_parameter_groups(p, mode="full_finetune_research", allow_muon_proposal=True)[
        "proposal"
    ]["muon_matrix_names"]


def test_nonmatrix_attention_param_stays_adamw():
    p = [row("model.layers.2.self_attn.q_proj.weight", [2048])]
    assert not plan_parameter_groups(p, mode="full_finetune_research", allow_muon_proposal=True)[
        "proposal"
    ]["muon_matrix_names"]


def test_duplicate_names_fail_closed():
    p = [row("x.weight", [2, 2]), row("x.weight", [2, 2])]
    assert plan_parameter_groups(p)["reason"] == "invalid_or_duplicate_parameter"


def test_unexpected_fields_and_raw_values_fail_closed():
    p = [row("x.weight", [2, 2])]
    p[0]["raw_tensor"] = "not allowed"
    assert plan_parameter_groups(p)["reason"] == "invalid_parameter_record"


def test_boolean_dimension_fail_closed():
    assert (
        plan_parameter_groups([row("x.weight", [True, 2])])["reason"]
        == "invalid_or_duplicate_parameter"
    )


def test_bad_mode_or_optin_fail_closed():
    assert plan_parameter_groups(sample(), mode="automatic")["status"] == "blocked"
    assert plan_parameter_groups(sample(), allow_muon_proposal=1)["status"] == "blocked"


def test_empty_and_excessive_inputs_fail_closed():
    assert plan_parameter_groups([])["reason"] == "invalid_parameter_count"
    assert plan_parameter_groups(sample() * 2501)["reason"] == "invalid_parameter_count"


def test_lo_ra_is_never_hidden_muon_by_shape():
    p = [row("model.layers.2.self_attn.q_proj.lora_A.default.weight", [8, 2048])]
    result = plan_parameter_groups(p, mode="full_finetune_research", allow_muon_proposal=True)
    assert result["proposal"]["muon_matrix_names"] == []
    assert result["proposal"]["lora_trainable_count"] == 1