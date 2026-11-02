"""Offline two-calibrator balance sensitivity from PRIVATE pinned LFM3 deltas.

This is a post-hoc direction geometry comparison, not a model edit, refusal
benchmark, or independent new prompt sample. Publishes aggregate numbers only.
"""
import argparse
import json
import math
from pathlib import Path

from src.calibration_direction_stability import (
    CALIBRATION, HOLDOUT, MODEL_SHA, PAIRS_SHA, DIRECTION_FILE_SHA,
    _check_manifest, _dot, _private_input, _round, _unit,
    audit as audit_stability, write_exclusive,
)

CANDIDATES = ('original_magnitude_weighted', 'equal_unit_weight',
              'calibrator_1_only', 'calibrator_2_only')


def evaluate(private_data):
    """Compute only public-safe aggregates; never return hidden vectors."""
    previous = audit_stability(private_data)  # full provenance+fitted hash check
    original, values = _check_manifest(private_data)
    raw1, raw2 = (values[key] for key in CALIBRATION)
    norm1 = math.hypot(*raw1)
    norm2 = math.hypot(*raw2)
    one, two = _unit(raw1), _unit(raw2)
    equal_weight = _unit([a+b for a,b in zip(one,two)])
    held = [_unit(values[k]) for k in HOLDOUT]

    directions = dict(zip(CANDIDATES, (original, equal_weight, one, two)))
    metrics = {}
    for name, vec in directions.items():
        scores = [_dot(v, vec) for v in held]
        metrics[name] = {
            'matched_heldout_pairs': len(held),
            'mean_signed_cosine': _round(math.fsum(scores)/len(held)),
            'min_signed_cosine': _round(min(scores)),
            'max_signed_cosine': _round(max(scores)),
            'mean_absolute_cosine': _round(
                math.fsum(abs(v) for v in scores)/len(held)),
            'positive_signed_alignment_count': sum(v > 0 for v in scores),
        }

    first = metrics['original_magnitude_weighted']
    balanced = metrics['equal_unit_weight']
    return {
        'evidence_class': 'post-hoc frozen LFM3 two-calibrator direction weight sensitivity; no causal behavioral inference',
        'source_model_sha256': MODEL_SHA,
        'source_pair_dataset_sha256': PAIRS_SHA,
        'source_direction_file_sha256': DIRECTION_FILE_SHA,
        'original_source_reconstruction_cosine': previous['frozen_fitted_direction_reconstruction_cosine'],
        'calibration_pairs': len(CALIBRATION),
        'heldout_pairs': len(HOLDOUT),
        'calibrator_norm_1': round(norm1,6),
        'calibrator_norm_2': round(norm2,6),
        'calibration_magnitude_ratio_larger_to_smaller': round(max(norm1,norm2)/min(norm1,norm2),6),
        'calibration_unit_cosine': previous['calibration_leave_one_out_cosine'],
        'equal_weight_vs_original_direction_cosine': _round(_dot(equal_weight,original)),
        'candidate_diagnostics': metrics,
        'balanced_minus_original_mean_signed_cosine': round(
            balanced['mean_signed_cosine']-first['mean_signed_cosine'],6),
        'balanced_minus_original_min_signed_cosine': round(
            balanced['min_signed_cosine']-first['min_signed_cosine'],6),
        'selection_was_post_hoc': True,
        'independent_new_prompts': 0,
        'new_model_forward_passes': 0,
        'new_model_generations': 0,
        'model_weights_changed': False,
        'candidate_checkpoint_created': False,
        'human_refusal_review_complete': False,
        'model_release_approved': False,
        'limitations': [
            'Only two calibration and six reused held-out contrasts; a post-hoc comparison, not generalization',
            'Equal weighting creates a possible direction candidate but it was not used for any model generation',
            'Better minimum held-out alignment alone may simply reflect topic or prompt differences',
            'The four directions are not randomized treatments and no significance testing is justified',
            'The private vector file and per-pair contrast directions must never be committed or logged',
            'No ability, unsafe actionability or appropriate refusal outcomes measured by these scores',
        ],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-vectors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=evaluate(_private_input(args.private_vectors))
    write_exclusive(args.output,result)
    print(json.dumps({
        'aggregate_created':True,
        'calibration_pairs':result['calibration_pairs'],
        'heldout_pairs':result['heldout_pairs'],
        'model_weights_changed':False,
        'release_authorized':False,
    }))


if __name__=='__main__':
    main()
