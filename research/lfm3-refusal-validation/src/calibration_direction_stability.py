"""Research-only LFM3 fitted-direction stability, private vector input.

The original two-pair M1 direction may be unstable; this is an offline,
fail-closed geometry check, not behavioral safety or efficacy evidence.
No model weights are edited and no private activation vectors are published.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import stat

MODEL_SHA = '135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330'
DIRECTION_FILE_SHA = '7f31109d5212b4bb1f0839544935a694b9d4f4475e9c3a36926eebe28bd9437f'
PAIRS_SHA = 'ebdff35111f03617632a557793122fda05e9a235444c3dd0d6396f925bbf3d81'
LAYER = 10
WIDTH = 2048
CALIBRATION = ('p01', 'p02')
HOLDOUT = tuple(f'p{i:02d}' for i in range(3, 9))
ALL_IDS = CALIBRATION + HOLDOUT
SHAM_SEEDS = (17, 23, 41, 59)
FORMAT = 'lfm3-layer10-eight-pair-contrast-deltas-v1'
REQUIRED_FIELDS = {'format', 'source_model_sha256', 'pair_dataset_sha256',
                   'direction_file_sha256', 'layer', 'hidden_size',
                   'calibration_pair_ids', 'holdout_pair_ids',
                   'fitted_direction', 'pair_deltas'}


def _check_vector(raw, *, allow_zero=False):
    if (not isinstance(raw, list) or len(raw) != WIDTH or any(
            type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e12
            for x in raw)):
        raise ValueError('malformed, nonfinite or oversized private direction vector')
    v = [float(x) for x in raw]
    norm = math.hypot(*v)
    if not math.isfinite(norm) or (not allow_zero and norm <= 1e-10):
        raise ValueError('zero or invalid private direction vector')
    return v


def _unit(x):
    n = math.hypot(*x)
    if not math.isfinite(n) or n <= 1e-10:
        raise ValueError('degenerate or cancelling calibration direction')
    return [v/n for v in x]


def _dot(a, b):
    return math.fsum(x*y for x, y in zip(a, b))


def _round(x):
    return round(max(-1., min(1., x)), 6)


def _check_manifest(input_data):
    if not isinstance(input_data, dict) or set(input_data) != REQUIRED_FIELDS:
        raise ValueError('private input must use the exact pinned schema')
    for k, value in [('format', FORMAT), ('source_model_sha256', MODEL_SHA),
                     ('pair_dataset_sha256', PAIRS_SHA),
                     ('direction_file_sha256', DIRECTION_FILE_SHA),
                     ('layer', LAYER), ('hidden_size', WIDTH),
                     ('calibration_pair_ids', list(CALIBRATION)),
                     ('holdout_pair_ids', list(HOLDOUT))]:
        if type(input_data[k]) is not type(value) or input_data[k] != value:
            raise ValueError('private source identity or calibration split mismatch')
    data = input_data['pair_deltas']
    if not isinstance(data, dict) or set(data) != set(ALL_IDS):
        raise ValueError('requires exactly eight disjoint source-pinned pair vectors')
    ref = _unit(_check_vector(input_data['fitted_direction']))
    vectors = {i: _check_vector(data[i]) for i in ALL_IDS}
    return ref, vectors


def _orthogonal_sham(reference, seed):
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(WIDTH)]
    parallel = _dot(raw, reference)
    return _unit([r - parallel * u for r, u in zip(raw, reference)])


def audit(private_data):
    """Aggregate-only deterministic audit, no vector, ID or private text output."""
    ref, vectors = _check_manifest(private_data)
    u1, u2 = (_unit(vectors[c]) for c in CALIBRATION)
    fit = _unit([vectors[CALIBRATION[0]][i] + vectors[CALIBRATION[1]][i]
                 for i in range(WIDTH)])
    # The original estimator averages unnormalized deltas, THEN normalizes.
    match = _dot(fit, ref)
    if match < 0.9995:
        raise ValueError('reconstructed calibration does not reproduce frozen fitted direction')
    held = [_unit(vectors[i]) for i in HOLDOUT]
    fitted_scores = [_dot(v, ref) for v in held]
    loo1_scores = [_dot(v, u1) for v in held]
    loo2_scores = [_dot(v, u2) for v in held]
    results = []
    for seed in SHAM_SEEDS:
        sham = _orthogonal_sham(ref, seed)
        points = [_dot(v, sham) for v in held]
        win = sum(abs(f) > abs(s) + 1e-12 for f, s in zip(fitted_scores, points))
        tie = sum(abs(abs(f) - abs(s)) <= 1e-12 for f, s in zip(fitted_scores, points))
        results.append({
            'seed': seed,
            'mean_absolute_holdout_cosine': _round(math.fsum(abs(x) for x in points)/len(points)),
            'fitted_higher_absolute_alignment_cases': win,
            'ties': tie,
            'sham_higher_absolute_alignment_cases': len(held)-win-tie,
        })
    calibration_cosine = _dot(u1, u2)
    report = {
        'evidence_class': 'private activation contrast geometry; not behavioral refusal or safety evidence',
        'method': 'last-token adversarial-minus-benign residual differences, layer 10, BF16 CPU',
        'source_model_sha256': MODEL_SHA,
        'source_pairs_sha256': PAIRS_SHA,
        'source_direction_file_sha256': DIRECTION_FILE_SHA,
        'layer': LAYER,
        'calibration_pairs': len(CALIBRATION),
        'heldout_pairs': len(HOLDOUT),
        'hidden_size': WIDTH,
        'frozen_fitted_direction_reconstruction_cosine': _round(match),
        'calibration_leave_one_out_cosine': _round(calibration_cosine),
        'heldout_fitted_alignment_mean': _round(math.fsum(fitted_scores)/len(held)),
        'heldout_fitted_alignment_min': _round(min(fitted_scores)),
        'heldout_fitted_alignment_max': _round(max(fitted_scores)),
        'heldout_fitted_absolute_alignment_mean': _round(math.fsum(abs(v) for v in fitted_scores)/len(held)),
        'heldout_positive_direction_matches': sum(v > 0 for v in fitted_scores),
        'heldout_loo_single_calibrator_absolute_alignment_mean': [
            _round(math.fsum(abs(v) for v in loo1_scores)/len(held)),
            _round(math.fsum(abs(v) for v in loo2_scores)/len(held)),
        ],
        'independent_sham_directions_orthogonal_to_fit': results,
        'source_activation_vectors_exported': False,
        'new_model_generations': 0,
        'model_weights_changed': False,
        'human_refusals_evaluated': False,
        'claim_of_direction_stability_verified': False,
        'model_release_approved': False,
        'limitations': [
            'Eight paired contrasts are only two calibration plus six held-out examples',
            'Direction is an activation contrast, not a demonstrated refusal causal vector',
            'Two leave-one-out single-calibrator directions cannot support statistical generalization',
            'Sham directions are orthogonal to fit and therefore not an exchangeable randomization null',
            'Sham comparisons are descriptive; no p-values or efficacy claims are computed',
            'Source-hash metadata alone is insufficient; the capture command must attest source bytes',
            'Results are ungraded activation geometry, not human-rated refusal, utility, or actionability',
        ],
    }
    if len(results) != 4 or not -1 <= calibration_cosine <= 1 + 1e-8:
        raise AssertionError('direction diagnostic calculation failed')
    return report


def _private_input(path):
    p = Path(path)
    if any(part.is_symlink() for part in (p, *p.parents)):
        raise ValueError('private source path may not be symlinked')
    for folder in (p.parent,):
        m = folder.stat().st_mode
        if not stat.S_ISDIR(m) or m & 0o077:
            raise PermissionError('private source directory must be owner-only')
    m = p.stat()
    if not stat.S_ISREG(m.st_mode) or m.st_nlink != 1 or m.st_mode & 0o077:
        raise PermissionError('private vector file must be owner-only, single-link regular file')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(p, flags)
    with os.fdopen(fd, 'r', encoding='utf-8') as stream:
        mm = os.fstat(stream.fileno())
        if not stat.S_ISREG(mm.st_mode) or mm.st_nlink != 1 or mm.st_mode & 0o077:
            raise PermissionError('private vector inode changed or is not owner-only')
        return json.load(stream)


def write_exclusive(path, obj):
    p = Path(path)
    if any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError('output path cannot contain symlinks')
    mode = p.parent.stat().st_mode
    if not stat.S_ISDIR(mode) or mode & 0o077:
        raise PermissionError('output directory must be owner-only')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(p, flags, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-vectors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = audit(_private_input(args.private_vectors))
    write_exclusive(args.output, report)
    print(json.dumps({'n_calibration_pairs': 2, 'n_heldout_pairs': 6,
                      'aggregate_report_created': True, 'weights_changed': False,
                      'behavioral_refusal_verified': False,
                      'model_release_approved': False}))


if __name__ == '__main__':
    main()
