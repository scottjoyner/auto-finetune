"""Capture LFM3's eight frozen contrast-pair deltas into PRIVATE SSD JSON.

Operator-invoked CPU read-only inference (forward passes, not generations).
Never start when capacity preflight fails. No model weights altered or new
model checkpoint saved. No private text or vectors printed to stdout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from src.calibration_direction_stability import (
    ALL_IDS, CALIBRATION, HOLDOUT, FORMAT, LAYER, WIDTH,
    MODEL_SHA, PAIRS_SHA, DIRECTION_FILE_SHA, write_exclusive,
)
from src.research_job_guard import require_research_write_capacity
from src.refusal_probe import MODEL, encode, hidden_by_layer, load_local_model
from src.reversible_alpha_controls import DIRECTIONS, PAIRS
from src.eval_contracts import read_jsonl


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(4 << 20), b''):
            h.update(piece)
    return h.hexdigest()


def pinned_inputs(model=MODEL, pair_file=PAIRS, vector_file=DIRECTIONS):
    model = Path(model)
    pair_file = Path(pair_file)
    vector_file = Path(vector_file)
    for path, expected in ((model / 'model.safetensors', MODEL_SHA),
                           (pair_file, PAIRS_SHA),
                           (vector_file, DIRECTION_FILE_SHA)):
        if sha256(path) != expected:
            raise ValueError('pinned model, calibration source or direction SHA256 differs')
    rows = read_jsonl(pair_file)
    if (len(rows) != len(ALL_IDS) or
            [row['id'] for row in rows] != list(ALL_IDS) or
            any(not isinstance(row.get('benign'), str) or
                not isinstance(row.get('adversarial'), str) or
                not row['benign'] or not row['adversarial'] for row in rows)):
        raise ValueError('frozen eight-pair source contract differs')
    return rows


def snapshot_vectors(model, tokenizer, rows, direction_tensor):
    import torch
    if (tuple(direction_tensor.shape) != (WIDTH,) or
            not bool(torch.isfinite(direction_tensor).all())):
        raise ValueError('invalid source M1 direction')
    if direction_tensor.float().norm().item() < 1e-10:
        raise ValueError('empty source M1 direction')
    deltas = {}
    for row in rows:
        benign = hidden_by_layer(model, encode(tokenizer, row['benign']))[LAYER]
        adversarial = hidden_by_layer(model, encode(tokenizer, row['adversarial']))[LAYER]
        delta = (adversarial - benign).to(dtype=torch.float32).contiguous()
        if delta.numel() != WIDTH or not bool(torch.isfinite(delta).all()):
            raise ValueError('invalid captured last-token activation contrast')
        deltas[row['id']] = delta.tolist()
    return {
        'format': FORMAT,
        'source_model_sha256': MODEL_SHA,
        'pair_dataset_sha256': PAIRS_SHA,
        'direction_file_sha256': DIRECTION_FILE_SHA,
        'layer': LAYER,
        'hidden_size': WIDTH,
        'calibration_pair_ids': list(CALIBRATION),
        'holdout_pair_ids': list(HOLDOUT),
        'fitted_direction': direction_tensor.float().tolist(),
        'pair_deltas': deltas,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-out',type=Path,required=True)
    args=p.parse_args()
    out=args.private_out
    if out.exists():
        raise FileExistsError('private capture already exists')
    # Protect the same source /home/scott/git root gate used across the study.
    require_research_write_capacity(out)
    if not out.parent.is_dir() or out.parent.is_symlink() or out.parent.stat().st_mode & 0o077:
        raise PermissionError('private SSD capture directory must exist and be 0700')
    import torch
    from safetensors.torch import load_file
    torch.manual_seed(0)
    rows = pinned_inputs()
    directions = load_file(str(DIRECTIONS), device='cpu')
    key = f'layer_{LAYER}'
    if key not in directions:
        raise ValueError('pinned source refusal contrast layer missing')
    model, tokenizer = load_local_model(MODEL)
    data = snapshot_vectors(model, tokenizer, rows, directions[key])
    # Recheck source files and disk gate after inference, immediately before write.
    pinned_inputs()
    require_research_write_capacity(out)
    from src.calibration_direction_stability import audit
    audit(data)  # refuses a mismatch to the frozen fitted M1 direction.
    os.umask(0o077)
    write_exclusive(out, data)
    print(json.dumps({
        'private_capture_created': True,
        'calibration_pairs': 2, 'heldout_pairs': 6,
        'private_prompts_or_activations_exposed': False,
        'model_weights_changed': False,
        'model_release_approved': False,
    }))


if __name__ == '__main__':
    main()
