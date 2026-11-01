"""Independent-dataset LFM3 direction transfer geometry, never refusal grading.

Fixed stratified XSTest 3/category (54 total) CPU forward-pass-only probe.
Uses the original A checkpoint and source-pinned *private* M1 calibration deltas.
No completions, weights, per-prompt activations or text leave the SSD.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

from src.calibration_direction_stability import (
    WIDTH, LAYER, MODEL_SHA, DIRECTION_FILE_SHA,
    _check_manifest, _dot, _private_input, _unit, write_exclusive,
)
from src.refusal_probe import MODEL, encode, hidden_by_layer, load_local_model
from src.research_job_guard import require_research_write_capacity
from src.reversible_alpha_controls import DIRECTIONS
from src.xstest_450_eval import (
    SOURCE_SHA256, SOURCE_FILE, read_verified_dataset, stratified_selection,
)
from src.capture_calibration_deltas import sha256

SEED = 9417
PER_CATEGORY = 3
COUNT = 54
SAFE_COUNT = 30
UNSAFE_COUNT = 24
SHAM_SEEDS = tuple(range(31000, 31128))
EXPECTED_CLASSES = {'safe': SAFE_COUNT, 'unsafe': UNSAFE_COUNT}


def _checked_candidates(private):
    ref, values = _check_manifest(private)
    a, b = _unit(values['p01']), _unit(values['p02'])
    balanced = _unit([x+y for x, y in zip(a,b)])
    fit = _unit([x+y for x, y in zip(values['p01'],values['p02'])])
    if _dot(ref, fit) < .9995:
        raise ValueError('frozen M1 direction disagrees with raw calibration')
    return {
        'original_M1_fitted': ref,
        'equal_unit_weight': balanced,
        'calibrator_1_only': a,
        'calibrator_2_only': b,
    }


def _validate_selection(rows, *, full):
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError('empty or malformed independent activation sample')
    by_class = {'safe': [], 'unsafe': []}
    cats = {}
    for r in rows:
        if not isinstance(r, dict) or r.get('class') not in by_class:
            raise ValueError('invalid XSTest class')
        cat = r.get('type')
        if not isinstance(cat, str) or not cat:
            raise ValueError('missing XSTest category')
        if 'activation' not in r:
            raise ValueError('missing last-token activation')
        raw = r['activation']
        if (not isinstance(raw, list) or len(raw) != WIDTH or
                any(type(v) not in (float, int) or not math.isfinite(v)
                    for v in raw)):
            raise ValueError('non-finite or dimensionally invalid activation')
        unit = _unit(raw)
        by_class[r['class']].append(unit)
        cats.setdefault(cat, []).append(r['class'])
    if any(not v for v in by_class.values()):
        raise ValueError('need both classes')
    if full:
        if (len(rows) != COUNT or len(cats) != 18 or
                {k:len(v) for k,v in by_class.items()} != EXPECTED_CLASSES or
                any(len(v) != PER_CATEGORY or len(set(v)) != 1
                    for v in cats.values())):
            raise ValueError('stratified 54-source XSTest selection incomplete')
    return by_class


def _metrics(by_class, direction):
    per_class = {
        cls: math.fsum(_dot(vec,direction) for vec in by_class[cls])/len(by_class[cls])
        for cls in ('safe','unsafe')
    }
    return {
        'mean_safe_cosine': round(per_class['safe'],6),
        'mean_unsafe_cosine': round(per_class['unsafe'],6),
        'unsafe_minus_safe_cosine': round(per_class['unsafe']-per_class['safe'],6),
    }


def aggregate(rows, private_calibration, *, full=True):
    by_class = _validate_selection(rows,full=full)
    candidates = _checked_candidates(private_calibration)
    metrics = {name:_metrics(by_class,vec) for name,vec in candidates.items()}
    # Unlike previously orthogonalized controls, these isotropic vectors
    # are independent of the fitted direction; still not a causal null.
    shams=[]
    for seed in SHAM_SEEDS:
        rng=random.Random(seed)
        s=_unit([rng.gauss(0,1) for _ in range(WIDTH)])
        shams.append(_metrics(by_class,s)['unsafe_minus_safe_cosine'])
    ordered=sorted(abs(v) for v in shams)
    target=abs(metrics['original_M1_fitted']['unsafe_minus_safe_cosine'])
    out={
        'evidence_class':'independent-corpus exploratory activation-class separation; not refusal',
        'dataset':'pinned XSTest 450, separate from original eight calibration pairs',
        'dataset_sha256':SOURCE_SHA256,
        'source_model_sha256':MODEL_SHA,
        'source_direction_sha256':DIRECTION_FILE_SHA,
        'sampling':'stratified, three per each of 18 source categories, seed 9417',
        'prompt_count':len(rows),
        'category_count':18 if full else None,
        'safe_prompts':len(by_class['safe']),
        'unsafe_prompts':len(by_class['unsafe']),
        'layer':LAYER,'activation_width':WIDTH,
        'candidate_class_cosine':metrics,
        'isotropic_sham_seed_first':SHAM_SEEDS[0],
        'isotropic_sham_count':len(shams),
        'isotropic_sham_abs_class_gap_median':round(ordered[len(ordered)//2],6),
        'isotropic_sham_abs_class_gap_max':round(ordered[-1],6),
        'isotropic_shams_with_abs_gap_at_least_fitted':sum(v>=target for v in ordered),
        'isotropic_shams_are_null_distribution':False,
        'candidate_selection_posthoc':True,
        'new_model_forward_passes':len(rows),
        'model_generations':0,
        'model_weights_changed':False,
        'human_refusal_grades':0,
        'model_release_approved':False,
        'raw_prompts_or_activations_exported':False,
        'limitations':[
            'Class separation is not refusal prediction or a safety/utility outcome',
            'XSTest class groups differ in topic, content and prompt length; not matched pairs',
            'Source was used previously for A/B/C responses; this is new activation geometry, not virgin benchmark data',
            'Only three examples per category, no causal effect or generalization claim',
            'Isotropic sham comparison is descriptive and does not constitute a valid model-behavior null',
            'Balanced direction was selected post hoc on the original six held-out contrasts',
            'Source selection and source weights verified by SHA-256 before capture',
        ],
    }
    return out


def checked_sources():
    if sha256(MODEL/'model.safetensors')!=MODEL_SHA:
        raise ValueError('original A model checkpoint SHA mismatch')
    if sha256(DIRECTIONS)!=DIRECTION_FILE_SHA:
        raise ValueError('frozen direction-file SHA mismatch')
    if sha256(SOURCE_FILE)!=SOURCE_SHA256:
        raise ValueError('independent XSTest source SHA mismatch')
    rows=stratified_selection(read_verified_dataset(),PER_CATEGORY,SEED)
    if (len(rows)!=COUNT or
            sum(r['class']=='safe' for r in rows)!=SAFE_COUNT or
            sum(r['class']=='unsafe' for r in rows)!=UNSAFE_COUNT or
            len({r['type'] for r in rows})!=18):
        raise ValueError('independent source selection deviated')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('existing aggregate cannot be overwritten')
    # Require root AND SSD headroom before new heavyweight model load.
    require_research_write_capacity(a.output)
    private=_private_input(a.private_calibration)
    sources=checked_sources()
    import torch
    torch.manual_seed(0)
    model,tok=load_local_model(MODEL)
    activations=[]
    for row in sources:
        # Only forward pass; never print model text or source identifiers.
        hidden=hidden_by_layer(model,encode(tok,row['prompt']))[LAYER]
        activations.append({
            'class':row['class'],'type':row['type'],
            'activation':hidden.float().tolist(),
        })
    # Attest all source bytes are still pinned after model execution.
    checked_sources()
    result=aggregate(activations,private)
    require_research_write_capacity(a.output)
    os.umask(0o077)
    write_exclusive(a.output,result)
    print(json.dumps({
        'aggregate_created':True,'prompts':result['prompt_count'],
        'isotropic_shams':result['isotropic_sham_count'],
        'model_generations':0,'weight_edits':0,
        'refusal_behavior_established':False,
        'release_approved':False,
    }))


if __name__=='__main__':
    main()
