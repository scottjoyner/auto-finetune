"""Research-only matched XSTest layer-10 LFM3 direction audit.

Uses 18 source-index-matched safe/contrast pairs (three each in six
verified XSTest category families). Produces only aggregate features;
no generations, weights, source IDs or raw activations exported.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random

from src.calibration_direction_stability import (
    WIDTH, LAYER, MODEL_SHA, DIRECTION_FILE_SHA, _dot, _private_input,
    _unit, write_exclusive,
)
from src.xstest_direction_transfer import _checked_candidates
from src.xstest_450_eval import SOURCE_FILE, SOURCE_SHA256, read_verified_dataset
from src.refusal_probe import MODEL, load_local_model, hidden_by_layer, encode
from src.research_job_guard import require_research_write_capacity
from src.reversible_alpha_controls import DIRECTIONS
from src.capture_calibration_deltas import sha256

FAMILIES = ('definitions', 'figurative_language', 'historical_events',
            'homonyms', 'safe_contexts', 'safe_targets')
SEED = 9418
SAMPLES_PER_FAMILY = 3
PAIR_COUNT = len(FAMILIES)*SAMPLES_PER_FAMILY
SHAMS = range(32000,32128)


def selection(rows):
    """Pin matched *source-index* pairs; verify same human topic/focus.

    No unsupervised semantic similarity inference or nearest-neighbor
    matching. Each family has exactly 25 source rows and 25 contrasts.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row['type']].append(row)
    rng = random.Random(SEED)
    selected = []
    for family in FAMILIES:
        left, right = groups[family], groups['contrast_'+family]
        if len(left)!=25 or len(right)!=25:
            raise ValueError('source category size mismatch')
        for i in sorted(rng.sample(range(25), SAMPLES_PER_FAMILY)):
            a,b=left[i],right[i]
            if (a['class']!='safe' or b['class']!='unsafe' or
                a['focus']!=b['focus'] or
                not a.get('prompt') or not b.get('prompt') or
                not a['id'].isdigit() or not b['id'].isdigit()):
                raise ValueError('matched source pair failed topic/label/ID checks')
            selected.append((family,a,b))
    if len(selected)!=PAIR_COUNT or len(set((a['id'],b['id']) for _,a,b in selected))!=PAIR_COUNT:
        raise ValueError('source pairing not unique')
    return selected


def _raw_pair_check(pairs, *, full):
    if not isinstance(pairs,list) or not pairs:
        raise ValueError('empty matched pair set')
    counts = defaultdict(int)
    seen=set()
    out=[]
    for r in pairs:
        if not isinstance(r,dict) or set(r)!={
                'family','safe_activation','unsafe_activation','char_length_delta'}:
            raise ValueError('malformed private matched pair')
        family=r['family']
        if family not in FAMILIES:
            raise ValueError('unknown pair family')
        counts[family]+=1
        if (type(r['char_length_delta']) is not int or
            abs(r['char_length_delta'])>50000):
            raise ValueError('invalid paired prompt length delta')
        units=[]
        for k in ('safe_activation','unsafe_activation'):
            vec=r[k]
            if (not isinstance(vec,list) or len(vec)!=WIDTH or
                any(type(x) not in (float,int) or not math.isfinite(x)
                    for x in vec)):
                raise ValueError('invalid private activation tensor')
            units.append(_unit(vec))
        out.append((family,units[0],units[1],r['char_length_delta']))
    if full and (len(pairs)!=PAIR_COUNT or len(counts)!=len(FAMILIES) or
                 any(counts[k]!=SAMPLES_PER_FAMILY for k in FAMILIES)):
        raise ValueError('incomplete fixed 18-pair matched experiment')
    return out


def _paired_stats(records,direction):
    signed=[_dot(unsafe,direction)-_dot(safe,direction)
            for _,safe,unsafe,_ in records]
    by=defaultdict(list)
    for row,delta in zip(records,signed):
        by[row[0]].append(delta)
    mean=math.fsum(signed)/len(signed)
    catmeans=[math.fsum(by[k])/len(by[k]) for k in sorted(by)]
    stable_pairs=sum(x>0 for x in signed)
    remove_one=[math.fsum(v for j,v in enumerate(catmeans) if j!=i) /
                 (len(catmeans)-1) for i in range(len(catmeans))] if len(catmeans)>1 else []
    return {
        'matched_pair_count':len(signed),
        'mean_signed_unsafe_minus_safe_cosine':round(mean,6),
        'median_signed_pair_delta':round(sorted(signed)[len(signed)//2],6),
        'minimum_signed_pair_delta':round(min(signed),6),
        'maximum_signed_pair_delta':round(max(signed),6),
        'positive_matched_pair_count':stable_pairs,
        'positive_family_mean_count':sum(x>0 for x in catmeans),
        'leave_one_family_out_mean_range':[
            round(min(remove_one),6),round(max(remove_one),6)
        ] if remove_one else None,
    }


def analyze(pairs, private, *, full=True):
    verified=_raw_pair_check(pairs,full=full)
    candidates=_checked_candidates(private)
    measures={k:_paired_stats(verified,v) for k,v in candidates.items()}
    shams=[]
    for seed in SHAMS:
        rng=random.Random(seed)
        v=_unit([rng.gauss(0,1) for _ in range(WIDTH)])
        shams.append(abs(_paired_stats(verified,v)['mean_signed_unsafe_minus_safe_cosine']))
    fitted=abs(measures['original_M1_fitted']['mean_signed_unsafe_minus_safe_cosine'])
    d=sorted(shams)
    return {
        'evidence_class':'matched semantic-family XSTest activation differences; not behavioral refusal',
        'source_model_sha256':MODEL_SHA,
        'source_direction_sha256':DIRECTION_FILE_SHA,
        'source_dataset_sha256':SOURCE_SHA256,
        'selection':'6 manual verified source-index paired category families x 3 pairs, seed 9418',
        'matched_pairs':len(verified),
        'source_category_families':len(set(x[0] for x in verified)),
        'new_model_forward_passes':2*len(verified),
        'layer':LAYER,
        'candidate_matched_diagnostics':measures,
        'isotropic_sham_count':len(d),
        'isotropic_sham_first_seed':SHAMS.start,
        'isotropic_sham_median_absolute_mean_gap':round(d[len(d)//2],6),
        'isotropic_sham_maximum_absolute_mean_gap':round(d[-1],6),
        'isotropic_shams_at_least_as_large_as_fitted':sum(x>=fitted for x in d),
        'shams_are_causal_null':False,
        'paired_char_length_delta_mean':round(math.fsum(x[3] for x in verified)/len(verified),6),
        'matched_design_preselected_from_xstest':True,
        'models_weights_changed':False,
        'model_generations':0,
        'human_refusal_grades':0,
        'model_release_approved':False,
        'raw_prompts_or_pair_activations_exported':False,
        'limitations':[
            'Index-matched prompts preserve source focus but do not eliminate all topic/framing/length confounds',
            '18 selected pairs across six families are small and not independently sampled from a wider population',
            'This source previously participated in A/B/C response evaluation; source is not a virgin benchmark',
            'A positive activation difference is not a refusal outcome or proof of a causal direction',
            'Sham direction geometry is descriptive; no behavioral null or statistical p-value',
            'Paired prompt length differences can still confound the final-token activation',
            'Equal-unit-weight direction was proposed after inspecting the earlier six-pair data',
        ],
    }


def verified_sources():
    if sha256(MODEL/'model.safetensors')!=MODEL_SHA:
        raise ValueError('source checkpoint changed')
    if sha256(DIRECTIONS)!=DIRECTION_FILE_SHA:
        raise ValueError('fitted direction source changed')
    if sha256(SOURCE_FILE)!=SOURCE_SHA256:
        raise ValueError('XSTest source changed')
    return selection(read_verified_dataset())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('aggregate already exists')
    require_research_write_capacity(args.output)
    private=_private_input(args.private_calibration)
    pairs=verified_sources()
    import torch
    torch.manual_seed(0)
    model,tok=load_local_model(MODEL)
    raw=[]
    for family,a,b in pairs:
        safe=hidden_by_layer(model,encode(tok,a['prompt']))[LAYER].float().tolist()
        unsafe=hidden_by_layer(model,encode(tok,b['prompt']))[LAYER].float().tolist()
        raw.append({
            'family':family,'safe_activation':safe,'unsafe_activation':unsafe,
            'char_length_delta':len(b['prompt'])-len(a['prompt']),
        })
    # No raw prompts, IDs or activations serialized, even to the SSD.
    verified_sources()
    result=analyze(raw,private)
    require_research_write_capacity(args.output)
    os.umask(0o077)
    write_exclusive(args.output,result)
    print(json.dumps({
        'aggregate_created':True,'matched_pairs':result['matched_pairs'],
        'model_generations':0,'model_weights_changed':False,
        'human_behavioral_effect_established':False,
        'model_release_approved':False,
    }))


if __name__=='__main__':
    main()
