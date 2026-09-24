"""Immutable multi-domain IC manifests and complete, balanced anchor sweeps."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from model.train.interface_flux_data import evaluate_manifest_case, sha256_json

REGIMES = ('linear_landau', 'nonlinear_landau_weak', 'nonlinear_landau_strong')
AMPLITUDES = (.001, .03, .3)
FUNDAMENTALS = tuple(i / 100 for i in range(25, 50, 2))
HELDOUT_FUNDAMENTALS = FUNDAMENTALS[2::3]
TRAIN_FUNDAMENTALS = tuple(k for k in FUNDAMENTALS if k not in HELDOUT_FUNDAMENTALS)
ANCHORS = 526


def build_manifest(original: dict, seed: int = 1729) -> dict:
    rng = np.random.default_rng(seed)
    cases = []
    for old in original['cases']:
        case = dict(old)
        case.update(domain_length=4 * math.pi, fundamental=.5, provenance='original',
                    panel='original', family='mixture')
        cases.append(case)

    def add(k, regime_i, harmonics, split, panel, serial, isolated=False):
        modes = np.asarray(harmonics) * k
        weights = np.ones(len(modes)) if isolated else rng.uniform(.5, 1.5, len(modes))
        phases = np.zeros(len(modes)) if isolated else rng.uniform(0, 2 * math.pi, len(modes))
        # Sum-absolute normalization provides a resolution-independent positivity bound.
        case = dict(case_id=f'k{round(k*100):03d}_{regime_i}_{panel}_{serial:02d}',
                    regime=REGIMES[regime_i], epsilon=AMPLITUDES[regime_i],
                    modes=modes.tolist(), mode_weights=weights.tolist(),
                    relative_phases=phases.tolist(), shape_normalization=1/float(np.sum(abs(weights))),
                    domain_length=2*math.pi/k, fundamental=k, split=split,
                    panel=panel, family='isolated' if isolated else 'mixture', provenance='expanded')
        cases.append(case)

    for k in TRAIN_FUNDAMENTALS:
        for r in range(3):
            for j in range(1, 5):
                add(k, r, [j], 'train', 'train', j, True)
            for q in range(2):
                count = 2 + ((2*r+q) % 3)
                hs = np.sort(rng.choice(np.arange(1,5), count, replace=False))
                add(k, r, hs, 'train', 'train', 5+q)
            add(k, r, [1,2,3,4], 'heldout', 'familiar_domain', 0)
    for k in HELDOUT_FUNDAMENTALS:
        for r in range(3):
            add(k, r, [1], 'heldout', 'unseen_domain', 0, True)
            add(k, r, [1,2,3,4], 'heldout', 'unseen_domain', 1)
    result = dict(format='vpml_multi_domain_v1', seed=seed, cases=cases,
                  original_manifest_sha256=sha256_json(original),
                  fundamentals=list(FUNDAMENTALS), heldout_fundamentals=list(HELDOUT_FUNDAMENTALS),
                  train_count=sum(c['split']=='train' for c in cases),
                  development_count=sum(c['split']=='heldout' for c in cases),
                  anchors_per_case=ANCHORS, anchor_spacing=.2, horizon=120.,
                  preparation=5., scored_horizon=10.)
    validate_manifest(result)
    result['manifest_sha256'] = sha256_json(result)
    return result


def validate_manifest(manifest):
    cases = manifest['cases']
    ids = [c['case_id'] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate case ID')
    for c in cases:
        indices = np.asarray(c['modes']) * c['domain_length']/(2*math.pi)
        if not np.allclose(indices, np.rint(indices), rtol=0, atol=1e-12):
            raise ValueError('Nonperiodic wavenumber')
        if c['provenance']=='expanded' and c['split']=='train' and c['fundamental'] in HELDOUT_FUNDAMENTALS:
            raise ValueError('Held-out domain leaked into training')
        x = np.arange(1024) * c['domain_length']/1024
        if np.min(1+evaluate_manifest_case(c,x)) <= 0:
            raise ValueError('Invalid IC density')
    counts = [sum(c['split']=='train' and c['regime']==r for c in cases) for r in REGIMES]
    if counts != [70,70,70] or sum(c['split']=='heldout' for c in cases)!=63:
        raise ValueError(f'Unexpected manifest exposure: {counts}')


def epoch_batches(manifest, epoch, seed=1729, per_regime=16):
    """Shuffle every training anchor once. No dropped final samples or replacement."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    pools = []
    for regime in REGIMES:
        indices = [i for i,c in enumerate(manifest['cases']) if c['split']=='train' and c['regime']==regime]
        rows = np.asarray([(i,a) for i in indices for a in range(ANCHORS)], dtype=np.int32)
        rng.shuffle(rows)
        pools.append(rows)
    if len({len(p) for p in pools}) != 1:
        raise ValueError('Regime exposure must match')
    for s in range(0,len(pools[0]),per_regime):
        yield np.concatenate([p[s:s+per_regime] for p in pools])


def write_new_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')
