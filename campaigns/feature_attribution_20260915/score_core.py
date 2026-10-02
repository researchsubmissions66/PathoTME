"""Common TME sensitivity scores; numpy only, no fitting or model selection."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = 'tme_train_mean_ablation_probability_delta_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def dump(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    pos = x >= 0
    result[pos] = 1 / (1 + np.exp(-x[pos]))
    exp = np.exp(x[~pos])
    result[~pos] = exp / (1 + exp)
    return result


def linear_scores(z, coef, intercept, units):
    z, coef = np.asarray(z, float), np.asarray(coef, float)
    if z.ndim != 2 or coef.shape != (z.shape[1],) or not np.isfinite(z).all() or not np.isfinite(coef).all() or not np.isfinite(intercept):
        raise ValueError('Invalid linear attribution input')
    log_contributions = z * coef
    logit = log_contributions.sum(axis=1) + intercept
    probability = sigmoid(logit)
    removed = np.stack([log_contributions[:, u['indices']].sum(axis=1) for u in units], axis=1)
    replaced = sigmoid(logit[:, None] - removed)
    positive_scores = 100 * (probability[:, None] - replaced)
    return probability, np.stack([-positive_scores, positive_scores], axis=-1), log_contributions


def patient_scores(rows, scores):
    if scores.shape[0] != len(rows) or not np.isfinite(scores).all():
        raise ValueError('Attribution rows/scores mismatch')
    ids = sorted({r['case_id'] for r in rows})
    values, labels = [], []
    for case in ids:
        idx = [i for i, r in enumerate(rows) if r['case_id'] == case]
        labs = {int(rows[i]['label']) for i in idx}
        if len(labs) != 1:
            raise ValueError('Conflicting patient labels')
        values.append(scores[idx].mean(axis=0)); labels.append(labs.pop())
    return ids, np.asarray(labels, dtype=int), np.stack(values)


def rank_fold(meta, rows, scores, units):
    cases, labels, values = patient_scores(rows, scores)
    result = []
    for cls, classname in enumerate(meta['classes']):
        for level in ['feature', 'group']:
            indices = [i for i, u in enumerate(units) if u['level'] == level]
            magnitude = np.abs(values[:, indices, cls]).mean(axis=0)
            order = sorted(range(len(indices)), key=lambda k: (-magnitude[k], units[indices[k]]['name']))
            ranks = {indices[j]: rank + 1 for rank, j in enumerate(order)}
            for i in indices:
                v = values[:, i, cls]
                result.append({**meta, 'level': level, 'feature_name': units[i]['name'],
                               'class_index': cls, 'class_name': classname,
                               'mean_absolute_pp': float(np.abs(v).mean()),
                               'mean_signed_pp': float(v.mean()),
                               'positive_patient_fraction': float((v > 1e-8).mean()),
                               'negative_patient_fraction': float((v < -1e-8).mean()),
                               'patients': len(cases), 'slides': len(rows),
                               'rank_in_fold': ranks[i] if float(np.abs(v).mean()) > 1e-12 else None})
    return result


def aggregate_ranks(fold_rows):
    from collections import defaultdict
    grouped = defaultdict(list)
    scope = ['variant', 'cohort', 'panel', 'method', 'encoder', 'shots', 'level', 'feature_name', 'class_index', 'class_name']
    for r in fold_rows:
        grouped[tuple(r[k] for k in scope)].append(r)
    output = []
    for key, rows in grouped.items():
        if len({r['fold'] for r in rows}) != len(rows):
            raise ValueError('Duplicate attribution fold for condition/feature/class')
        values = [r['mean_absolute_pp'] for r in rows]
        output.append({**dict(zip(scope, key)),
                       'mean_absolute_pp': float(np.mean(values)),
                       'fold_sd_absolute_pp': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       'mean_signed_pp': float(np.mean([r['mean_signed_pp'] for r in rows])),
                       'mean_rank': float(np.mean([r['rank_in_fold'] for r in rows if r['rank_in_fold'] is not None])) if any(r['rank_in_fold'] is not None for r in rows) else None,
                       'top10_fold_fraction': float(np.mean([r['rank_in_fold'] is not None and r['rank_in_fold'] <= 10 for r in rows])),
                       'folds': len(rows), 'fold_ids': ','.join(map(str, sorted(r['fold'] for r in rows))),
                       'patients': sum(r['patients'] for r in rows),
                       'complete_five_fold': set(r['fold'] for r in rows) == set(range(5))
                           and all(r.get('full_test_patients', True) for r in rows)})
    return sorted(output, key=lambda r: (r['cohort'], r['variant'], r['method'], r['encoder'], r['class_index'], r['level'], -r['mean_absolute_pp'], r['feature_name']))


def make_units(names, groups):
    if len(set(names)) != len(names) or sorted(i for g in groups for i in g['indices']) != list(range(len(names))):
        raise ValueError('Groups must partition exact unique feature names')
    return [{'level': 'feature', 'name': n, 'indices': [i]} for i, n in enumerate(names)] + [dict(g, level='group') for g in groups]
