#!/usr/bin/env python3
"""Patient-level DeLong tests of each PathoTME variant against its Native model.

The five folds of a cohort are disjoint test partitions: every held-out patient
is scored exactly once, and the fold sizes sum to the cohort universe. Pooling
the folds therefore yields one prediction per patient with no repetition, and a
paired comparison can be made on the full cohort rather than on five fold means.

Every condition scores the identical patients, so the two ROC curves being
compared are correlated. DeLong's test is the standard treatment for that case
and is used here with the Sun and Xu fast midrank covariance.

Scope. The published fusion predictions come from the September 14 baseline
campaign, whose HiVE-MIL entries predate the October 3 refresh. Following
AGENTS.md those entries are excluded and only the six remaining architectures
are tested; the HiVE gap is reported rather than filled from stale records.

Slide probabilities are aggregated to patients by arithmetic mean, matching
campaigns/baselines_20260914/baseline_core.py.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ('nsclc', 'brca', 'crc', 'blca')
EXCLUDED_METHOD = 'hive_mil'          # stale pre-refresh entries, see AGENTS.md
VARIANTS = {
    'tme_only_probability_1': 'PathoTME-LR',
    'late_fusion_fixed_half_probability_1': 'PathoTME-Fusion50',
    'late_fusion_validation_probability_1': 'PathoTME-FusionVal',
}
NATIVE = 'native_probability_1'


# --------------------------------------------------------------------------
# DeLong's test for two correlated ROC curves.
# --------------------------------------------------------------------------

def _midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_variance(labels, *score_sets):
    """Return AUCs and their covariance matrix for correlated predictors.

    Implements the structural-components estimator: V10 over positives and V01
    over negatives, as in Sun and Xu's O(n log n) formulation of DeLong.
    """
    labels = np.asarray(labels, dtype=int)
    positive = labels == 1
    negative = ~positive
    m, n = int(positive.sum()), int(negative.sum())
    if m == 0 or n == 0:
        raise ValueError('DeLong needs both classes present')
    k = len(score_sets)
    v10 = np.empty((k, m))
    v01 = np.empty((k, n))
    aucs = np.empty(k)
    for i, scores in enumerate(score_sets):
        scores = np.asarray(scores, dtype=float)
        pos, neg = scores[positive], scores[negative]
        tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(scores)
        aucs[i] = (tz[positive].sum() - m * (m + 1) / 2) / (m * n)
        v10[i] = (tz[positive] - tx) / n
        v01[i] = 1.0 - (tz[negative] - ty) / m
    s10 = np.cov(v10, ddof=1) if k > 1 else np.array([[np.var(v10[0], ddof=1)]])
    s01 = np.cov(v01, ddof=1) if k > 1 else np.array([[np.var(v01[0], ddof=1)]])
    covariance = np.atleast_2d(s10) / m + np.atleast_2d(s01) / n
    return aucs, covariance


def delong_test(labels, a, b):
    """Two-sided DeLong comparison of two correlated AUCs on the same cases."""
    aucs, cov = delong_variance(labels, a, b)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ cov @ contrast)
    difference = float(aucs[0] - aucs[1])
    if variance <= 0:
        return {'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]),
                'difference': difference, 'se': 0.0, 'z': float('nan'),
                'p': 1.0, 'ci95': [difference, difference]}
    se = float(np.sqrt(variance))
    z = difference / se
    return {'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]),
            'difference': difference, 'se': se, 'z': float(z),
            'p': float(2 * stats.norm.sf(abs(z))),
            'ci95': [difference - 1.96 * se, difference + 1.96 * se]}


# --------------------------------------------------------------------------
# Data assembly.
# --------------------------------------------------------------------------

def discover(fusion_root: Path):
    """Map (cohort, method, encoder) -> {fold: predictions.csv}."""
    cells = collections.defaultdict(dict)
    for entry in sorted(os.listdir(fusion_root)):
        head, _, fold = entry.rpartition('_')
        if not fold.isdigit():
            continue
        parts = head.split('_')
        cohort, encoder, method = parts[0], parts[-1], '_'.join(parts[1:-1])
        path = fusion_root / entry / 'predictions.csv'
        if path.is_file():
            cells[(cohort, method, encoder)][int(fold)] = path
    return cells


def pooled_patients(fold_paths: dict) -> dict:
    """Pool disjoint folds into one patient-level table for a group.

    Slide probabilities are averaged within a patient, exactly as the baseline
    campaign does. The disjointness of the folds is verified, not assumed: a
    patient appearing in two folds aborts the run.
    """
    slides = collections.defaultdict(lambda: collections.defaultdict(list))
    labels, origin = {}, {}
    for fold, path in sorted(fold_paths.items()):
        with Path(path).open(newline='') as handle:
            for row in csv.DictReader(handle):
                case = row['case_id']
                if case in origin and origin[case] != fold:
                    raise ValueError(
                        f'Patient {case} appears in folds {origin[case]} and {fold}; '
                        'folds are not disjoint and pooling would double-count')
                origin[case] = fold
                label = int(row['label'])
                if labels.setdefault(case, label) != label:
                    raise ValueError(f'Conflicting label for patient {case}')
                for column in (NATIVE, *VARIANTS):
                    value = row.get(column, '')
                    if value not in ('', 'nan', 'None', None):
                        slides[case][column].append(float(value))
    cases = sorted(labels)
    table = {'case_id': cases, 'label': np.array([labels[c] for c in cases], dtype=int)}
    for column in (NATIVE, *VARIANTS):
        values = []
        for case in cases:
            got = slides[case].get(column, [])
            values.append(float(np.mean(got)) if got else np.nan)
        table[column] = np.asarray(values, dtype=float)
    return table


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def holm(results, key, out_key):
    order = sorted(range(len(results)), key=lambda i: results[i][key])
    running, m = 0.0, len(results)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * results[index][key]))
        results[index][out_key] = running
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fusion-root', type=Path, required=True,
                        help='Directory of per-cell fusion predictions.csv folders')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args(argv)

    cells = discover(args.fusion_root)
    tested = {k: v for k, v in cells.items()
              if k[1] != EXCLUDED_METHOD and len(v) == 5}
    skipped = sorted(k for k in cells if k[1] == EXCLUDED_METHOD)

    results = []
    for (cohort, method, encoder), folds in sorted(tested.items()):
        table = pooled_patients(folds)
        for column, label in VARIANTS.items():
            variant, native = table[column], table[NATIVE]
            usable = ~(np.isnan(variant) | np.isnan(native))
            if usable.sum() < 20 or len(set(table['label'][usable])) < 2:
                continue
            test = delong_test(table['label'][usable], variant[usable], native[usable])
            results.append({'cohort': cohort, 'method': method, 'encoder': encoder,
                            'variant': label, 'patients': int(usable.sum()), **test})
    holm(results, 'p', 'holm')

    print(f'Groups tested: {len(tested)}  (HiVE-MIL excluded: {len(skipped)} groups, '
          f'stale pre-refresh predictions)')
    print(f'DeLong comparisons: {len(results)}, Holm-corrected together\n')
    for cohort in COHORTS:
        subset = [r for r in results if r['cohort'] == cohort]
        if not subset:
            continue
        n = subset[0]['patients']
        print(f'--- {cohort.upper()}  (pooled across 5 disjoint folds, n={n} patients)')
        for label in VARIANTS.values():
            rows = [r for r in subset if r['variant'] == label]
            wins = [r for r in rows if r['holm'] < 0.05 and r['difference'] > 0]
            losses = [r for r in rows if r['holm'] < 0.05 and r['difference'] < 0]
            median = float(np.median([r['difference'] for r in rows]))
            print(f'    {label:20s} median dAUC {median:+.4f}  '
                  f'significant better {len(wins)}/{len(rows)}  worse {len(losses)}/{len(rows)}')
        print()

    wins = [r for r in results if r['holm'] < 0.05 and r['difference'] > 0]
    print(f'Total comparisons significantly favouring a PathoTME variant: {len(wins)}/{len(results)}')
    for r in sorted(wins, key=lambda r: r['holm'])[:12]:
        print(f"   {r['cohort'].upper():6s} {r['variant']:20s} vs {r['method']}/{r['encoder']:10s} "
              f"dAUC {r['difference']:+.4f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] "
              f"Holm p={r['holm']:.2e}")

    if args.json:
        # Bind the result to its inputs without republishing patient-level
        # predictions: hash every consumed CSV and fold the per-cell digests
        # into one manifest digest that a reader with the private store can
        # recompute. Only basenames are recorded, never the local store path.
        manifest = {}
        for (cohort, method, encoder), folds in sorted(tested.items()):
            for fold, path in sorted(folds.items()):
                manifest[f'{cohort}/{method}/{encoder}/fold{fold}'] = digest(path)
        manifest_digest = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        payload = {
            'test': 'DeLong two-sided, correlated ROC curves on identical patients',
            'input_cells': len(manifest),
            'input_manifest_sha256': manifest_digest,
            'input_cell_sha256': manifest,
            'pooling': 'five disjoint fold test partitions pooled; each patient scored once',
            'patient_aggregation': 'arithmetic mean of held-out slide probabilities',
            'excluded': {'method': EXCLUDED_METHOD,
                         'reason': 'September 14 baseline predictions predate the '
                                   'October 3 HiVE refresh required by AGENTS.md',
                         'groups': ['/'.join(k) for k in skipped]},
            'multiplicity': 'Holm-Bonferroni across all reported comparisons',
            'script_sha256': digest(Path(__file__)),
            'comparisons': results,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'\nWrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
