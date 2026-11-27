#!/usr/bin/env python3
"""Paired significance tests for the current 16-shot PathoTME comparison.

The input is the published fold-level patient-AUROC table. No model is loaded,
no checkpoint is replayed and no prediction is recomputed. Every number this
script prints is derived from that one CSV.

Two families of tests are reported.

  Family A  adapter versus each available control, within cohort.
  Family B  each PathoTME variant versus the matched Native model, within cohort.

Both are paired over the fourteen architecture/encoder groups of a cohort,
using each group's mean across its five patient-disjoint folds.

The fourteen groups of a cohort are not independent: they score one shared
held-out patient universe under one shared set of fold splits, and the
TME-only logistic regression is literally one fit reused by every group in a
(cohort, fold) cell. A naive paired test over them therefore overstates the
evidence. Each contrast is reported twice: the naive signed-rank result, and
the same contrast after inflating the paired standard error by a design effect
estimated from the observed between-group correlation. The adjusted column is
the one to quote.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'docs/data/current-16shot/current_fold_metrics.csv'

COHORTS = ('nsclc', 'brca', 'crc', 'blca')
ENCODERS = ('plip', 'clip-rn50')
FOLDS = 5

# Stored condition key -> presentation name, matching docs/data/variant-names.json.
DISPLAY = {
    'native': 'Native',
    'tme_only': 'PathoTME-LR',
    'late_fusion_fixed_half': 'PathoTME-Fusion50',
    'late_fusion_validation': 'PathoTME-FusionVal',
    'zero': 'PathoTME-Adapter-Zero',
    'shuffled': 'PathoTME-Adapter-Shuffled',
    'actual': 'PathoTME-Adapter',
}
CONTROLS = ('native', 'zero', 'shuffled', 'tme_only', 'late_fusion_fixed_half')
VARIANTS = ('tme_only', 'late_fusion_validation', 'late_fusion_fixed_half', 'actual')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def value(row: dict, condition: str) -> float:
    """Return a stored AUROC, or NaN where the comparator was never computed."""
    raw = row.get(condition, '')
    if raw in ('', 'nan', 'None', None):
        return float('nan')
    return float(raw)


def load(path: Path) -> list[dict]:
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 280:
        raise ValueError(f'Expected 280 fold rows for the current 16-shot table, found {len(rows)}')
    methods = sorted({r['method'] for r in rows})
    if len(methods) != 7 or set(r['cohort'] for r in rows) != set(COHORTS):
        raise ValueError('Input is not the seven-architecture, four-cohort current table')
    return rows


def fold_vector(rows, cohort, method, encoder, condition):
    """Return a group's five fold values in fold order, or None if incomplete."""
    picked = {r['fold']: r for r in rows
              if (r['cohort'], r['method'], r['encoder']) == (cohort, method, encoder)}
    if len(picked) != FOLDS:
        return None
    return np.array([value(picked[str(f)], condition) for f in range(FOLDS)], dtype=float)


def groups(rows, cohort):
    methods = sorted({r['method'] for r in rows})
    return [(m, e) for m in methods for e in ENCODERS]


def group_means(rows, cohort, condition):
    """Mean over folds for each architecture/encoder group of one cohort.

    A group whose comparator was never computed contributes NaN and is dropped
    from the pairing rather than imputed. PathoTME-FusionVal is missing for
    HiVE-MIL, so its contrasts pair twelve groups instead of fourteen.
    """
    out = []
    for method, encoder in groups(rows, cohort):
        vector = fold_vector(rows, cohort, method, encoder, condition)
        if vector is None or np.isnan(vector).all():
            out.append(np.nan)
        else:
            out.append(np.nanmean(vector))
    return np.asarray(out, dtype=float)


def intraclass_correlation(rows, cohort, a, b):
    """Mean pairwise correlation of the per-fold paired difference across groups.

    This measures how much the fourteen groups move together when the same
    contrast is formed, which is exactly the dependence that a paired test over
    those groups would otherwise ignore. Groups with an incomplete comparator
    are dropped. Returns 0.0 when fewer than two usable groups remain.
    """
    matrix = []
    for method, encoder in groups(rows, cohort):
        x = fold_vector(rows, cohort, method, encoder, a)
        y = fold_vector(rows, cohort, method, encoder, b)
        if x is None or y is None:
            continue
        difference = x - y
        if np.isnan(difference).any() or np.std(difference) < 1e-12:
            continue
        matrix.append(difference)
    if len(matrix) < 2:
        return 0.0
    correlation = np.corrcoef(np.vstack(matrix))
    upper = correlation[np.triu_indices_from(correlation, 1)]
    if not np.isfinite(upper).any():
        return 0.0
    return float(np.nanmean(upper))


def paired_test(rows, cohort, a, b):
    """Paired comparison of two conditions over a cohort's groups."""
    x, y = group_means(rows, cohort, a), group_means(rows, cohort, b)
    usable = ~(np.isnan(x) | np.isnan(y))
    x, y = x[usable], y[usable]
    n = len(x)
    if n < 3:
        raise ValueError(f'{cohort} {a} vs {b}: too few paired groups ({n})')
    difference = x - y

    # Naive paired tests, which treat the groups as independent replicates.
    wilcoxon = stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided')
    naive_t = stats.ttest_rel(x, y)

    # Dependence-adjusted paired t. The design effect 1 + (m - 1) * ICC is the
    # standard cluster-survey inflation; the effective sample size is capped at
    # the real group count so a negative ICC cannot manufacture extra evidence.
    icc = intraclass_correlation(rows, cohort, a, b)
    deff = max(1.0, 1.0 + (n - 1) * icc)
    effective_n = min(float(n), n / deff)
    standard_error = difference.std(ddof=1) / np.sqrt(n) * np.sqrt(deff)
    degrees = max(1.0, effective_n - 1.0)
    if standard_error <= 0:
        adjusted_p, low, high = 1.0, float('nan'), float('nan')
    else:
        statistic = difference.mean() / standard_error
        adjusted_p = float(2 * stats.t.sf(abs(statistic), degrees))
        half = stats.t.ppf(0.975, degrees) * standard_error
        low, high = difference.mean() - half, difference.mean() + half
    return {
        'cohort': cohort, 'a': a, 'b': b,
        'label': f'{DISPLAY[a]} - {DISPLAY[b]}',
        'groups': int(n),
        'mean_difference': float(difference.mean()),
        'icc': icc, 'design_effect': float(deff), 'effective_n': float(effective_n),
        'p_wilcoxon': float(wilcoxon.pvalue),
        'p_paired_t': float(naive_t.pvalue),
        'p_adjusted': adjusted_p,
        'ci95_adjusted': [float(low), float(high)],
    }


def holm(results, key, out_key):
    """Holm-Bonferroni step-down correction within one family of tests."""
    order = sorted(range(len(results)), key=lambda i: results[i][key])
    running, m = 0.0, len(results)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * results[index][key]))
        results[index][out_key] = running
    return results


def table(results, title, pcols):
    lines = [f'### {title}', '',
             '| Cohort | Contrast | Groups | Mean delta | 95% CI (adj.) | n_eff | '
             + ' | '.join(pcols) + ' |',
             '|---|---|---:|---:|---:|---:|' + '---:|' * len(pcols)]
    for r in results:
        low, high = r['ci95_adjusted']
        cells = ' | '.join(f"{r[c]:.4f}" for c in pcols)
        lines.append(
            f"| {r['cohort'].upper()} | {r['label']} | {r['groups']} | "
            f"{r['mean_difference']:+.4f} | [{low:+.3f}, {high:+.3f}] | "
            f"{r['effective_n']:.1f} | {cells} |")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='Fold-level patient-AUROC table (default: the published 16-shot table)')
    parser.add_argument('--json', type=Path, help='Optional machine-readable output path')
    args = parser.parse_args(argv)

    rows = load(args.input)

    family_a, family_b = [], []
    for cohort in COHORTS:
        for control in CONTROLS:
            family_a.append(paired_test(rows, cohort, 'actual', control))
        for variant in VARIANTS:
            family_b.append(paired_test(rows, cohort, variant, 'native'))
    for family in (family_a, family_b):
        holm(family, 'p_wilcoxon', 'holm_wilcoxon')
        holm(family, 'p_adjusted', 'holm_adjusted')

    print(table(family_a, 'Family A - PathoTME-Adapter versus each control',
                ['p_wilcoxon', 'holm_wilcoxon', 'p_adjusted', 'holm_adjusted']))
    print()
    print(table(family_b, 'Family B - each PathoTME variant versus Native',
                ['p_wilcoxon', 'holm_wilcoxon', 'p_adjusted', 'holm_adjusted']))
    print()
    survivors = [r for r in family_b if r['holm_adjusted'] < 0.05 and r['mean_difference'] > 0]
    print(f'Family B results significant after adjustment and Holm: {len(survivors)}')
    for r in sorted(survivors, key=lambda r: r['holm_adjusted']):
        print(f"  {r['cohort'].upper():6s} {r['label']:40s} "
              f"{r['mean_difference']:+.4f}  Holm p = {r['holm_adjusted']:.5f}")

    # Record the input repository-relative. An absolute path would leak the
    # local account name into a published artifact.
    resolved = args.input.resolve()
    try:
        recorded = str(resolved.relative_to(ROOT))
    except ValueError:
        recorded = resolved.name
    payload = {
        'input': recorded,
        'input_sha256': digest(args.input),
        'script_sha256': digest(Path(__file__)),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'scipy': stats.__name__ and __import__('scipy').__version__,
        'unit_of_analysis': 'architecture/encoder group mean over five folds, paired within cohort',
        'adjustment': 'paired SE inflated by design effect 1 + (m-1)*ICC; ICC = mean pairwise '
                      'correlation of per-fold paired differences across groups; effective n capped at m',
        'multiplicity': 'Holm-Bonferroni within each family',
        'family_a_adapter_vs_controls': family_a,
        'family_b_variant_vs_native': family_b,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'\nWrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
