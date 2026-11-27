# Significance testing for the current 16-shot comparison

This document records a paired significance analysis of the published 16-shot
patient-AUROC table. It is reproducible from the repository with one command
and loads no checkpoint, replays no model and recomputes no prediction.

Run from the repository root:

```bash
python scripts/run_significance_tests.py \
  --json docs/data/current-16shot/significance_tests.json
```

Input: [`docs/data/current-16shot/current_fold_metrics.csv`](../data/current-16shot/current_fold_metrics.csv),
280 rows = 4 cohorts x 7 architectures x 2 encoders x 5 patient-disjoint folds.
Machine-readable output with full provenance, including the input and script
SHA-256: [`docs/data/current-16shot/significance_tests.json`](../data/current-16shot/significance_tests.json).

## Headline result

**One comparison survives both the dependence adjustment and multiplicity
correction: PathoTME-LR beats the matched Native model in BLCA by +0.1015
patient AUROC (Holm-adjusted p = 0.012).**

Nothing else does. In particular, PathoTME-Adapter does not significantly
outperform Native in any cohort once group dependence is accounted for. The
adapter's apparent advantages in NSCLC and BRCA are an artifact of treating
fourteen correlated architecture/encoder groups as independent replicates.

## Why the obvious test is wrong

The tempting analysis pairs the seven conditions across all 56
cohort/architecture/encoder groups, or across the 280 fold rows, and runs a
signed-rank test. That overstates the evidence for three compounding reasons.

**One patient universe per cohort.** Every group in a cohort scores the same
held-out patients: 944 NSCLC, 900 BRCA, 536 CRC, 380 BLCA. Fourteen
architectures agreeing on those patients is not fourteen independent
validations. This is the hard ceiling on any test in this document.

**One shared set of fold splits.** The five folds are five few-shot draws from
the same universe, and all groups in a cohort use the identical splits.

**PathoTME-LR is literally one fit.** The TME-only logistic regression is
shared by every group in a `(cohort, fold)` cell. The entire 280-row table
contains only 20 distinct `tme_only` values. Any contrast involving LR, and
any contrast involving a fusion variant that contains LR, therefore has far
less independent information than the group count suggests.

## Unit of analysis

Each architecture/encoder group contributes its mean patient AUROC across its
five folds. Contrasts are paired over the groups within a cohort, so each test
has at most 14 paired differences (12 where PathoTME-FusionVal is missing for
HiVE-MIL; those groups are dropped, never imputed). Cohorts are never pooled,
because a pooled test would need to treat four cohorts as exchangeable.

## Dependence adjustment

For each contrast separately, the between-group dependence is estimated as the
mean pairwise Pearson correlation of the per-fold paired difference vectors
across groups. That correlation is converted to the standard cluster-survey
design effect

```
DEFF  = 1 + (m - 1) * ICC          m = number of paired groups
n_eff = m / DEFF                   capped at m
SE    = sd(difference)/sqrt(m) * sqrt(DEFF)
```

and the paired t statistic is evaluated on `n_eff - 1` degrees of freedom. A
negative ICC cannot manufacture evidence: `DEFF` is floored at 1.

Estimating the ICC **per contrast** matters. An earlier draft of this analysis
reused a single cohort-level ICC for every contrast and reported four
significant results; that was wrong. Contrasts differ sharply in how correlated
they are across groups. `PathoTME-Fusion50 - Native` has `n_eff` between 1.6
and 2.3 in every cohort, because Fusion50 is an average of Native and the
single shared LR fit, so its advantage over Native is largely one piece of
shared information rather than fourteen. `PathoTME-Adapter - Native` in CRC and
BLCA has `n_eff = 14.0`, because those adapters genuinely vary across
architectures.

## Multiplicity

Holm-Bonferroni within each family. Family A is the adapter against each of its
five controls in four cohorts (20 tests). Family B is each of four PathoTME
variants against Native in four cohorts (16 tests). Families are corrected
separately because they answer different questions.

## Results

Both p-columns are reported for every contrast. `p_wilcoxon` is the naive
signed-rank test that ignores group dependence; it is shown only so the size of
the correction is visible. **`holm_adjusted` is the column to quote.**

### Family B - each PathoTME variant versus Native

| Cohort | Contrast | Groups | Mean delta | 95% CI (adj.) | n_eff | Holm (naive) | Holm (adjusted) |
|---|---|---:|---:|---:|---:|---:|---:|
| NSCLC | PathoTME-LR | 14 | +0.0715 | [-0.821, +0.964] | 1.7 | 0.0671 | 1.0000 |
| NSCLC | PathoTME-FusionVal | 12 | +0.0994 | [-0.191, +0.390] | 2.4 | 0.0049 | 1.0000 |
| NSCLC | PathoTME-Fusion50 | 14 | +0.0860 | [-0.395, +0.567] | 1.6 | 0.0020 | 1.0000 |
| NSCLC | PathoTME-Adapter | 14 | +0.0484 | [-0.033, +0.130] | 3.8 | 0.0060 | 1.0000 |
| BRCA | PathoTME-LR | 14 | +0.1916 | [-0.162, +0.545] | 2.2 | 0.0020 | 1.0000 |
| BRCA | PathoTME-FusionVal | 12 | +0.1846 | [-0.365, +0.735] | 1.7 | 0.0049 | 1.0000 |
| BRCA | PathoTME-Fusion50 | 14 | +0.1299 | [-0.411, +0.671] | 1.7 | 0.0020 | 1.0000 |
| BRCA | PathoTME-Adapter | 14 | +0.1085 | [-0.284, +0.501] | 2.3 | 0.0020 | 1.0000 |
| CRC | PathoTME-LR | 14 | -0.0012 | [-0.565, +0.562] | 2.0 | 1.0000 | 1.0000 |
| CRC | PathoTME-FusionVal | 12 | +0.0202 | [-0.088, +0.128] | 2.8 | 0.5288 | 1.0000 |
| CRC | PathoTME-Fusion50 | 14 | +0.0344 | [-0.123, +0.192] | 2.3 | 0.0073 | 1.0000 |
| CRC | PathoTME-Adapter | 14 | +0.0023 | [-0.016, +0.021] | 14.0 | 1.0000 | 1.0000 |
| **BLCA** | **PathoTME-LR** | 14 | **+0.1015** | **[+0.064, +0.139]** | 6.3 | 0.0020 | **0.0123** |
| BLCA | PathoTME-FusionVal | 12 | +0.0502 | [+0.003, +0.097] | 4.0 | 0.0049 | 0.5981 |
| BLCA | PathoTME-Fusion50 | 14 | +0.0574 | [-0.170, +0.285] | 1.8 | 0.0020 | 1.0000 |
| BLCA | PathoTME-Adapter | 14 | +0.0209 | [+0.004, +0.038] | 14.0 | 0.0671 | 0.3012 |

### Family A - PathoTME-Adapter versus each control

Two contrasts survive, and both describe where the adapter stands rather than
what it achieves: it beats its own shuffled-TME control in NSCLC, and it loses
to PathoTME-LR in BLCA.

| Cohort | Contrast | Groups | Mean delta | 95% CI (adj.) | n_eff | Holm (adjusted) |
|---|---|---:|---:|---:|---:|---:|
| **NSCLC** | **Adapter - Adapter-Shuffled** | 14 | **+0.0710** | **[+0.033, +0.109]** | 14.0 | **0.0260** |
| **BLCA** | **Adapter - PathoTME-LR** | 14 | **-0.0806** | **[-0.116, -0.045]** | 6.5 | **0.0344** |
| NSCLC | Adapter - Native | 14 | +0.0484 | [-0.033, +0.130] | 3.8 | 1.0000 |
| NSCLC | Adapter - Adapter-Zero | 14 | +0.0458 | [-0.015, +0.107] | 4.7 | 1.0000 |
| BRCA | Adapter - Native | 14 | +0.1085 | [-0.284, +0.501] | 2.3 | 1.0000 |
| BRCA | Adapter - Adapter-Zero | 14 | +0.1057 | [-0.155, +0.366] | 2.6 | 1.0000 |
| BRCA | Adapter - PathoTME-LR | 14 | -0.0831 | [-0.242, +0.075] | 3.2 | 1.0000 |
| BLCA | Adapter - Native | 14 | +0.0209 | [+0.004, +0.038] | 14.0 | 0.3451 |
| BLCA | Adapter - Adapter-Zero | 14 | +0.0208 | [+0.004, +0.038] | 14.0 | 0.3451 |
| CRC | Adapter - Native | 14 | +0.0023 | [-0.016, +0.021] | 14.0 | 1.0000 |
| CRC | Adapter - Adapter-Zero | 14 | -0.0004 | [-0.018, +0.017] | 14.0 | 1.0000 |

The full 20-contrast table is in the JSON output.

## Reading these numbers

**The NSCLC shuffled-control result is the adapter's strongest claim.** Beating
a different patient's TME row by +0.071 with `n_eff = 14` says the conditioning
uses slide-specific measured information rather than acting as extra capacity.
It does **not** say the adapter beats Native, which it fails to do after
correction in all four cohorts.

**Effect size and significance point in opposite directions across cohorts.**
BRCA has the largest raw gains and no significant result; BLCA has smaller
gains and the only win. That is entirely the dependence structure: BRCA's
groups are the most correlated, BLCA's the least. Significance here tracks
where the design has power, not where the method works best.

**The LR comparison has an uncounted source of uncertainty.** In a
`PathoTME-LR - Native` contrast, the LR side is one shared fit, so all
between-group variance comes from Native differing across architectures. The
test therefore asks "is the single LR fit better than the typical native
model?" and does not propagate uncertainty in the LR estimate itself. The BLCA
result should be read with that caveat.

**Nothing here is an independent validation.** All four cohorts draw on one
patient universe each, reused by every architecture and every fold. Site-held-out
evaluation remains the more informative experiment, particularly for NSCLC,
where a train-only tissue-source-site diagnostic reaches 0.895 AUROC using no
image or TME input at all.

## Known limitations

- Group-level means discard within-fold patient variation. A patient-level
  paired analysis on the saved prediction CSVs (for example DeLong for
  correlated ROC curves within each fold) would replace `n_eff` of 2-4 with
  hundreds of paired patients and is the natural next step. It is more
  rigorous than this analysis, not less.
- The design-effect adjustment assumes a single exchangeable correlation among
  groups. A mixed-effects model with crossed architecture and encoder effects
  would be preferable; `statsmodels` is not available in the pinned
  environment.
- Selection objectives differ between arms: LR and the fusion variants select
  on validation patient AUROC then NLL, while adapters retain
  architecture-specific loss/error/F1 selection. This limits any strict
  algorithm ranking regardless of the test used.
- Choosing the best-performing variant post hoc and then testing it would be
  invalid. The variants tested here are pre-specified by the naming table in
  [`docs/data/variant-names.json`](../data/variant-names.json).
- HiVE-MIL entries follow the October 3 refresh policy in `AGENTS.md`.
  PathoTME-FusionVal has not been recomputed against refreshed HiVE and is
  therefore absent for that architecture; those contrasts pair 12 groups.
