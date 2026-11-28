# Significance testing for the current 16-shot comparison

Two analyses of the same 16-shot results are recorded here. They differ only in
the unit of analysis, and they reach different conclusions. The patient-level
analysis is the primary one; the group-level analysis is kept because it is the
conservative bound and because the disagreement between them is itself
informative.

Neither loads a checkpoint, replays a model, or recomputes a prediction.

## Headline

**At patient level, 66 of 144 comparisons significantly favour a PathoTME
variant over its matched Native model, and none favour Native.** The effect is
carried by NSCLC and BRCA. CRC is a true negative. BLCA is underpowered and
should not be reported as a win.

| Cohort | Patients | PathoTME-LR | PathoTME-Fusion50 | PathoTME-FusionVal |
|---|---:|---|---|---|
| NSCLC | 944 | +0.107 · 7/12 | +0.100 · **12/12** | +0.118 · 10/12 |
| BRCA | 900 | +0.199 · 11/12 | +0.121 · **12/12** | +0.195 · 12/12 |
| CRC | 536 | −0.012 · 0/12 | +0.029 · 0/12 | +0.004 · 0/12 |
| BLCA | 380 | +0.091 · 1/12 | +0.049 · 1/12 | +0.032 · 0/12 |

Median delta patient AUROC against Native, and the number of the twelve
architecture/encoder groups where the variant is significantly better after
Holm correction across all 144 comparisons.

**PathoTME-Fusion50 is the strongest variant**: significant against all twelve
native models in both NSCLC and BRCA. Its largest single effect is +0.1403
[+0.119, +0.161] against MUSE/CLIP-RN50 in NSCLC, z = 13.17, Holm p = 1.8e-37.

HiVE-MIL is excluded throughout. See [Scope](#scope).

## Why patient level, and why it changes the answer

The fold-level table reports one patient AUROC per architecture, encoder and
fold. A paired test over those numbers has at most fourteen observations per
cohort, and those observations are not independent: every group in a cohort
scores one shared held-out patient universe, under one shared set of fold
splits, and PathoTME-LR is a single fit reused by every group in a
`(cohort, fold)` cell. The whole 280-row table contains only twenty distinct LR
values. After adjusting for that dependence, the effective sample size falls to
roughly 2 in BRCA and 5 in NSCLC, and almost nothing reaches significance.

That is a limitation of the chosen unit, not evidence that the effects are
absent. The underlying data has far more information in it.

**The five folds are disjoint test partitions.** Each held-out patient is
scored in exactly one fold, and the fold sizes sum to the cohort universe:
944 NSCLC, 900 BRCA, 536 CRC, 380 BLCA. Pooling the folds therefore yields one
prediction per patient with no repetition. The analysis script verifies this
rather than assuming it, and aborts if any patient appears in two folds.

Pooling replaces fourteen correlated group means with 380 to 944 individually
scored patients, and the comparison becomes a test between two ROC curves
computed on identical cases.

## The patient-level test

**Patient scores.** Slide probabilities are averaged within `case_id`, matching
`campaigns/baselines_20260914/baseline_core.py` exactly rather than adopting a
new aggregation.

**DeLong's test for correlated ROC curves.** Each variant and its Native model
score the same patients, so the two ROC curves are correlated and an unpaired
comparison of two AUCs would misstate the standard error. DeLong's test is the
standard treatment. The implementation uses the structural-components
estimator in Sun and Xu's O(n log n) midrank form: per-case pseudo-values
`V10` over positives and `V01` over negatives give the covariance
`Cov = S10/m + S01/n`, whose off-diagonal terms carry the correlation between
the two models. With contrast `c = [1, -1]`,

```
SE = sqrt(c' Cov c)
z  = (AUC_variant - AUC_native) / SE      two-sided normal
```

**A note on simpler tests.** A plain t-test does not apply: AUROC is not a mean
over patients but a probability over positive/negative pairs, so there is no
per-patient quantity to average. A t-test over the five fold AUROCs is
available and is essentially the group-level analysis below, which is
underpowered at n = 5. The one genuinely simpler alternative that does apply is
a patient bootstrap, and it agrees — see [Cross-checks](#cross-checks).

**Multiplicity.** Holm-Bonferroni across all 144 comparisons at once
(4 cohorts x 12 groups x 3 variants), not per cohort.

## Cross-checks

The DeLong implementation is ours, so it was validated before use.

| Check | Result |
|---|---|
| AUC against `sklearn.roc_auc_score`, including ties | agrees to 1.1e-16 |
| Identical predictors must yield nothing | difference 0, p = 1.000 |
| Type-I error, 2000 null simulations at alpha = 0.05 | **0.0565** |
| Power on a known difference | +0.314, p = 1e-20 |

A patient bootstrap (2000 resamples, recomputing both AUCs on each resample)
reproduces the DeLong intervals to within 0.002 on every comparison checked,
including the CRC null, where DeLong gives p = 0.244 and the bootstrap 0.249.

| Comparison | delta AUC | DeLong 95% CI | Bootstrap 95% CI |
|---|---:|---|---|
| NSCLC muse/clip-rn50 | +0.140 | [+0.119, +0.161] | [+0.119, +0.159] |
| BRCA mscpt/clip-rn50 | +0.132 | [+0.108, +0.157] | [+0.106, +0.157] |
| NSCLC vila_mil/plip | +0.029 | [+0.016, +0.042] | [+0.015, +0.041] |
| CRC muse/plip | +0.029 | [−0.020, +0.078] | [−0.019, +0.078] |
| BLCA dyko/plip | +0.100 | [+0.040, +0.160] | [+0.040, +0.160] |

The bootstrap cannot express a p-value below 1/2000, which is why the strongest
results show 0.0005 there and 1e-39 under DeLong. For intervals the two are
interchangeable.

## Cohort detail

PathoTME-LR is one fit per cohort, so its AUROC is a single number compared
against twelve different baselines. What varies within a cohort is the native
model, not the variant.

| Cohort | Patients | Minority class | Native AUROC range (median) | LR AUROC |
|---|---:|---|---|---:|
| NSCLC | 944 | 466 (49.4%) | 0.537–0.835 (0.674) | 0.781 |
| BRCA | 900 | 176 (19.6%) | 0.542–0.749 (0.609) | 0.807 |
| CRC | 536 | 73 (13.6%) | 0.511–0.694 (0.638) | 0.626 |
| BLCA | 380 | 123 (32.4%) | 0.497–0.602 (0.548) | 0.639 |

**BRCA is the strongest cohort.** LR at 0.807 beats eleven of twelve natives,
several by +0.22 to +0.27, to Holm p = 6e-18. The exception is ViLa-MIL/PLIP at
0.749.

**NSCLC splits by encoder.** LR beats all six CLIP-RN50 natives and does not
beat the strong PLIP ones; ViLa-MIL/PLIP (0.835) and MSCPT/PLIP (0.817) exceed
LR, though not significantly. Fusion50 avoids this failure mode because it
retains the native model rather than replacing it, which is why it reaches
12/12 where LR reaches 7/12.

### CRC is a true negative

LR at 0.626 sits inside the native range (0.511–0.694), beating the weak
CLIP-RN50 baselines by about +0.12 and losing to the PLIP ones by about −0.07,
with a median difference of −0.012 and no significant comparison in either
direction. At n = 536 a real +0.10 would be detectable, so this is an absence
of effect rather than an absence of power. The quantitative panel does not
predict the CRC target. Only 73 patients carry the minority class.

This should be reported as a negative result, not as a cohort needing more
data.

### BLCA is underpowered and near chance

Eleven of twelve native baselines fall below 0.60 AUROC and six are at or below
0.55, the weakest being 0.497. LR at 0.639 is the best model in the cohort and
is still weak. Five comparisons show effects of +0.11 to +0.14, but at n = 380
only one clears Holm correction (dyko/PLIP, +0.1425, Holm p = 0.031).

Note the disagreement with the group-level analysis below, which found BLCA to
be the *only* significant cohort. At patient level it is the weakest. The
better-powered analysis should govern: BLCA is equivocal, and the paper should
not lean on it.

## Secondary: the group-level analysis

Retained as the conservative bound. Each architecture/encoder group contributes
its mean across five folds; contrasts are paired over the groups within a
cohort; the paired standard error is inflated by a design effect
`DEFF = 1 + (m - 1) * ICC` with the ICC estimated per contrast as the mean
pairwise correlation of per-fold paired differences across groups; effective n
is capped at the group count; Holm is applied within each family.

Under that treatment **one** variant-versus-Native comparison survives:
PathoTME-LR in BLCA, +0.1015 [+0.064, +0.139], Holm p = 0.012. Two further
contrasts survive in the adapter family: the adapter beats its shuffled-TME
control in NSCLC (+0.0710, Holm p = 0.026) and loses to PathoTME-LR in BLCA
(−0.0806, Holm p = 0.034).

Estimating the design effect per contrast rather than per cohort matters. An
earlier pass reused one cohort-level ICC everywhere and reported four
significant results; that was wrong. `Fusion50 − Native` carries an effective n
between 1.6 and 2.3 in every cohort, because Fusion50 is an average of Native
and the single shared LR fit, so its advantage is largely one piece of shared
information rather than fourteen.

### Reconciling the two

They are not contradictory; they answer different questions.

- Group level asks whether a variant beats Native *consistently across
  architectures*, treating architectures as the replicates. With fourteen
  heavily correlated architectures this has an effective n near 2 and almost no
  power.
- Patient level asks whether a variant's ROC curve is better *on the patients*,
  which is the question the AUROC endpoint actually poses, and has n in the
  hundreds.

The patient-level analysis is both better powered and more rigorous: paired at
the level of the observation, with a test designed for correlated ROC curves
and a validated variance estimator. It should be treated as primary. The
group-level result stands as a reminder that architecture-level consistency is
not separately established.

## The PathoTME-Adapter

The adapter is **not** in the patient-level analysis. Its predictions live in
the per-campaign run directories rather than in the baseline fusion set used
here. In the group-level analysis it fails against Native in all four cohorts
after adjustment, its best case being BLCA at Holm p = 0.30. Its one surviving
result there is that it beats its own shuffled-TME control in NSCLC, which
supports the conditioning using slide-specific measured information rather than
acting as additional capacity, and does not support a performance claim.

No claim of adapter superiority is made anywhere in this document.

## Scope

HiVE-MIL is excluded from the patient-level analysis. The fusion predictions
come from the September 14 baseline campaign, whose HiVE entries predate the
October 3 refresh that `AGENTS.md` requires, and refreshed HiVE has no
recomputed validation-selected fusion. Following that policy the eight HiVE
groups are reported as a gap rather than filled from stale records, and the
remaining six architectures are tested. The excluded groups are listed in the
JSON output.

## Limitations

- **One patient universe per cohort.** Twelve architectures agreeing on the
  same 944 patients is not twelve independent validations. Holm controls
  multiplicity across comparisons; it does not convert shared patients into
  independent evidence. No test here provides external validation.
- **NSCLC admits a site shortcut.** A train-only tissue-source-site diagnostic
  reaches 0.895 AUROC using no image or TME input. That applies to the native
  baselines as much as to PathoTME, but it limits any generalization claim from
  NSCLC.
- **Pooled cross-validation ROC mixes five models.** A group's pooled curve
  combines predictions from five fold-specific models. This is standard
  cross-validation practice, but the curve is not that of a single deployable
  model.
- **Selection objectives differ between arms.** LR and the fusion variants
  select on validation patient AUROC then NLL; adapters retain
  architecture-specific loss, error or F1 selection. This limits strict
  algorithm ranking regardless of test.
- **Discrimination is not calibration.** Mean patient NLL is worse for the real
  adapter than for the zero control in every cohort. AUROC gains do not imply
  better probabilistic prediction.
- **Variants are pre-specified.** Selecting the best-performing variant from
  these results and then testing it would be invalid. The variants tested are
  those named in [`variant-names.json`](../data/variant-names.json).

## Reproduction

Group-level, fully reproducible from this repository:

```bash
python scripts/run_significance_tests.py \
  --json docs/data/current-16shot/significance_tests.json
```

Patient-level, which additionally requires the private results store holding
the per-cell fusion predictions:

```bash
python scripts/run_patient_level_delong.py \
  --fusion-root "$PATHOTME_RESULTS/baselines_16shot_20260914_v1/fusion" \
  --json docs/data/current-16shot/patient_level_delong.json
```

Patient-level predictions are not redistributed. The output instead records a
SHA-256 for each of the 240 consumed prediction files and a single manifest
digest over all of them, so a reader with the store can verify that the inputs
match the ones analysed here.

Outputs, with every comparison's AUCs, difference, standard error, z, raw p,
Holm-adjusted p and 95% confidence interval:

- [`significance_tests.json`](../data/current-16shot/significance_tests.json) — group level
- [`patient_level_delong.json`](../data/current-16shot/patient_level_delong.json) — patient level
- [`current_fold_metrics.csv`](../data/current-16shot/current_fold_metrics.csv) — the input table
