# 16-shot TME feature attribution: completed LR and fusion snapshot

This snapshot contains the completed, five-fold, patient-aggregated feature
attribution scores for the 16-shot TCGA NSCLC, BRCA, CRC, and BLCA experiments.
It covers PathoTME-LR and matched Native, PathoTME-Fusion50, and
PathoTME-FusionVal conditions. Fusion covers ViLa-MIL, MGPATH, FOCUS, MUSE,
HiVE-MIL, DyKo, and MSCPT with PLIP and CLIP-RN50. The TME-only LR model has
no visual architecture or encoder. Its matching panel and fold share one fit
across the visual conditions.

## Find a result

Start with the top-10 file for a cancer: [NSCLC](top10_nsclc.csv),
[BRCA](top10_brca.csv), [CRC](top10_crc.csv), or [BLCA](top10_blca.csv).
Each has 580 rows: the ten largest individual-feature and ten largest group
scores for every LR or fusion variant/architecture/encoder combination. Filter
`variant`, `image_architecture`, `image_encoder`, and `level` together. The
`rank_by_mean_absolute` column ranks within that exact combination; its value
does not compare different cancers or models. Columns `fold_0_absolute_pp`
through `fold_4_absolute_pp` and their matching `fold_*_rank` show the five
folds on the same row. A blank fold rank means that fold's TME score is zero.
`none` in the two image columns means the TME-only LR model. For example,
BRCA ViLa-MIL/PLIP validation fusion has `variant=PathoTME-FusionVal`,
`image_architecture=vila_mil`, and `image_encoder=plip`. The top-10 files use
the named positive class
(LUSC, ILC, MUCINOUS, or PAPILLARY); binary class-0 absolute scores are the
same. For every feature and both classes, use [feature_rankings.csv](feature_rankings.csv).
For every fold, use [fold_feature_scores.csv](fold_feature_scores.csv).

| Variant | Prediction inputs | What its TME feature score measures | Available here |
|---|---|---|---|
| PathoTME-LR | TME only | Change in a saved logistic-regression prediction | All four cancers, five folds each |
| Native | Image only | Exactly zero by construction; omitted from the top-10 files | All seven architectures and both encoders |
| PathoTME-Fusion50 | Image and TME-LR predictions | `0.5 ×` the matched LR score; the image prediction stays fixed | All seven architectures and both encoders |
| PathoTME-FusionVal | Image and TME-LR predictions | Saved validation-selected TME weight `×` the matched LR score; the image prediction stays fixed | All seven architectures and both encoders |
| Learned PathoTME adapter | Image and TME in a learned model | Full model rerun after changing a TME measurement | In progress; no cohort-level adapter scores in this snapshot |

The fusion rows include an image in the **prediction**, but their TME feature
scores are scaled LR scores. They do not measure an image-dependent TME feature
effect. Only the learned-adapter attribution can answer that question. Keep
results for different architectures and encoders separate; a larger score does
not establish better predictive accuracy.

As a quick orientation, the largest **PathoTME-LR** group and individual
measurement in each cancer are below. Values are five-fold means of patient
mean absolute probability changes, in percentage points (pp). They are a
TME-only reference, not a ranking for the learned image-aware adapters.

| Cancer | Largest LR group (pp; top-10 folds) | Largest LR measurement (pp; top-10 folds) |
|---|---|---|
| NSCLC | Spatial fibroblasts (4.76; 5/5) | Fibroblast-to-carcinoma mean distance, radius 40 (2.29; 4/5) |
| BRCA | Spatial lymphocytes (2.90; 4/5) | Lymphocyte-to-carcinoma mean distance, radius 20 (1.24; 4/5) |
| CRC | Tumor-core composition (5.89; 4/5) | Plasma-cell density in the inner invasive margin (3.06; 2/5) |
| BLCA | Spatial plasma cells (4.57; 3/5) | Fibroblast-to-carcinoma mean distance, radius 20 (2.94; 3/5) |

BRCA illustrates why the folds matter. The two leading LR groups have these
fold-level mean absolute scores (pp); each fold has 180 held-out patients.

| BRCA group | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Five-fold mean |
|---|---:|---:|---:|---:|---:|---:|
| Spatial lymphocytes | 0.40 | 6.16 | 0.03 | 6.01 | 1.89 | 2.90 |
| Macrophage cell measurements | 0.62 | 3.33 | 0.59 | 6.54 | 2.36 | 2.69 |

The macrophage group ranks first among BRCA groups in four folds, while
spatial lymphocytes have the larger five-fold mean. Neither establishes an
image-aware BRCA winner.

## Files and calculation

The files were exported on 2026-10-05 UTC from saved models, preprocessing,
predictions, and validation-selected fusion weights. No classifier was refit
and no feature panel was selected for this export. The underlying [scoring
implementation](../../../campaigns/feature_attribution_20260915/score_core.py)
replaces one standardized measurement or biological group with the saved
training-fold reference, then reports
`100 × (observed class probability − reference class probability)` in
probability percentage points. Zero in this transformed space means the
training-fold mean; it does not mean biological absence. Group scores replace
their constituent measurements jointly and are not sums of feature scores.

| File | Contents |
|---|---|
| `top10_{nsclc,brca,crc,blca}.csv` | Per-cancer browsing views derived from `feature_rankings.csv`; class 1 and nonzero TME variants only. |
| `feature_rankings.csv` | Mean absolute and signed patient scores, fold SD, ranks, top-10 frequency, and coverage for each feature/group and class. |
| `fold_feature_scores.csv` | The same sensitivity summarized separately within each fold. |
| `lr_coefficients.csv` | Saved LR coefficient for each standardized feature and fold. These are class-1 log-odds coefficients, not probability-point scores. |
| `lr_coefficient_stability.csv` | Descriptive cross-fold coefficient direction and spread. |
| `summary.json` | Export counts and SHA-256 digests of the four CSV files. |

The ranking CSV contains 27,004 rows and the fold CSV contains 135,020 rows.
There are 20 TME-only LR fits and 280 matched native/fusion evaluations. Every
ranking row has all five folds. BRCA uses its historical morph64 panel; the
other cohorts use core62. Cohorts, panels, classes, architectures, encoders,
and shot counts remain explicit in the files.

The top-10 views are sorted by `mean_absolute_pp` within cancer, variant,
architecture, encoder, and level. `fold_sd_absolute_pp` describes variability
over the five fold means; `mean_rank` averages within-fold ranks; and
`top10_fold_fraction` is the fraction of folds with rank 10 or better. These
views are derived convenience files, not additional experiments. The full
rankings and original export hashes remain the authoritative snapshot.

Signed slide deltas are averaged within a patient before calculating their
absolute magnitude. Patient magnitudes are averaged within each fold, and
folds are weighted equally. A positive signed score supports the named class;
a negative score opposes it. Fusion50 and FusionVal feature scores are exactly
their fixed TME weight times the matching LR probability-change scores, with
the native prediction held fixed. Native has zero TME feature dependence by
construction. These are model sensitivities, not causal effects, significance
tests, or accuracy changes. Correlated measurements can make isolated reference
replacement biologically unrealistic.

No slide or patient identifiers, raw TME values, feature bags, checkpoints, or
individual score arrays are included. Learned-adapter exports remain in
progress; the available one-patient neural example is excluded from this
cohort-level snapshot.
