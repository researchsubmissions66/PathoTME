# 16-shot TME feature attribution: completed LR and fusion snapshot

This snapshot contains the completed, five-fold, patient-aggregated feature
attribution scores for the 16-shot TCGA NSCLC, BRCA, CRC, and BLCA experiments.
It covers PathoTME-LR and matched Native, PathoTME-Fusion50, and
PathoTME-FusionVal conditions. Fusion covers ViLa-MIL, MGPATH, FOCUS, MUSE,
HiVE-MIL, DyKo, and MSCPT with PLIP and CLIP-RN50. The TME-only LR model has
no visual architecture or encoder. Its matching panel and fold share one fit
across the visual conditions.

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
