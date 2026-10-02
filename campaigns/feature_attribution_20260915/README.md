# PathoTME feature attribution

This isolated post-hoc workflow supplies a score for every numeric TME feature
and each of the 16 biological groups. It reuses saved checkpoints, coefficients,
training-fold preprocessing, patient splits and frozen fusion weights. It never
fits a classifier or selects a feature panel.

## Score and interpretation

The common score is `100 * (p_class_observed - p_class_reference)` in probability
percentage points. A reference replaces one feature, or a whole biological
group jointly, with zero after the saved fold standardizer. This means the
transformed training mean, not biological absence. Positive scores support the
selected class; negative scores oppose it. Individual probability scores are
not additive and group scores are not the sum of their feature scores.

Patient scores average signed scores over all that patient's test slides.
Rankings average absolute patient scores within a fold, then weight the
available folds equally. Reports retain exact feature names, class names,
cohort, panel, architecture, encoder, shot count, fold count and patient count.
They include signed means, absolute means, fold SD, ranks and top-10 fold
frequency. Fold SD is descriptive, not a confidence interval. Zero scores have
no arbitrary top-10 rank.

| Variant | Derivation |
|---|---|
| Native | Zero TME dependence by construction; native probabilities stay fixed. |
| PathoTME-LR | Exact saved logistic forward after each feature/group reference replacement. |
| PathoTME-Fusion50 | Exactly 0.5 times the matching LR probability-change scores. |
| PathoTME-FusionVal | Exactly the saved validation-selected alpha times LR scores, including alpha=0 and alpha=1. |
| PathoTME-Adapter | Replay the saved model with feature/group interventions. |
| PathoTME-Adapter-Zero | Structural zero: the saved zero-input arm discards numeric TME before conditioning. Labeled as a derivation, not a new forward measurement. |
| PathoTME-Adapter-Shuffled | Replay the checkpoint using the donor TME it actually consumed, not the recipient's observed biology. |

The LR coefficient file provides a separate directional quantity: class-1
log-odds per unit of a saved standardized feature. Exact per-slide additive
log-odds contributions are retained in private score arrays, separate from the
nonadditive probability scores. Fusion rankings inherit LR dependence; they
are not independent feature discoveries.

## CPU LR/fusion scores

Use the baseline manifest and a fresh directory in private results storage:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python campaigns/feature_attribution_20260915/run_linear.py \
  --manifest "$BASELINE_MANIFEST" --output "$ATTRIBUTION_OUTPUT"
```

This requires only NumPy and the dependency-light project feature definitions.
It checks source/input hashes and split memberships, replays saved test
probabilities with the frozen scaler/coefficients, and verifies exact fusion
predictions. Missing fusion evaluations stay explicitly pending. Arrays are
stored once per LR cohort/fold; fusion artifacts reference those arrays with
their fixed score multiplier rather than duplicating patient data.

Outputs include `feature_rankings.csv`, `fold_feature_scores.csv`,
`lr_coefficient_stability.csv`, `feature_heatmap.html`, `REPORT.md` and compact
private `scores.npz` arrays. For a selected fusion artifact, multiply only
`score_pp` and `patient_score_pp` by its recorded multiplier, not LR log-odds
contributions. Generate standalone PNG/PDF/SVG plots using `plot_scores.py
--output "$ATTRIBUTION_OUTPUT"` in the research Python environment.

## Neural feature scores

Use the research Python environment. The command plans by default; add
`--execute` for inference. It covers the exact NSCLC/core62 and BRCA/morph64
ports and the CRC/BLCA common62 runtime for all seven architectures. Only data
loader concurrency is changed; constructors receive the original saved config.

```bash
python campaigns/feature_attribution_20260915/run_adapter.py \
  --completion "$FOLD/fold_complete.json" --launch "$LAUNCH" \
  --arm actual --all-test-patients --output "$NEURAL_OUTPUT" --execute
```

Use `--arm shuffled` for the donor control. A bounded initial sample is
available through `--max-patients N`: it selects a fixed hash order independent
of labels, predictions and feature values, then includes every test slide of
each selected patient. `--patients-file` accepts an explicit JSON list. Subset
coverage never becomes a full-cohort claim. CPU and two threads are the default;
`--device cuda:0` uses an already allocated GPU and submits no Slurm jobs.

`export_zero.py --manifest "$BASELINE_MANIFEST" --output "$ZERO_OUTPUT"`
exports the explicitly labeled structural zero control. `merge_reports.py`
combines linear, zero and completed neural directories into a fresh common
viewer, preserving distinct variants and coverage. A neural directory without
its completed record and verified score array is rejected.

The legacy thumbnail exporter remains available separately. These feature
scores do not themselves localize cell measurements in a WSI thumbnail. LR
operates on slide-level measurements and has no patch pathway to visualize.

## Research boundaries

- Interpret these as model sensitivities, not causal biomarkers, significance
  tests, or an a priori justification of the hand-defined panel.
- Correlated and compositional measurements can make isolated mean replacement
  unrealistic; inspect joint biological groups alongside individual features.
- Historical BRCA remains morph64. Do not pool its numeric panel with core62.
- All requested feature names are retained, including small, zero or unstable
  effects. Ranking is descriptive and does not tune the model on test data.
- Keep patient score arrays and JSON/CSV metadata private. HTML excludes patient
  identifiers, standardized feature values and private paths. No results or raw
  TME tables are added to the public website or source repository.

Focused numerical checks are in `check_scores.py`. Completed real-cohort
examples and their coverage are recorded in the private result summary; a
supported constructor is not proof of a completed cohort attribution study.
Original source-bound training, baseline and legacy attribution modules remain
unchanged. Earlier revisions of this new exporter are retained under
`source_history/` for its completed output provenance.
