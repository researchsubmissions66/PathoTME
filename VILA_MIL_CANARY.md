# ViLa-MIL + OpenTME signal canary

## Purpose

This first experiment asks a deliberately narrow question: does the
predeclared OpenTME panel add held-out information beyond native ViLa-MIL for
16-shot TCGA-NSCLC subtype classification? The immediate exact-match analysis
uses folds 0, 2, and 3, whose train, validation, and holdout slides all have
OpenTME measurements.

It is a **PathoTME extension**, not an upstream ViLa-MIL condition. Native
ViLa-MIL remains unchanged and supplies probabilities from its existing
validation-selected fold-0 checkpoint.

## Contract

- Native ViLa-MIL uses its CLIP-RN50 5x/10x feature bags, released NSCLC prompt
  bank, 16 prototypes, and existing PGVL-Gym checkpoint.
- OpenTME contributes a fixed 62-variable panel: 14 tissue-architecture, 28
  cell-composition, 16 spatial-interaction, and four derived TLS variables.
- No TCGA identifier, project, indication, label, or image-resolution field is
  an input.
- Percentage-valued measurements are converted to fractions, while cell
  densities and positive spatial distances receive deterministic `log1p`
  transforms. Other fractions/ratios remain unchanged.
- The TME imputer, scaler, and linear classifier fit on the 32 training slides
  only.
- Logistic regularization and the ViLa/TME log-odds weight are selected on the
  32 validation slides by balanced accuracy, then AUROC and macro-F1.
- The 208-slide fold-0 holdout is evaluated only after scripted selection, but
  it has already been inspected during exploratory development and is therefore
  reported as a diagnostic partition, not a final untouched test set.
- Folds 0, 2, and 3 contain none of the two PGVL NSCLC slides missing from
  OpenTME, so each comparison uses exactly the original
  train/validation/holdout membership. Folds 1 and 4 are excluded from this
  canary rather than silently dropping unavailable slides. A later five-fold
  experiment must regenerate both methods over the common 1,041-slide universe.

The late-fusion canary is intentionally simpler than the planned
prototype-conditioned architecture. A positive result establishes incremental
signal before adding TME-conditioned low/high prototype queries.

## Outputs

Generated features remain under
`/path/to/shared/PathoTME-data/processed`. Results are written to
`/path/to/shared/PathoTME-results`; neither belongs in source control.

The three exact-coverage folds are submitted as independent 15-minute,
single-GPU jobs so each can backfill separately. Run a submission preview with:

```bash
scripts/launch_vila_canary.sh --dry-run
```

Submit with:

```bash
scripts/launch_vila_canary.sh
```
