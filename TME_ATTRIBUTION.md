# TME feature attribution and thumbnail overlays

PathoTME can explain a saved actual, zero or shuffled adapter checkpoint for
ViLa, MGPATH, FOCUS, MUSE, HiVE-MIL, DyKo and MSCPT, on NSCLC/core62 and BRCA/morph64,
with PLIP or CLIP-RN50. Existing training and checkpoints are unchanged.

## Unified feature scores for the named variants

The [new feature-score workflow](campaigns/feature_attribution_20260915/README.md)
adds exact PathoTME-LR, PathoTME-Fusion50 and PathoTME-FusionVal feature/group
probability scores, structural Native/PathoTME-Adapter-Zero scores, and a
four-cohort neural exporter for PathoTME-Adapter and PathoTME-Adapter-Shuffled.
It provides patient/fold rankings, fold SD, top-10 frequency, LR coefficient
direction, an offline feature heatmap, and standalone figures. Fusion scores
are exactly the frozen alpha times LR scores; they are not independent
feature discoveries. Generated coverage is recorded in each private report.

The commands below describe the legacy NSCLC/BRCA exporter. Use the new
workflow for LR/fusion scores or CRC/BLCA adapters. Both use the same feature
reference and probability-point score, while retaining exact panel names and
explicit partial-patient/fold coverage. Completed example exports do not imply
that all neural checkpoints or full test cohorts have been scored.

## Feature and biological-group scores

For class `c`, the score is `100 × [p(c | observed TME) − p(c | replaced TME)]`.
Each feature, or all measurements in one of the existing 16 biological groups,
is replaced at the output of the saved fold standardizer with zero. This is
the **training-fold mean in transformed feature space**, not biological absence.
The reference statistics are loaded from the adapter; they are never refitted.
Positive scores support the selected class; negative scores oppose it.

This follows [feature ablation](https://captum.ai/api/feature_ablation.html).
It does not require gradients, so the complete downstream hard selection,
expert routing and internal graph computation can respond to each intervention.
The score is not SHAP, an additive decomposition, biological causality, or a
measurement of model accuracy. Joint group scores are computed jointly and
are not sums of their individual feature scores. Correlated/compositional
features make isolated reference replacements imperfect biological examples.
Use the biological-group view as the primary interpretation, and individual
features as a finer sensitivity analysis. These post-hoc scores do not provide
an a priori justification for the panel selection.

## Generate a feature heatmap

Use the PGVL environment, with `PGVL_ROOT` pointing to the sibling PGVL-Gym
checkout if necessary. `RUN` is one completed fold directory and `OUTPUT` is
a new private results directory outside either source repository.

```bash
python scripts/generate_tme_attribution.py --list-providers
python scripts/generate_tme_attribution.py \
  --completion "$RUN/fold_complete.json" --slide-id "$SLIDE_ID" \
  --granularity both --output "$OUTPUT"
```

The command plans by default. Add `--execute` to reconstruct the saved model
and explain the selected held-out slide. CPU is the default; `--device cuda:0`
uses an already available GPU without submitting jobs. Default execution is
bounded to one slide. Repeat `--slide-id` or explicitly use `--all-test-slides`
to expand it. Existing files are never overwritten. Prediction reproduction
is checked against the saved CSV before scoring. Missing/incompatible assets
raise errors rather than selecting another checkpoint, encoder or slide.

Outputs are `heatmap.html`, `attribution.json`, `slide_scores.csv`,
`patient_scores.csv` and `ranking.csv`. The interactive heatmap has class,
feature/group and signed/ranking views. Patient ranking first averages slide
probability deltas within each patient, then averages absolute patient scores
within a fold and equally across available folds. Patients with unexplained
slides are excluded; subset coverage remains visible. Classes are separate.
Do not interpret a subset ranking as a complete cohort result.

```bash
python scripts/summarize_tme_attribution.py \
  --inputs "$FOLD0/attribution.json" "$FOLD1/attribution.json" \
  --output "$SUMMARY_OUTPUT"
```

Only matching model/cohort/encoder/shot/arm/panel exports can be combined.
Duplicate slides, patient overlap across test folds and checkpoint mismatches
are rejected. The existing trained zero-TME control is a separate checkpoint;
zeroing TME in the actual checkpoint here is an inference intervention.
For shuffled checkpoints, attribution explains the donor TME row actually
consumed by that checkpoint, not the recipient's observed biology.

## Heatmaps over a thumbnail

Add a source WSI to make a registered thumbnail overlay for the same slide:

```bash
python scripts/generate_tme_attribution.py \
  --completion "$RUN/fold_complete.json" --slide-id "$SLIDE_ID" \
  --output "$OUTPUT" --execute --wsi "$WSI" \
  --spatial-groups 16 --spatial-reference all
```

Alternatively pass `--thumbnail "$THUMBNAIL" --level0-size WIDTH HEIGHT`.
This must be the uncropped, unrotated full-slide thumbnail in the same level-0
coordinate frame; the dimensions are the original WSI dimensions, not the
thumbnail dimensions. Aspect ratio and coordinate bounds are checked. A real
background is required; a blank coordinate canvas is never called a thumbnail.

Every architecture uses the same prediction-based spatial score: replace one
spatial region's consumed patch features with the fixed original bag mean,
measure the class-probability change, and repeat with reference TME. The viewer
shows the observed-TME region effect, the reference-TME region effect, or their
difference. The difference measures dependence of visual-region sensitivity on
the selected TME intervention. It is not native attention or cell localization.

The default reference replaces all TME measurements. `--spatial-reference` can
instead name one exact feature or biological group printed by the plan, e.g.
`high/spatial_lymphocytes`. This is one selected TME reference per export, not
an unbounded cross product of every feature with every spatial region.

Regions use exact consumed feature coordinates and selected-row identities.
Supplied graph edges and hierarchy padding are preserved; synthetic padded
patches have no displayed location. Each region has one joint score repeated
on its patches: do not sum the displayed patch scores. Unmeasured tissue stays
unchanged; there is no interpolation of unobserved biology.

The `spatial/` directory contains the original thumbnail, an annotated PNG,
exact JSON scores and a standalone interactive `heatmap.html`. It offers class,
scale, observed/reference/difference selection, opacity, hover scores and PNG
download. The overview PNG shows class 1 and the first input scale explicitly.

## Validation and provenance

The exporter verifies completion/launch/config identities, both checkpoint
hashes, exact test prediction membership, TME/donor/split hashes, selected feature
file inventory and the reproduced slide probabilities. It restores intervention
hooks after errors. Feature attribution also restores model modes and RNG.
No optimizer, standardizer fitting, training, GPU submission or runtime download
is performed. Existing extraction-weight provenance limitations remain.

Focused CPU checks cover signed/joint scores, zero dependency, state restoration,
native panel-group correspondence, patient weighting, subset/duplicate rejection,
thumbnail registration/background, spatial TME/visual-region interaction and
safe rendering. Synthetic checks are not
real-cohort interpretation or validation of all six native encoders/checkpoints.

A saved MUSE/PLIP NSCLC fold-0 slide was also checked on the login CPU:
recorded probabilities reproduced, all 62 feature and 16 group scores exported,
and 16 spatial regions rendered over its existing full-slide thumbnail with
dimensions from the matching extraction state. This small-bag slide was chosen
for a bounded compatibility check, not because of its outcome or attribution.
It is one-slide evidence, not a cohort ranking or an all-architecture live audit.

Raw TME tables are never copied into the source repository or exported HTML.
Exact IDs and standardized values occur only in private JSON/CSV artifacts.
HTML uses slide pseudonyms and excludes private paths/feature measurements.
Keep outputs in private results storage; no results are published to the website.
