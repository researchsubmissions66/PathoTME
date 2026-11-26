# Qualitative patch–TME analysis

`scripts/export_tme_patch_examples.py` explains one real held-out BRCA slide
from a completed PathoTME-Adapter · MUSE · CLIP-RN50 checkpoint. It uses the
saved fold preprocessing, the exact coordinate-bearing feature bag, and the
original WSI when tissue crops are requested. It does not train a model or
generate synthetic patches.

## What a patch score means

For a selected class \(c\), measured TME feature \(j\), patch bag \(X\), and
saved standardized TME vector \(z\), let \(X_{-i}\) replace consumed patch \(i\)
with the fixed mean vector of the original bag. Let \(z_{j=0}\) set feature
\(j\) to zero **after** the saved training-fold standardizer. The reported
interaction is

\[
I_{i,j,c} = 100\{[p_c(X,z)-p_c(X_{-i},z)]
                 -[p_c(X,z_{j=0})-p_c(X_{-i},z_{j=0})]\}.
\]

A positive value means that patch removal reduces the selected class
probability more under the observed TME than under the feature reference.
The standardized zero is the training-fold reference, not biological absence.
These are finite-difference **model sensitivities**. They do not establish
that the measured phenotype occurs inside the patch, identify a cell type,
or demonstrate a causal mechanism. Patch scores need not add up to the
slide-level feature effect.

MUSE attention screens candidates, using the union of the top positive-weight
patches under observed and reference TME. The exporter then computes every
reported score from full prediction reruns. `patch_scores.csv` ranks **only
the screened candidates**; `analysis.json` records both the candidate count
and the total consumed patch count. Ties are broken by consumed patch index.

## Reproduce an export

Set `PATHOTME_RESULTS` to the private directory containing the saved results.
The selected fold must be completed and its input hashes must still match.

```bash
python scripts/export_tme_patch_examples.py \
  --completion "$PATHOTME_RESULTS/muse_tcga_16shot_20260912_v1/runs/brca/clip-rn50/fold0/fold_complete.json" \
  --feature-name RELATIVE_AREA_OUTER_INVASIVE_MARGIN \
  --candidate-per-condition 32 \
  --selection-note 'Exploratory opposing feature; not a top supporting feature' \
  --output "$PATHOTME_RESULTS/interpretability/qualitative_muse_brca_fold0_margin"
```

Without `--slide-id`, the slide is the first lexicographic held-out slide in
that fold. Without `--class-name`, the target is its saved true class. These
rules are recorded in `analysis.json`. A feature used for a paper figure
should be selected by an explicitly reported rule; the example above is an
exploratory, post hoc feature choice. The candidate screen can miss a large
interaction on a low-attention patch, so the montage must not be described as
a global ranking of all patches.

To export unaltered tissue crops and a montage, add `--wsi "$BRCA_WSI"` where
`BRCA_WSI` points to the original slide whose filename contains the exact
analyzed slide ID. The WSI dimensions must match the HDF5 level-0 geometry.
The output folder contains `analysis.json`, `patch_scores.csv`, and, when a
WSI is supplied, `crops/` with individual PNG crops and `montage.png`. Keep
WSIs and result exports outside this source repository.

For spatial context, `scripts/render_tme_patch_overview.py --analysis
"$EXPORT/analysis.json" --wsi "$BRCA_WSI" --output "$EXPORT/overview.png"`
marks the highest and lowest scored crops on a real full-slide thumbnail. It
shows only scored candidates and does not interpolate values into a heatmap.

To compare two prespecified features on the same slide, run
`scripts/compare_tme_patch_examples.py` with `--first` and `--second` pointing
to their respective `analysis.json` files. The joined CSV preserves blank
scores for patches screened under only one feature; blanks are not zero.

## Top-feature TME-Adapter attention maps

The [four-cohort TME-Adapter figure set](docs/adapter-feature-attention/README.md)
uses the saved MUSE/CLIP-RN50 fold-0 models. On each of eight real held-out
WSIs, it ranks the individual measured TME inputs by their positive effect
on the **predicted-class** probability. The top three features each get a
full-thumbnail map of how that feature changes MUSE's class-query patch
attention relative to the saved training-fold reference. The script is
[`scripts/render_adapter_feature_attention.py`](scripts/render_adapter_feature_attention.py).
The maps are model attention shifts at exact consumed patch coordinates;
they do not locate the named cells or establish patch-level probability
contributions. The two fixed NSCLC slides are misclassified and are labeled
accordingly in the figures.

## Feature-localized WSI heatmaps

The [four-cohort OpenTME atlas set](docs/qualitative-heatmaps/README.md) overlays four
real OpenTME whole-tumor-region area features on each held-out WSI thumbnail:
carcinoma in tumor core, inner invasive margin, stroma in that margin, and
outer invasive margin. A saved fold-0 TME-LR model provides the coefficient
and slide-level logit term beneath each panel. Color shows smoothed local
occupancy of the annotated compartment. It does not show MUSE attention,
pixel-level LR weights, or a cell-density estimate. The atlas index gives
the exact measurements and reproduction command for all four cohorts.

## Separate MUSE model-sensitivity maps

`scripts/generate_tme_attribution.py` can evaluate deterministic spatial
groups on the same saved checkpoint and real WSI. It exports an interactive
`spatial/heatmap.html`, the original `spatial/thumbnail.png`, and a static
interaction `spatial/overlay.png`. For BRCA MUSE/CLIP-RN50, use the same
completed fold and slide as above with `--granularity groups`,
`--spatial-reference` set to the exact feature name, `--spatial-groups 64`,
`--wsi "$BRCA_WSI"`, and `--execute`. Then render a single full-slide thumbnail
heatmap from the saved real patch coordinates and interaction scores:

```bash
python scripts/render_tme_thumbnail_heatmap.py \
  --spatial "$HEATMAP_EXPORT/spatial/spatial_scores.json" \
  --class-name ILC \
  --output "$HEATMAP_EXPORT/spatial/thumbnail_heatmap.png"
```

The heatmap shows the difference in each spatial group's visual removal
effect under measured TME and the selected feature's training-mean reference.
Each group receives **one joint score**, repeated on its member patches.
Every plotted tile comes from a real feature-bag patch; no values are filled
into unmeasured tissue. A patch-weighted 95th-percentile color scale reveals
the full thumbnail while the numeric legend and maximum score retain the
effect magnitude. The map does not claim individual patch scores or locate
the named TME measurement.
For CRC and BLCA, the loader reconstructs the exact saved shared-core62
campaign wrapper; the BRCA model uses morph64 and NSCLC uses core62.

For a direct but sparse map of the individual patch interactions already
computed by `export_tme_patch_examples.py`, use
`scripts/render_tme_patch_heatmap.py --analysis "$EXPORT/analysis.json"
--wsi "$BRCA_WSI" --output "$EXPORT/patch_heatmap.png"`. It colors only the
attention-screened patches and leaves every unscored location unchanged.

## Figure use

For a model-interpretability caption, use the
[TME-Adapter figure caption](docs/adapter-feature-attention/README.md) and
report both the probability change and attention redistribution. These are
different measurements; neither localizes the slide-level biological input.

Suggested caption for the feature-localization set: “Held-out OpenTME WSI
thumbnails for BRCA, NSCLC, CRC, and BLCA. Four columns localize carcinoma in
tumor core, inner invasive margin, stroma within that margin, and outer
invasive margin on the same slides. The TME-LR coefficient and standardized
logit term are slide-level quantities; warm color is normalized within each
panel and is not pixel-level model attention.”

A compact supplement panel can show three highest and three lowest
interactions with their signed percentage-point scores, followed by the
slide-level class probability with measured and reference TME. State the
number of screened patches out of all consumed patches. For an overview,
`scripts/generate_tme_attribution.py` already exports spatial region effects
on a registered real WSI thumbnail; its **region** scores are not individual
patch scores. A second panel can show a preselected disagreement case, or
how the MUSE group attention and expert routing change under the same TME
reference, with the same sensitivity caveat.

Suggested caption: “Held-out BRCA MUSE/CLIP-RN50 example. Patches were
screened by model attention under measured and training-mean-reference TME,
then ranked by the difference in their class-probability removal effects.
Scores are reference-dependent model interactions in percentage points; the
numeric TME feature is measured at slide level and is not a patch annotation.”
