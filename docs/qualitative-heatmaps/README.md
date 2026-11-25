# Four-feature OpenTME atlases on held-out WSI thumbnails

Each cohort atlas shows **four measured TME features on the same two real TCGA slides**. Columns are features; rows are the first lexicographic held-out fold-0 slide of each class. This slide rule was fixed independently of prediction and segmentation appearance. OpenTME supplies the registered WSI and spatial annotation thumbnails. The saved 16-shot TME-only logistic-regression (TME-LR) checkpoint supplies the coefficient and training-fold standardizer shown under each panel.

| Cohort | Held-out slide classes | Atlas |
|---|---|---|
| BRCA | ILC, IDC | [Four-feature atlas](brca_tme_feature_atlas.png) |
| NSCLC | LUAD, LUSC | [Four-feature atlas](nsclc_tme_feature_atlas.png) |
| CRC | ADENO_NOS, MUCINOUS | [Four-feature atlas](crc_tme_feature_atlas.png) |
| BLCA | NON_PAPILLARY, PAPILLARY | [Four-feature atlas](blca_tme_feature_atlas.png) |

The same four whole-tumor-region area features are shown for every cohort:

| Feature in the saved TME panel | Spatial layer used | Absolute-coefficient rank: BRCA / NSCLC / CRC / BLCA |
|---|---|---|
| `RELATIVE_AREA_CARCINOMA_IN_TUMOR_CORE` | Carcinoma tissue ∩ tumor core | 58 / 46 / 20 / 62 |
| `RELATIVE_AREA_INNER_INVASIVE_MARGIN` | Inner invasive margin | 60 / 25 / 4 / 47 |
| `RELATIVE_AREA_STROMA_IN_INNER_INVASIVE_MARGIN` | Stroma tissue ∩ inner invasive margin | 63 / 10 / 5 / 9 |
| `RELATIVE_AREA_OUTER_INVASIVE_MARGIN` | Outer invasive margin | 23 / 40 / 21 / 2 |

These features were chosen because their corresponding region and tissue labels can be located in **all eight OpenTME thumbnails**. They are illustrative tissue compartments, **not the four most influential LR features**. In particular, BRCA's saved LR gives three of these four features near-zero coefficients. Several higher-ranked cell-density and cell-neighborhood measurements have only slide-level values in the available OpenTME tables; assigning them pixel locations would be unsupported. The full ranks and exact values are in [figure metadata](figure_metadata.json).

## Reading the panels

Warm color is a **local-occupancy view of the matching OpenTME segmentation**, overlaid on its registered WSI thumbnail. The renderer extracts the exact thumbnail palette label (or intersection of two labels), removes the embedded legend, applies a Gaussian blur with radius `max(6 pixels, 2.8% of the shorter side)`, and clips color to annotated tissue. The 99.5th percentile sets the color scale separately **for each panel**. Yellow in two different panels therefore does not imply equal area, equal model weight, or equal effect. The numeric area fraction under a panel is the original OpenTME slide-level measurement, checked against the exact processed feature table used by the model.

The LR coefficient `β` is the positive-class log-odds change per one training-fold standard deviation of the feature, with other features fixed. The displayed logit term is `β × (measured area − training mean) / training SD`. The positive class is ILC (BRCA), LUSC (NSCLC), MUCINOUS (CRC), or PAPILLARY (BLCA). A visible warm region can have a negative term when the slide's area fraction is below the training mean. **The heatmap is feature localization, not pixel-level LR attention or a model saliency map.** Visible pen marks on some original WSIs are slide artifacts, not heatmap values.

## Reproduce with the actual data

Use the registered OpenTME release, the saved fold-0 TME-LR models, and the exact processed panel CSV for each cohort. If a model's recorded panel path still exists, its `--feature-table` override can be omitted. The renderer verifies held-out membership and every plotted measurement against the original OpenTME CSV.

```bash
python -m pip install Pillow
python scripts/render_tme_feature_heatmaps.py \
  --opentme-root "$OPENTME_ROOT" \
  --models-root "$TME_LR_MODELS_ROOT" \
  --feature-table "brca=$BRCA_PANEL_CSV" \
  --feature-table "nsclc=$NSCLC_PANEL_CSV" \
  --feature-table "crc=$CRC_PANEL_CSV" \
  --feature-table "blca=$BLCA_PANEL_CSV" \
  --output-dir "$FIGURE_OUTPUT_DIR"
```

The [GDC source manifest](source_wsis.tsv) records original slide identities and checksums. [Figure metadata](figure_metadata.json) records the saved model IDs, measurement and coefficient values, and hashes of the OpenTME thumbnail inputs. The LR models came from the [matched baseline campaign](../../campaigns/baselines_20260914/). The gated OpenTME release, full WSIs, and raw annotation layers are not redistributed with these derived publication figures.

**Data attribution:** Galama et al., [*OpenTME: An Open Dataset of AI-Powered H&E Tumor Microenvironment Profiles from TCGA*](https://arxiv.org/abs/2604.12075), 2026; Aignostics and OpenTME. TCGA/GDC provided the source whole-slide images.
