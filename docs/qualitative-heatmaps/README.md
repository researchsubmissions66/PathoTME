# Held-out TME feature heatmaps on whole-slide thumbnails

These eight figures place **real OpenTME feature-defined regions** on registered TCGA whole-slide thumbnails. They cover one held-out slide per class in BRCA, NSCLC, CRC, and BLCA, using the lexicographically first slide ID for each class in fold 0. The slide rule was fixed independently of prediction correctness and the visual appearance of the segmentation. The saved **16-shot TME-only logistic-regression (TME-LR)** model supplies each feature's coefficient; OpenTME supplies the spatial annotation and the measured slide-level value.

| Cohort | Held-out class | Slide | Plotted feature | Figure | LR rank | Measured area | LR logit term |
|---|---|---|---|---|---:|---:|---:|
| BRCA | ILC | TCGA-4H-AAAK-01Z-00-DX1 | Outer invasive margin | [PNG](brca_ilc_tme_feature.png) | 23/64 | 0.0429 | −0.0056 |
| BRCA | IDC | TCGA-A1-A0SD-01Z-00-DX1 | Outer invasive margin | [PNG](brca_idc_tme_feature.png) | 23/64 | 0.1304 | +0.0017 |
| NSCLC | LUAD | TCGA-05-4382-01Z-00-DX1 | Stroma in inner invasive margin | [PNG](nsclc_luad_tme_feature.png) | 10/62 | 0.0084 | +0.0037 |
| NSCLC | LUSC | TCGA-18-3417-01Z-00-DX1 | Stroma in inner invasive margin | [PNG](nsclc_lusc_tme_feature.png) | 10/62 | 0.0044 | −0.0013 |
| CRC | ADENO_NOS | TCGA-5M-AAT6-01Z-00-DX1 | Inner invasive margin | [PNG](crc_adeno_nos_tme_feature.png) | 4/62 | 0.1375 | −0.0047 |
| CRC | MUCINOUS | TCGA-AA-3852-01Z-00-DX1 | Inner invasive margin | [PNG](crc_mucinous_tme_feature.png) | 4/62 | 0.4218 | +0.0065 |
| BLCA | NON_PAPILLARY | TCGA-2F-A9KQ-01Z-00-DX1 | Outer invasive margin | [PNG](blca_non_papillary_tme_feature.png) | 2/62 | 0.1930 | +0.2657 |
| BLCA | PAPILLARY | TCGA-4Z-AA81-01Z-00-DX1 | Outer invasive margin | [PNG](blca_papillary_tme_feature.png) | 2/62 | 0.0643 | −0.0948 |

The rank is by absolute standardized TME-LR coefficient across the saved model's feature panel. We explored that ranking and chose one whole-tumor-region area feature per cohort whose spatial layer is visible at thumbnail resolution. This is an **exploratory figure selection**, not a confirmatory claim that these are the most important features across all cohorts. In BRCA, higher-ranked epithelial-tissue-in-margin measurements are nonzero in the CSV but too sparse to resolve in these thumbnails.

## What the color means

Warm color shows **local occupancy of the feature's real OpenTME segmentation**. For BRCA and BLCA, the mask is OpenTME's outer invasive margin; for CRC, its inner invasive margin. For NSCLC, the mask is the intersection of OpenTME stroma and inner invasive margin annotations. The script removes the embedded legend, smooths the binary mask with a Gaussian radius of `max(6 pixels, 2.8% of the shorter thumbnail side)`, and clips color to annotated tissue. The color scale is normalized to the 99.5th percentile **separately for each slide**, so color intensity is spatial context within a slide and is not comparable between figures. No cell coordinates, raw WSIs, raw OpenTME thumbnail layers, or full OpenTME dataset are distributed here.

The TME-LR coefficient is the change in the **positive-class log-odds per one training-fold standard deviation** of that feature, holding the other features fixed. The displayed slide-level logit term is `β × (measured area − training mean) / training SD`. The positive class is ILC for BRCA, LUSC for NSCLC, MUCINOUS for CRC, and PAPILLARY for BLCA. A negative logit term can accompany a visible warm region when the measured area is below the training mean. **TME-LR has no pixel attention weights:** the color localizes a measured feature, while the number describes its contribution to the slide-level linear model. Smoothing is for visualization and does not create new measurements. Visible pen marks on some source thumbnails are slide artifacts, not heatmap values.

## Reproduce from the actual inputs

Install Pillow and point the script to the registered OpenTME release, the saved fold-0 TME-LR baseline models, and the exact processed feature tables used by those models. If a model's recorded table path still exists, the `--feature-table` override for that cohort can be omitted. The script checks each slide against the model's held-out membership and checks each plotted panel value against the original OpenTME CSV.

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

The [GDC source manifest](source_wsis.tsv) records the original slide identities and checksums; [figure metadata](figure_metadata.json) records the saved model IDs and hashes of the OpenTME thumbnail inputs. The saved TME-LR models came from the [matched baseline campaign](../../campaigns/baselines_20260914/). Keep the gated OpenTME release, processed measurement tables, checkpoints, and generated numeric records in their authorized locations; this repository includes only the derived publication figures and renderer.

**Data attribution:** Galama et al., [*OpenTME: An Open Dataset of AI-Powered H&E Tumor Microenvironment Profiles from TCGA*](https://arxiv.org/abs/2604.12075), 2026; Aignostics and OpenTME. TCGA/GDC provided the source whole-slide images.
