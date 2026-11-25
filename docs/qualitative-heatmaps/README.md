# Held-out WSI thumbnail heatmaps

These eight figures show real TCGA whole-slide thumbnails with spatial model-sensitivity scores from the saved **MUSE · CLIP-RN50 · 16-shot · fold 0 · actual-TME** checkpoints. There is one slide per class in BRCA, NSCLC, CRC, and BLCA. Each is the lexicographically first slide ID for that class in the checkpoint's held-out fold. The source slides were downloaded from GDC and checked against their manifest MD5 values. No raw WSIs, OpenTME measurement rows, or feature HDF5 files are included here.

| Cohort | Held-out slide | Displayed class | Figure | Difference range (pp) | Saved class probability |
|---|---|---|---|---:|---:|
| BRCA | TCGA-4H-AAAK-01Z-00-DX1 | ILC | [PNG](brca_ilc_heatmap.png) | −7.1041 to +6.6640 | 0.8211 |
| BRCA | TCGA-A1-A0SD-01Z-00-DX1 | IDC | [PNG](brca_idc_heatmap.png) | −1.0084 to +0.2007 | 0.8544 |
| NSCLC | TCGA-05-4382-01Z-00-DX1 | LUAD | [PNG](nsclc_luad_heatmap.png) | −0.0108 to +0.0053 | 0.4218 |
| NSCLC | TCGA-18-3417-01Z-00-DX1 | LUSC | [PNG](nsclc_lusc_heatmap.png) | −0.0014 to +0.0014 | 0.4906 |
| CRC | TCGA-5M-AAT6-01Z-00-DX1 | ADENO_NOS | [PNG](crc_adeno_nos_heatmap.png) | −0.1405 to +0.6832 | 0.5937 |
| CRC | TCGA-AA-3852-01Z-00-DX1 | MUCINOUS | [PNG](crc_mucinous_heatmap.png) | −7.6849 to +0.4454 | 0.6441 |
| BLCA | TCGA-2F-A9KQ-01Z-00-DX1 | NON_PAPILLARY | [PNG](blca_non_papillary_heatmap.png) | −0.0129 to +0.0099 | 0.6729 |
| BLCA | TCGA-4Z-AA81-01Z-00-DX1 | PAPILLARY | [PNG](blca_papillary_heatmap.png) | −0.1802 to +0.2052 | 0.5265 |

The two NSCLC examples are misclassified by the saved checkpoint. They remain in the table because the slide rule was fixed independently of prediction correctness. The displayed class is the slide's held-out label throughout.

## Reading a thumbnail

Each PNG is **one entire WSI thumbnail**. Colored tiles cover the real patch coordinates consumed by the model; they span the measured portion of the slide. The color shows how the slide-level TME measurement changes the classifier's response to each spatial group. It does **not** show where that TME feature, a cell type, or a tissue compartment was measured.

All figures use `RELATIVE_AREA_OUTER_INVASIVE_MARGIN` as an exploratory reference. It was chosen after inspecting BRCA attribution results and then held fixed across cohorts; these figures are not a confirmatory cross-cohort test. The reference sets this measurement to its **saved training-fold mean after standardization**. It does not mean zero tumor margin or absence of the measured phenotype.

The slide's consumed visual patches are partitioned into at most 64 deterministic spatial groups by recursively bisecting the largest group along its longest coordinate axis. For each group, its patch embeddings are replaced with the fixed mean embedding of the original bag, and the model is rerun. For displayed class *c*:

`Heatmap score = 100 × {[p_c(original bag, measured TME) − p_c(group-replaced bag, measured TME)] − [p_c(original bag, reference TME) − p_c(group-replaced bag, reference TME)]}`

Values are percentage points. Red means the region's class-probability effect is higher with measured TME; blue means lower. A group's **joint score** is repeated on its member patches; it is not an individual-patch contribution. No values are interpolated between patches, and unmeasured tissue is left uncolored. Each thumbnail has its own symmetric color scale, clipped at the patch-weighted 95th percentile of absolute scores so that one outlier does not wash out the whole slide. The maximum absolute score is printed below the color bar, and the full score range is in the table. Color intensity cannot be compared across slides without reading these scales. In particular, the NSCLC differences are very small. These maps show model sensitivity under a specific replacement, not cell localization, an attention map, an additive explanation, or a causal effect.

The source and formula are described in the [qualitative patch–TME guide](../../QUALITATIVE_PATCH_TME.md). The figures were rendered with [`generate_tme_attribution.py`](../../scripts/generate_tme_attribution.py) and [`render_tme_thumbnail_heatmap.py`](../../scripts/render_tme_thumbnail_heatmap.py). The spatial computation checks that rerunning the original input reproduces the saved prediction and that the model returns to its original state after interventions.

## Reproduce with the actual inputs

The original result artifacts, real feature files, OpenTME table, and source WSI must exist at the paths recorded in each saved launch. Set `RESULTS` to the directory containing those artifacts and `WSI` to the matching original slide. For NSCLC and BRCA, the completion is under `muse_tcga_16shot_20260912_v1/runs/<cohort>/clip-rn50/fold0/fold_complete.json`; for CRC and BLCA, it is under `crc_blca_core62_20260914_v1/<cohort>/runs/muse/clip-rn50/fold0/fold_complete.json`.

```bash
python scripts/generate_tme_attribution.py \
  --completion "$COMPLETION" --slide-id "$SLIDE_ID" \
  --granularity groups --device cpu --threads 2 \
  --spatial-reference RELATIVE_AREA_OUTER_INVASIVE_MARGIN \
  --spatial-groups 64 --wsi "$WSI" --output "$EXPORT" --execute
python scripts/render_tme_thumbnail_heatmap.py \
  --spatial "$EXPORT/spatial/spatial_scores.json" \
  --class-name "$CLASS" --output "$EXPORT/spatial/thumbnail_heatmap.png"
```

The original WSI identities and checksums are recorded in [GDC source manifest](source_wsis.tsv). Keep downloaded WSIs and generated numeric attribution records outside the source repository.
