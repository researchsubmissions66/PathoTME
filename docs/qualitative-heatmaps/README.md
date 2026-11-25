# Held-out WSI thumbnail heatmaps

These eight figures show real TCGA whole-slide thumbnails with spatial model-sensitivity scores from the saved **MUSE · CLIP-RN50 · 16-shot · fold 0 · actual-TME** checkpoints. There is one slide per class in BRCA, NSCLC, CRC, and BLCA. Each is the lexicographically first slide ID for that class in the checkpoint's held-out fold. The source slides were downloaded from GDC and checked against their manifest MD5 values. No raw WSIs, OpenTME measurement rows, or feature HDF5 files are included here.

| Cohort | Held-out slide | Displayed class | Figure | Difference range (pp) | Saved class probability |
|---|---|---|---|---:|---:|
| BRCA | TCGA-4H-AAAK-01Z-00-DX1 | ILC | [PNG](brca_ilc.png) | −7.1041 to +6.6640 | 0.8211 |
| BRCA | TCGA-A1-A0SD-01Z-00-DX1 | IDC | [PNG](brca_idc.png) | −1.0084 to +0.2007 | 0.8544 |
| NSCLC | TCGA-05-4382-01Z-00-DX1 | LUAD | [PNG](nsclc_luad.png) | −0.0108 to +0.0053 | 0.4218 |
| NSCLC | TCGA-18-3417-01Z-00-DX1 | LUSC | [PNG](nsclc_lusc.png) | −0.0014 to +0.0014 | 0.4906 |
| CRC | TCGA-5M-AAT6-01Z-00-DX1 | ADENO_NOS | [PNG](crc_adeno_nos.png) | −0.1405 to +0.6832 | 0.5937 |
| CRC | TCGA-AA-3852-01Z-00-DX1 | MUCINOUS | [PNG](crc_mucinous.png) | −7.6849 to +0.4454 | 0.6441 |
| BLCA | TCGA-2F-A9KQ-01Z-00-DX1 | NON_PAPILLARY | [PNG](blca_non_papillary.png) | −0.0129 to +0.0099 | 0.6729 |
| BLCA | TCGA-4Z-AA81-01Z-00-DX1 | PAPILLARY | [PNG](blca_papillary.png) | −0.1802 to +0.2052 | 0.5265 |

The two NSCLC examples are misclassified by the saved checkpoint. They remain in the table because the slide rule was fixed independently of prediction correctness. The displayed class is the slide's held-out label throughout.

## Score and reading the panels

All figures use the same slide-level TME measurement, `RELATIVE_AREA_OUTER_INVASIVE_MARGIN`, as an exploratory reference. It was chosen after inspecting BRCA attribution results and then held fixed across cohorts; these figures are not a confirmatory cross-cohort test. In the middle panel this measurement is replaced by its **saved training-fold mean after standardization**. It does not mean zero tumor margin or absence of the measured phenotype.

The slide's consumed visual patches are partitioned into at most 64 deterministic spatial groups by recursively bisecting the largest group along its longest coordinate axis. For each group, its patch embeddings are replaced with the fixed mean embedding of the original bag, and the model is rerun. For displayed class *c*:

- **Measured TME:** `100 × [p_c(original bag, measured TME) − p_c(group-replaced bag, measured TME)]`.
- **Feature at training mean:** the same group effect after replacing the selected TME measurement with its training-fold reference.
- **Measured − reference:** the difference between those two effects.

Values are percentage points. Red is positive and blue is negative for the displayed class. A group's **joint score** is repeated on its member patches; it is not an individual-patch contribution. Unmeasured tissue is left uncolored. The measured and reference panels share a color scale **within each figure**; the difference panel has its own labeled scale. Color intensity cannot be compared across different slides without reading their numeric scales. In particular, the NSCLC difference ranges are very small. These maps show model sensitivity under a specific replacement, not cell localization, an attention map, an additive explanation, or a causal effect.

The source and formula are described in the [qualitative patch–TME guide](../../QUALITATIVE_PATCH_TME.md). The figures were rendered with [`generate_tme_attribution.py`](../../scripts/generate_tme_attribution.py) and [`render_tme_spatial_triptych.py`](../../scripts/render_tme_spatial_triptych.py). The spatial computation checks that rerunning the original input reproduces the saved prediction and that the model returns to its original state after interventions.

## Reproduce with the actual inputs

The original result artifacts, real feature files, OpenTME table, and source WSI must exist at the paths recorded in each saved launch. Set `RESULTS` to the directory containing those artifacts and `WSI` to the matching original slide. For NSCLC and BRCA, the completion is under `muse_tcga_16shot_20260912_v1/runs/<cohort>/clip-rn50/fold0/fold_complete.json`; for CRC and BLCA, it is under `crc_blca_core62_20260914_v1/<cohort>/runs/muse/clip-rn50/fold0/fold_complete.json`.

```bash
python scripts/generate_tme_attribution.py \
  --completion "$COMPLETION" --slide-id "$SLIDE_ID" \
  --granularity groups --device cpu --threads 2 \
  --spatial-reference RELATIVE_AREA_OUTER_INVASIVE_MARGIN \
  --spatial-groups 64 --wsi "$WSI" --output "$EXPORT" --execute
python scripts/render_tme_spatial_triptych.py \
  --spatial "$EXPORT/spatial/spatial_scores.json" \
  --class-name "$CLASS" --output "$EXPORT/spatial/triptych.png"
```

The original WSI identities and checksums are recorded in [GDC source manifest](source_wsis.tsv). Keep downloaded WSIs and generated numeric attribution records outside the source repository.
