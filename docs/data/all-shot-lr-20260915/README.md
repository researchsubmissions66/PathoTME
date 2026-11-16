# All-shot TME-only LR results

This release contains five-fold, patient-level aggregate results for the quantitative TME-only logistic-regression classifier and its feature ablations. It has no image encoder or WSI architecture. Shots are training examples per class, with a disjoint validation set of the same size. AUROC values below are five-fold mean ± sample SD, in percent.

| Cohort | Shots | Full-panel AUROC (%) |
|---|---:|---:|
| NSCLC | 4 | 61.75 ± 5.77 |
| NSCLC | 8 | 66.59 ± 5.62 |
| NSCLC | 16 | 78.70 ± 4.41 |
| NSCLC | 32 | 81.79 ± 1.95 |
| NSCLC | 64 | 84.66 ± 1.19 |
| BRCA | 4 | 80.47 ± 4.76 |
| BRCA | 8 | 80.67 ± 3.82 |
| BRCA | 16 | 83.41 ± 4.54 |
| BRCA | 32 | 87.35 ± 2.41 |
| BRCA | 64 | 89.90 ± 1.88 |
| CRC | 4 | 64.74 ± 6.09 |
| CRC | 8 | 60.54 ± 5.92 |
| CRC | 16 | 62.31 ± 6.60 |
| BLCA | 4 | 54.44 ± 13.25 |
| BLCA | 8 | 60.53 ± 8.53 |
| BLCA | 16 | 65.49 ± 4.59 |
| BLCA | 32 | 67.19 ± 5.38 |

CRC has no supported 32- or 64-shot split, and BLCA has no supported 64-shot split because their development folds cannot supply equal-sized disjoint training and validation sets at those sizes. Every published condition has five folds. The 16-shot conditions reuse the verified original 16-shot fit. BRCA uses the historical morphology64 panel; the other cohorts use core62.

## Files

- [condition_summary.csv](condition_summary.csv): 714 conditions with mean, fold SD, and paired delta metrics; includes full panel, selected 8/16/32 subsets, group-only and group-removal ablations, and random panels.
- [fold_metrics.csv](fold_metrics.csv): 3,570 per-fold aggregate metric rows, with patient counts but no patient identifiers.
- [random_panel_summary.csv](random_panel_summary.csv): spread across 10 random panels at each supported size; the range is not a confidence interval.
- [selection_frequency.csv](selection_frequency.csv): feature selection frequency over the five folds.

All metrics are calculated at patient level. Feature selection and regularization are based on training and validation data; held-out test data are evaluated afterward. The ablations describe model sensitivity and do not establish causal effects. No raw patient measurements, predictions, slide IDs, or checkpoints are included.

This is the TME-only result set. It does not establish complete all-shot coverage for Native, Fusion50, FusionVal, or Adapter WSI models. For the current matched 16-shot WSI comparison, see [current-16shot](../current-16shot/README.md).

## SHA-256

- `condition_summary.csv`: `ccd2cf881a99bbf23cd7d9cbc9c4a896c9c71d2334cc0574a8a7dbb710398095`
- `fold_metrics.csv`: `850e68a14b99d6d93e2ecb52da8b8f6c2a8fab2a7c2f119884d6f3629d16c5d2`
- `random_panel_summary.csv`: `b7b5d63a716e41e4f30f5a11288ced971bff69dc34709724c0463410eca9bad9`
- `selection_frequency.csv`: `b9aeb4932d893296b7614e0694d58fb75522fa6015927592b0269c2ffd75a8b6`
