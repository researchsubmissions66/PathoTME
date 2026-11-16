# Current matched 16-shot WSI results

Patient-level AUROC for four TCGA cohorts, seven WSI architectures, PLIP and CLIP-RN50, five patient-disjoint folds, and 16 training examples per class. Values are on the 0–1 scale. The group table has 56 architecture–encoder–cohort rows; the fold table has 280 rows.

- [current_group_metrics.csv](current_group_metrics.csv): five-fold means and fold SDs for Native, TME-only LR, fixed-half fusion, actual/zero/shuffled adapters, and validation-selected fusion where available.
- [current_fold_metrics.csv](current_fold_metrics.csv): the corresponding five individual fold AUROCs per architecture, encoder, and cohort.

HiVE-MIL rows are exclusively from the October 3 refresh (`hive_refresh_20261003_v2`). Their fixed-half fusion was recomputed with refreshed native predictions and matching TME-only LR predictions. HiVE-MIL validation-selected fusion is unavailable and appears as a blank cell. All other rows are from the matched original 16-shot campaign. A blank cell must not be interpreted as zero.

These tables report aggregate and fold-level metrics only; no patient or slide identifiers, raw predictions, TME measurements, or checkpoints are included.

## SHA-256

- `current_group_metrics.csv`: `0f5ac5de8424306e9b4b812dd98e25d696f5d8f475bb0fe802d447c1989fb595`
- `current_fold_metrics.csv`: `8c6597758aca0841d18cb8466791d3d2a74dfc47352e67610bdb05d1f6f70cfa`
