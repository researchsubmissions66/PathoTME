# Supported-shot WSI results

Five-fold patient AUROC for Native and actual, zero-TME, and shuffled-TME adapters across four TCGA cohorts, seven architectures, and PLIP and CLIP-RN50. All values are on the 0–1 scale. Each group is one cohort, shot count, architecture, and encoder. Shots are training examples per class.

| Cohort | Supported shots | Architecture–encoder groups |
|---|---|---:|
| NSCLC | 4, 8, 16, 32, 64 | 70 |
| BRCA | 4, 8, 16, 32, 64 | 70 |
| CRC | 4, 8, 16 | 42 |
| BLCA | 4, 8, 16, 32 | 56 |

The files contain **238 complete groups and 1,190 fold rows**. Every reported Native, Actual, Zero, and Shuffled group has five folds. CRC does not support 32/64 shots and BLCA does not support 64 shots under the equal-sized, disjoint training and validation protocol.

- [group_auroc.csv](group_auroc.csv): five-fold means, sample fold SDs, and Actual minus Native AUROC for each group.
- [fold_auroc.csv](fold_auroc.csv): the individual patient-level fold AUROCs and source campaign for each condition. `native_record_id` and `native_checkpoint_sha256` are provenance hashes where available; blank hash cells do not mean missing AUROC.

HiVE-MIL 4/8/32/64-shot records come exclusively from `hive_refresh_20261003_v2`; the current 16-shot HiVE-MIL records come from the same refresh through the [current 16-shot table](../current-16shot/README.md). Historical HiVE-MIL records from shared low/high-shot campaigns were excluded. The 4/8-shot ViLa-MIL and MGPATH runs for NSCLC and BRCA are from the earlier dedicated low-shot campaign; their encoder labels were checked against each saved run config, and reused Native metrics were resolved from each run’s recorded Native source. Other non-HiVE 4/8-shot runs come from the all-model completion campaign. Non-HiVE 32/64-shot runs come from the high-shot campaign.

This table covers Native and adapter ablations. The [TC-LR release](../all-shot-lr-20260915/README.md) contains the TME-only result at each supported shot. The [current 16-shot table](../current-16shot/README.md) also contains matched fixed and validation-selected fusion results where available. Multi-shot fusion needs separate matching against refreshed HiVE-MIL Native predictions and is not included here.

No raw patient measurements, identifiers, slide predictions, features, or checkpoints are published.

## SHA-256

- `group_auroc.csv`: `bdff8ecb2906fb49d5d5a258110fba21fd5ab5d8fcb96f66c5b03136986532b5`
- `fold_auroc.csv`: `a3cd39cb51fa068b0df7e559b20a99268fe08256061de708fcd974ba19d1d82a`
