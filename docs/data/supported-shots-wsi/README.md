# Supported-shot TME and WSI results

Patient AUROC for TC-LR, Native, Fusion50, FusionVal, and actual, zero-TME, and shuffled-TME adapters across four TCGA cohorts, seven architectures, and PLIP and CLIP-RN50. All AUROCs are on the 0–1 scale. A group is one cohort, shot count, architecture, and encoder. Shots are training examples per class.

| Cohort | Supported shots | Architecture–encoder groups |
|---|---|---:|
| NSCLC | 4, 8, 16, 32, 64 | 70 |
| BRCA | 4, 8, 16, 32, 64 | 70 |
| CRC | 4, 8, 16 | 42 |
| BLCA | 4, 8, 16, 32 | 56 |

The files contain **238 groups and 1,190 fold rows**. Native, TC-LR, Actual, Zero, and Shuffled have five folds for every group. CRC does not support 32/64 shots and BLCA does not support 64 shots under the equal-sized, disjoint training and validation protocol.

- [all_variants_group_auroc.csv](all_variants_group_auroc.csv): the combined table, with fold counts, five-fold means, and sample fold SDs for every variant. A mean and SD are blank when fewer than five folds are available.
- [all_variants_fold_auroc.csv](all_variants_fold_auroc.csv): individual fold AUROCs, campaign tags, and fusion status. Blank fusion cells mean no current matched fusion result.
- [group_auroc.csv](group_auroc.csv): complete five-fold Native and adapter means, sample fold SDs, and Actual minus Native AUROC.
- [fold_auroc.csv](fold_auroc.csv): individual Native and adapter fold AUROCs and source campaign. `native_record_id` and `native_checkpoint_sha256` are provenance hashes where available; blank hash cells do not mean missing AUROC.

HiVE-MIL 4/8/32/64-shot Native and adapter records come exclusively from `hive_refresh_20261003_v2`; the current 16-shot HiVE-MIL records come from the same refresh through the [current 16-shot table](../current-16shot/README.md). Historical HiVE-MIL records from shared low/high-shot campaigns were excluded. The 4/8-shot ViLa-MIL and MGPATH runs for NSCLC and BRCA come from the earlier dedicated low-shot campaign; encoder labels were checked against saved run configs, and reused Native metrics were resolved from each run’s recorded Native source. Other non-HiVE 4/8-shot runs come from the all-model completion campaign. Non-HiVE 32/64-shot runs come from the high-shot campaign.

TC-LR values come from the [all-shot LR release](../all-shot-lr-20260915/README.md). The current 16-shot Fusion50 and FusionVal values come from the [matched 16-shot table](../current-16shot/README.md). For other shots, 750 non-HiVE fusion folds were retained only after their saved Native and TC-LR AUROCs matched these published fold records. Thirty non-HiVE higher-shot fusion folds remain waiting or failed, so affected group means are blank. This leaves 204 groups with complete five-fold Fusion50 and 196 with complete five-fold FusionVal. The eight current 16-shot HiVE-MIL groups have Fusion50 but no FusionVal. HiVE-MIL fusion at other shots has not been recomputed against the refresh and is blank; older HiVE-MIL fusion records are excluded.

No raw patient measurements, identifiers, slide predictions, features, or checkpoints are published.

## SHA-256

- `all_variants_group_auroc.csv`: `5fe42d4ce4404eccfa9c927651cc6f2508ff48ee1b6cb1fba7efcd34313c33b3`
- `all_variants_fold_auroc.csv`: `7c61a0a2dd67ba2df2ad1806737159129f9617864f23974e11724537c7881324`
- `group_auroc.csv`: `bdff8ecb2906fb49d5d5a258110fba21fd5ab5d8fcb96f66c5b03136986532b5`
- `fold_auroc.csv`: `a3cd39cb51fa068b0df7e559b20a99268fe08256061de708fcd974ba19d1d82a`
