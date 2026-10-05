# 16-shot FusionVal TME attribution charts

Each row names the image architecture and encoder used for the saved
prediction. The chart plots the top 10 TME groups and top 10 individual
features for that exact condition. Its TME feature score equals the
saved validation-selected TME weight (α) times the matched PathoTME-LR
score. The image prediction stays fixed under the TME intervention;
these charts do not measure image-dependent feature effects.

The α values below are recovered from the published per-fold fusion/LR
score ratios and checked across all groups. Fold values and sample SDs
come from the published aggregate CSVs. Native has structural zero
TME attribution; learned-adapter attribution is still incomplete.

## NSCLC

| Architecture | Encoder | α, folds 0–4 | PNG | SVG | PDF |
|---|---|---|---|---|---|
| DyKo | CLIP-RN50 | 0.75, 1.00, 0.50, 1.00, 1.00 | [PNG](nsclc/nsclc_dyko_clip-rn50_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_dyko_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_dyko_clip-rn50_fusionval_16shot_attribution.pdf) |
| DyKo | PLIP | 1.00, 0.50, 1.00, 1.00, 0.75 | [PNG](nsclc/nsclc_dyko_plip_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_dyko_plip_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_dyko_plip_fusionval_16shot_attribution.pdf) |
| FOCUS | CLIP-RN50 | 1.00, 1.00, 0.75, 1.00, 1.00 | [PNG](nsclc/nsclc_focus_clip-rn50_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_focus_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_focus_clip-rn50_fusionval_16shot_attribution.pdf) |
| FOCUS | PLIP | 0.75, 0.75, 1.00, 1.00, 0.75 | [PNG](nsclc/nsclc_focus_plip_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_focus_plip_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_focus_plip_fusionval_16shot_attribution.pdf) |
| HiVE-MIL | CLIP-RN50 | 0.25, 1.00, 0.75, 0.25, 1.00 | [PNG](nsclc/nsclc_hive_mil_clip-rn50_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_hive_mil_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_hive_mil_clip-rn50_fusionval_16shot_attribution.pdf) |
| HiVE-MIL | PLIP | 0.25, 0.75, 0.25, 1.00, 0.00 | [PNG](nsclc/nsclc_hive_mil_plip_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_hive_mil_plip_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_hive_mil_plip_fusionval_16shot_attribution.pdf) |
| MGPATH | CLIP-RN50 | 0.50, 0.75, 0.75, 1.00, 1.00 | [PNG](nsclc/nsclc_mgpath_clip-rn50_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_mgpath_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_mgpath_clip-rn50_fusionval_16shot_attribution.pdf) |
| MGPATH | PLIP | 0.75, 0.75, 1.00, 1.00, 0.50 | [PNG](nsclc/nsclc_mgpath_plip_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_mgpath_plip_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_mgpath_plip_fusionval_16shot_attribution.pdf) |
| MSCPT | CLIP-RN50 | 1.00, 0.75, 1.00, 1.00, 0.75 | [PNG](nsclc/nsclc_mscpt_clip-rn50_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_mscpt_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_mscpt_clip-rn50_fusionval_16shot_attribution.pdf) |
| MSCPT | PLIP | 0.25, 0.75, 0.50, 1.00, 0.25 | [PNG](nsclc/nsclc_mscpt_plip_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_mscpt_plip_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_mscpt_plip_fusionval_16shot_attribution.pdf) |
| MUSE | CLIP-RN50 | 1.00, 0.75, 0.75, 1.00, 1.00 | [PNG](nsclc/nsclc_muse_clip-rn50_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_muse_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_muse_clip-rn50_fusionval_16shot_attribution.pdf) |
| MUSE | PLIP | 0.75, 0.75, 1.00, 1.00, 0.75 | [PNG](nsclc/nsclc_muse_plip_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_muse_plip_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_muse_plip_fusionval_16shot_attribution.pdf) |
| ViLa-MIL | CLIP-RN50 | 0.75, 0.75, 1.00, 1.00, 1.00 | [PNG](nsclc/nsclc_vila_mil_clip-rn50_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_vila_mil_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_vila_mil_clip-rn50_fusionval_16shot_attribution.pdf) |
| ViLa-MIL | PLIP | 0.25, 0.75, 0.50, 1.00, 0.25 | [PNG](nsclc/nsclc_vila_mil_plip_fusionval_16shot_attribution.png) | [SVG](nsclc/nsclc_vila_mil_plip_fusionval_16shot_attribution.svg) | [PDF](nsclc/nsclc_vila_mil_plip_fusionval_16shot_attribution.pdf) |

## BRCA

| Architecture | Encoder | α, folds 0–4 | PNG | SVG | PDF |
|---|---|---|---|---|---|
| DyKo | CLIP-RN50 | 1.00, 0.75, 1.00, 1.00, 1.00 | [PNG](brca/brca_dyko_clip-rn50_fusionval_16shot_attribution.png) | [SVG](brca/brca_dyko_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](brca/brca_dyko_clip-rn50_fusionval_16shot_attribution.pdf) |
| DyKo | PLIP | 0.75, 1.00, 1.00, 1.00, 1.00 | [PNG](brca/brca_dyko_plip_fusionval_16shot_attribution.png) | [SVG](brca/brca_dyko_plip_fusionval_16shot_attribution.svg) | [PDF](brca/brca_dyko_plip_fusionval_16shot_attribution.pdf) |
| FOCUS | CLIP-RN50 | 1.00, 0.75, 1.00, 0.75, 1.00 | [PNG](brca/brca_focus_clip-rn50_fusionval_16shot_attribution.png) | [SVG](brca/brca_focus_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](brca/brca_focus_clip-rn50_fusionval_16shot_attribution.pdf) |
| FOCUS | PLIP | 0.25, 0.75, 1.00, 0.75, 1.00 | [PNG](brca/brca_focus_plip_fusionval_16shot_attribution.png) | [SVG](brca/brca_focus_plip_fusionval_16shot_attribution.svg) | [PDF](brca/brca_focus_plip_fusionval_16shot_attribution.pdf) |
| HiVE-MIL | CLIP-RN50 | 1.00, 1.00, 1.00, 1.00, 1.00 | [PNG](brca/brca_hive_mil_clip-rn50_fusionval_16shot_attribution.png) | [SVG](brca/brca_hive_mil_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](brca/brca_hive_mil_clip-rn50_fusionval_16shot_attribution.pdf) |
| HiVE-MIL | PLIP | 0.00, 0.75, 0.25, 0.25, 1.00 | [PNG](brca/brca_hive_mil_plip_fusionval_16shot_attribution.png) | [SVG](brca/brca_hive_mil_plip_fusionval_16shot_attribution.svg) | [PDF](brca/brca_hive_mil_plip_fusionval_16shot_attribution.pdf) |
| MGPATH | CLIP-RN50 | 1.00, 0.50, 1.00, 0.75, 1.00 | [PNG](brca/brca_mgpath_clip-rn50_fusionval_16shot_attribution.png) | [SVG](brca/brca_mgpath_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](brca/brca_mgpath_clip-rn50_fusionval_16shot_attribution.pdf) |
| MGPATH | PLIP | 0.75, 0.75, 1.00, 0.75, 1.00 | [PNG](brca/brca_mgpath_plip_fusionval_16shot_attribution.png) | [SVG](brca/brca_mgpath_plip_fusionval_16shot_attribution.svg) | [PDF](brca/brca_mgpath_plip_fusionval_16shot_attribution.pdf) |
| MSCPT | CLIP-RN50 | 1.00, 1.00, 1.00, 1.00, 1.00 | [PNG](brca/brca_mscpt_clip-rn50_fusionval_16shot_attribution.png) | [SVG](brca/brca_mscpt_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](brca/brca_mscpt_clip-rn50_fusionval_16shot_attribution.pdf) |
| MSCPT | PLIP | 0.50, 0.75, 1.00, 0.75, 1.00 | [PNG](brca/brca_mscpt_plip_fusionval_16shot_attribution.png) | [SVG](brca/brca_mscpt_plip_fusionval_16shot_attribution.svg) | [PDF](brca/brca_mscpt_plip_fusionval_16shot_attribution.pdf) |
| MUSE | CLIP-RN50 | 1.00, 1.00, 0.75, 1.00, 0.75 | [PNG](brca/brca_muse_clip-rn50_fusionval_16shot_attribution.png) | [SVG](brca/brca_muse_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](brca/brca_muse_clip-rn50_fusionval_16shot_attribution.pdf) |
| MUSE | PLIP | 0.75, 0.75, 0.75, 0.75, 1.00 | [PNG](brca/brca_muse_plip_fusionval_16shot_attribution.png) | [SVG](brca/brca_muse_plip_fusionval_16shot_attribution.svg) | [PDF](brca/brca_muse_plip_fusionval_16shot_attribution.pdf) |
| ViLa-MIL | CLIP-RN50 | 1.00, 0.75, 1.00, 0.75, 1.00 | [PNG](brca/brca_vila_mil_clip-rn50_fusionval_16shot_attribution.png) | [SVG](brca/brca_vila_mil_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](brca/brca_vila_mil_clip-rn50_fusionval_16shot_attribution.pdf) |
| ViLa-MIL | PLIP | 0.75, 0.75, 1.00, 1.00, 0.75 | [PNG](brca/brca_vila_mil_plip_fusionval_16shot_attribution.png) | [SVG](brca/brca_vila_mil_plip_fusionval_16shot_attribution.svg) | [PDF](brca/brca_vila_mil_plip_fusionval_16shot_attribution.pdf) |

## CRC

| Architecture | Encoder | α, folds 0–4 | PNG | SVG | PDF |
|---|---|---|---|---|---|
| DyKo | CLIP-RN50 | 0.00, 0.50, 0.50, 1.00, 0.75 | [PNG](crc/crc_dyko_clip-rn50_fusionval_16shot_attribution.png) | [SVG](crc/crc_dyko_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](crc/crc_dyko_clip-rn50_fusionval_16shot_attribution.pdf) |
| DyKo | PLIP | 0.75, 0.25, 0.50, 0.75, 0.25 | [PNG](crc/crc_dyko_plip_fusionval_16shot_attribution.png) | [SVG](crc/crc_dyko_plip_fusionval_16shot_attribution.svg) | [PDF](crc/crc_dyko_plip_fusionval_16shot_attribution.pdf) |
| FOCUS | CLIP-RN50 | 1.00, 0.75, 0.50, 0.75, 1.00 | [PNG](crc/crc_focus_clip-rn50_fusionval_16shot_attribution.png) | [SVG](crc/crc_focus_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](crc/crc_focus_clip-rn50_fusionval_16shot_attribution.pdf) |
| FOCUS | PLIP | 1.00, 0.25, 0.25, 1.00, 0.25 | [PNG](crc/crc_focus_plip_fusionval_16shot_attribution.png) | [SVG](crc/crc_focus_plip_fusionval_16shot_attribution.svg) | [PDF](crc/crc_focus_plip_fusionval_16shot_attribution.pdf) |
| HiVE-MIL | CLIP-RN50 | 0.00, 0.50, 0.75, 0.25, 0.75 | [PNG](crc/crc_hive_mil_clip-rn50_fusionval_16shot_attribution.png) | [SVG](crc/crc_hive_mil_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](crc/crc_hive_mil_clip-rn50_fusionval_16shot_attribution.pdf) |
| HiVE-MIL | PLIP | 0.50, 0.25, 0.00, 0.50, 0.25 | [PNG](crc/crc_hive_mil_plip_fusionval_16shot_attribution.png) | [SVG](crc/crc_hive_mil_plip_fusionval_16shot_attribution.svg) | [PDF](crc/crc_hive_mil_plip_fusionval_16shot_attribution.pdf) |
| MGPATH | CLIP-RN50 | 1.00, 0.25, 0.75, 1.00, 1.00 | [PNG](crc/crc_mgpath_clip-rn50_fusionval_16shot_attribution.png) | [SVG](crc/crc_mgpath_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](crc/crc_mgpath_clip-rn50_fusionval_16shot_attribution.pdf) |
| MGPATH | PLIP | 0.25, 0.50, 0.50, 0.75, 1.00 | [PNG](crc/crc_mgpath_plip_fusionval_16shot_attribution.png) | [SVG](crc/crc_mgpath_plip_fusionval_16shot_attribution.svg) | [PDF](crc/crc_mgpath_plip_fusionval_16shot_attribution.pdf) |
| MSCPT | CLIP-RN50 | 1.00, 0.75, 0.75, 1.00, 1.00 | [PNG](crc/crc_mscpt_clip-rn50_fusionval_16shot_attribution.png) | [SVG](crc/crc_mscpt_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](crc/crc_mscpt_clip-rn50_fusionval_16shot_attribution.pdf) |
| MSCPT | PLIP | 0.00, 0.25, 0.50, 0.75, 1.00 | [PNG](crc/crc_mscpt_plip_fusionval_16shot_attribution.png) | [SVG](crc/crc_mscpt_plip_fusionval_16shot_attribution.svg) | [PDF](crc/crc_mscpt_plip_fusionval_16shot_attribution.pdf) |
| MUSE | CLIP-RN50 | 0.50, 0.50, 0.50, 0.50, 0.75 | [PNG](crc/crc_muse_clip-rn50_fusionval_16shot_attribution.png) | [SVG](crc/crc_muse_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](crc/crc_muse_clip-rn50_fusionval_16shot_attribution.pdf) |
| MUSE | PLIP | 0.00, 0.75, 0.25, 0.75, 1.00 | [PNG](crc/crc_muse_plip_fusionval_16shot_attribution.png) | [SVG](crc/crc_muse_plip_fusionval_16shot_attribution.svg) | [PDF](crc/crc_muse_plip_fusionval_16shot_attribution.pdf) |
| ViLa-MIL | CLIP-RN50 | 1.00, 1.00, 0.75, 1.00, 1.00 | [PNG](crc/crc_vila_mil_clip-rn50_fusionval_16shot_attribution.png) | [SVG](crc/crc_vila_mil_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](crc/crc_vila_mil_clip-rn50_fusionval_16shot_attribution.pdf) |
| ViLa-MIL | PLIP | 0.50, 0.50, 0.75, 0.00, 1.00 | [PNG](crc/crc_vila_mil_plip_fusionval_16shot_attribution.png) | [SVG](crc/crc_vila_mil_plip_fusionval_16shot_attribution.svg) | [PDF](crc/crc_vila_mil_plip_fusionval_16shot_attribution.pdf) |

## BLCA

| Architecture | Encoder | α, folds 0–4 | PNG | SVG | PDF |
|---|---|---|---|---|---|
| DyKo | CLIP-RN50 | 0.25, 0.50, 0.50, 0.25, 0.00 | [PNG](blca/blca_dyko_clip-rn50_fusionval_16shot_attribution.png) | [SVG](blca/blca_dyko_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](blca/blca_dyko_clip-rn50_fusionval_16shot_attribution.pdf) |
| DyKo | PLIP | 0.00, 0.75, 0.00, 0.50, 0.75 | [PNG](blca/blca_dyko_plip_fusionval_16shot_attribution.png) | [SVG](blca/blca_dyko_plip_fusionval_16shot_attribution.svg) | [PDF](blca/blca_dyko_plip_fusionval_16shot_attribution.pdf) |
| FOCUS | CLIP-RN50 | 1.00, 0.75, 0.50, 1.00, 0.25 | [PNG](blca/blca_focus_clip-rn50_fusionval_16shot_attribution.png) | [SVG](blca/blca_focus_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](blca/blca_focus_clip-rn50_fusionval_16shot_attribution.pdf) |
| FOCUS | PLIP | 0.25, 0.75, 0.75, 1.00, 1.00 | [PNG](blca/blca_focus_plip_fusionval_16shot_attribution.png) | [SVG](blca/blca_focus_plip_fusionval_16shot_attribution.svg) | [PDF](blca/blca_focus_plip_fusionval_16shot_attribution.pdf) |
| HiVE-MIL | CLIP-RN50 | 0.25, 1.00, 0.00, 1.00, 0.25 | [PNG](blca/blca_hive_mil_clip-rn50_fusionval_16shot_attribution.png) | [SVG](blca/blca_hive_mil_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](blca/blca_hive_mil_clip-rn50_fusionval_16shot_attribution.pdf) |
| HiVE-MIL | PLIP | 0.00, 0.50, 0.25, 0.25, 0.00 | [PNG](blca/blca_hive_mil_plip_fusionval_16shot_attribution.png) | [SVG](blca/blca_hive_mil_plip_fusionval_16shot_attribution.svg) | [PDF](blca/blca_hive_mil_plip_fusionval_16shot_attribution.pdf) |
| MGPATH | CLIP-RN50 | 1.00, 0.75, 1.00, 0.50, 1.00 | [PNG](blca/blca_mgpath_clip-rn50_fusionval_16shot_attribution.png) | [SVG](blca/blca_mgpath_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](blca/blca_mgpath_clip-rn50_fusionval_16shot_attribution.pdf) |
| MGPATH | PLIP | 0.25, 0.50, 0.75, 0.75, 0.50 | [PNG](blca/blca_mgpath_plip_fusionval_16shot_attribution.png) | [SVG](blca/blca_mgpath_plip_fusionval_16shot_attribution.svg) | [PDF](blca/blca_mgpath_plip_fusionval_16shot_attribution.pdf) |
| MSCPT | CLIP-RN50 | 0.50, 0.75, 1.00, 0.25, 0.00 | [PNG](blca/blca_mscpt_clip-rn50_fusionval_16shot_attribution.png) | [SVG](blca/blca_mscpt_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](blca/blca_mscpt_clip-rn50_fusionval_16shot_attribution.pdf) |
| MSCPT | PLIP | 0.50, 1.00, 0.50, 0.00, 1.00 | [PNG](blca/blca_mscpt_plip_fusionval_16shot_attribution.png) | [SVG](blca/blca_mscpt_plip_fusionval_16shot_attribution.svg) | [PDF](blca/blca_mscpt_plip_fusionval_16shot_attribution.pdf) |
| MUSE | CLIP-RN50 | 0.25, 0.50, 0.75, 0.50, 0.00 | [PNG](blca/blca_muse_clip-rn50_fusionval_16shot_attribution.png) | [SVG](blca/blca_muse_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](blca/blca_muse_clip-rn50_fusionval_16shot_attribution.pdf) |
| MUSE | PLIP | 0.25, 0.25, 1.00, 1.00, 0.25 | [PNG](blca/blca_muse_plip_fusionval_16shot_attribution.png) | [SVG](blca/blca_muse_plip_fusionval_16shot_attribution.svg) | [PDF](blca/blca_muse_plip_fusionval_16shot_attribution.pdf) |
| ViLa-MIL | CLIP-RN50 | 0.75, 0.75, 0.50, 0.75, 0.25 | [PNG](blca/blca_vila_mil_clip-rn50_fusionval_16shot_attribution.png) | [SVG](blca/blca_vila_mil_clip-rn50_fusionval_16shot_attribution.svg) | [PDF](blca/blca_vila_mil_clip-rn50_fusionval_16shot_attribution.pdf) |
| ViLa-MIL | PLIP | 0.00, 0.75, 0.25, 0.75, 0.25 | [PNG](blca/blca_vila_mil_plip_fusionval_16shot_attribution.png) | [SVG](blca/blca_vila_mil_plip_fusionval_16shot_attribution.svg) | [PDF](blca/blca_vila_mil_plip_fusionval_16shot_attribution.pdf) |
