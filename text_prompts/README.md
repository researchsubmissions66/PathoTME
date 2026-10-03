# PathoTME text prompt banks

This directory includes the text assets used by all seven PathoTME architectures across NSCLC, BRCA, CRC, and BLCA. PLIP and CLIP-RN50 use the same source text banks; encoder-specific text processing remains in the model implementations.

| Cohort | Prompt location |
| --- | --- |
| NSCLC | `native_tcga/text_prompts/`: ViLa-MIL, MGPATH, FOCUS, MUSE, HiVE-MIL, and DyKo; `native_tcga/train_data/gpt/description/`: MSCPT descriptions and patch-selection prompts |
| BRCA | `native_tcga/text_prompts/`: ViLa-MIL, MGPATH, FOCUS, MUSE, HiVE-MIL, and MSCPT; `dyko_brca_v1/`: DyKo class descriptions and knowledge bank |
| CRC | `crc_blca_gpt6_astra_v1/crc/`: all seven architectures |
| BLCA | `crc_blca_gpt6_astra_v1/blca/`: all seven architectures |

## Direct links by architecture and cohort

All 28 combinations have source text banks. [COVERAGE.json](COVERAGE.json) records the published assets and their SHA-256 hashes.

| Architecture | NSCLC | BRCA | CRC | BLCA |
| --- | --- | --- | --- | --- |
| ViLa-MIL | [TCGA_Lung_two_scale_text_prompt.csv](native_tcga/text_prompts/vila_mil/TCGA_Lung_two_scale_text_prompt.csv) | [TCGA_BRCA_two_scale_text_prompt.csv](native_tcga/text_prompts/vila_mil/TCGA_BRCA_two_scale_text_prompt.csv) | [two_scale.csv](crc_blca_gpt6_astra_v1/crc/vila_mil/two_scale.csv) | [two_scale.csv](crc_blca_gpt6_astra_v1/blca/vila_mil/two_scale.csv) |
| MGPATH | [tcga_nsclc.csv](native_tcga/text_prompts/mgpath/tcga_nsclc.csv) | [tcga_brca.csv](native_tcga/text_prompts/mgpath/tcga_brca.csv) | [two_scale.csv](crc_blca_gpt6_astra_v1/crc/mgpath/two_scale.csv) | [two_scale.csv](crc_blca_gpt6_astra_v1/blca/mgpath/two_scale.csv) |
| FOCUS | [TCGA_NSCLC_two_scale_text_prompt.csv](native_tcga/text_prompts/focus/TCGA_NSCLC_two_scale_text_prompt.csv) | [TCGA_BRCA_two_scale_text_prompt.csv](native_tcga/text_prompts/focus/TCGA_BRCA_two_scale_text_prompt.csv) | [two_scale.csv](crc_blca_gpt6_astra_v1/crc/focus/two_scale.csv) | [two_scale.csv](crc_blca_gpt6_astra_v1/blca/focus/two_scale.csv) |
| MUSE | [class_template.json](native_tcga/text_prompts/muse/class_template.json)<br>[generated_new_0.csv](native_tcga/text_prompts/muse/tcga_nsclc/generated_new_0.csv)<br>[generated_new_1.csv](native_tcga/text_prompts/muse/tcga_nsclc/generated_new_1.csv) | [class_template.json](native_tcga/text_prompts/muse/class_template.json)<br>[generated_new_0.csv](native_tcga/text_prompts/muse/tcga_brca/generated_new_0.csv)<br>[generated_new_1.csv](native_tcga/text_prompts/muse/tcga_brca/generated_new_1.csv) | [class_semantics.json](crc_blca_gpt6_astra_v1/crc/muse/class_semantics.json)<br>[generated_new_0.csv](crc_blca_gpt6_astra_v1/crc/muse/generated_new_0.csv)<br>[generated_new_1.csv](crc_blca_gpt6_astra_v1/crc/muse/generated_new_1.csv) | [class_semantics.json](crc_blca_gpt6_astra_v1/blca/muse/class_semantics.json)<br>[generated_new_0.csv](crc_blca_gpt6_astra_v1/blca/muse/generated_new_0.csv)<br>[generated_new_1.csv](crc_blca_gpt6_astra_v1/blca/muse/generated_new_1.csv) |
| HiVE-MIL | [tcga_nsclc.json](native_tcga/text_prompts/hive_mil/tcga_nsclc.json) | [tcga_brca.json](native_tcga/text_prompts/hive_mil/tcga_brca.json) | [hierarchy.json](crc_blca_gpt6_astra_v1/crc/hive_mil/hierarchy.json) | [hierarchy.json](crc_blca_gpt6_astra_v1/blca/hive_mil/hierarchy.json) |
| DyKo | [NSCLC_text_prompt.csv](native_tcga/text_prompts/dyko/NSCLC_text_prompt.csv) | [class_descriptions.csv](dyko_brca_v1/class_descriptions.csv)<br>[knowledge_bank.json](dyko_brca_v1/knowledge_bank.json) | [class_descriptions.csv](crc_blca_gpt6_astra_v1/crc/dyko/class_descriptions.csv)<br>[knowledge_bank.json](crc_blca_gpt6_astra_v1/crc/dyko/knowledge_bank.json) | [class_descriptions.csv](crc_blca_gpt6_astra_v1/blca/dyko/class_descriptions.csv)<br>[knowledge_bank.json](crc_blca_gpt6_astra_v1/blca/dyko/knowledge_bank.json) |
| MSCPT | [Lung.json](native_tcga/train_data/gpt/description/Lung.json)<br>[Lung_select_pic.json](native_tcga/train_data/gpt/description/Lung_select_pic.json) | [TCGA_BRCA_IDC_ILC.json](native_tcga/text_prompts/mscpt/description/TCGA_BRCA_IDC_ILC.json) | [TCGA_CRC.json](crc_blca_gpt6_astra_v1/crc/mscpt/description/TCGA_CRC.json) | [TCGA_BLCA.json](crc_blca_gpt6_astra_v1/blca/mscpt/description/TCGA_BLCA.json) |

## Native NSCLC and BRCA assets

The files under `native_tcga/` are byte-identical copies of the PGVL-Gym sources referenced by matched PathoTME experiment configurations. Its directory structure preserves the original repository-relative paths. [MANIFEST.json](native_tcga/MANIFEST.json) records each asset's cohort, method, original relative path, SHA-256, and available source provenance. MUSE also uses the shared `class_template.json`.

These copies complete the public release. Existing queued experiments continue to load their original files; this addition does not rewrite sealed configurations or change prompts. When setting up a new installation, resolve the manifest's source paths against this snapshot or provide the corresponding files in PGVL-Gym.

MSCPT NSCLC uses `Lung.json` for descriptions and `Lung_select_pic.json` for frozen patch selection. BRCA uses `TCGA_BRCA_IDC_ILC.json` without that NSCLC selector. The existing MSCPT implementation retains its disclosed token truncation policy.

## Authored banks

CRC and BLCA banks retain their [manifest](crc_blca_gpt6_astra_v1/MANIFEST.json) and [source record](crc_blca_gpt6_astra_v1/source.json). BRCA DyKo retains its class descriptions and authored knowledge bank. Text banks are separate from encoded concept tensors, model weights, and patient data.
