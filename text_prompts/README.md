# PathoTME text prompt banks

This directory includes the text assets used by all seven PathoTME architectures across NSCLC, BRCA, CRC, and BLCA. PLIP and CLIP-RN50 use the same source text banks; encoder-specific text processing remains in the model implementations.

| Cohort | Prompt location |
| --- | --- |
| NSCLC | `native_tcga/text_prompts/`: ViLa-MIL, MGPATH, FOCUS, MUSE, HiVE-MIL, and DyKo; `native_tcga/train_data/gpt/description/`: MSCPT descriptions and patch-selection prompts |
| BRCA | `native_tcga/text_prompts/`: ViLa-MIL, MGPATH, FOCUS, MUSE, HiVE-MIL, and MSCPT; `dyko_brca_v1/`: DyKo class descriptions and knowledge bank |
| CRC | `crc_blca_gpt6_astra_v1/crc/`: all seven architectures |
| BLCA | `crc_blca_gpt6_astra_v1/blca/`: all seven architectures |

## Native NSCLC and BRCA assets

The files under `native_tcga/` are byte-identical copies of the PGVL-Gym sources referenced by matched PathoTME experiment configurations. Its directory structure preserves the original repository-relative paths. [MANIFEST.json](native_tcga/MANIFEST.json) records each asset's cohort, method, original relative path, SHA-256, and available source provenance. MUSE also uses the shared `class_template.json`.

These copies complete the public release. Existing queued experiments continue to load their original files; this addition does not rewrite sealed configurations or change prompts. When setting up a new installation, resolve the manifest's source paths against this snapshot or provide the corresponding files in PGVL-Gym.

MSCPT NSCLC uses `Lung.json` for descriptions and `Lung_select_pic.json` for frozen patch selection. BRCA uses `TCGA_BRCA_IDC_ILC.json` without that NSCLC selector. The existing MSCPT implementation retains its disclosed token truncation policy.

## Authored banks

CRC and BLCA banks retain their [manifest](crc_blca_gpt6_astra_v1/MANIFEST.json) and [source record](crc_blca_gpt6_astra_v1/source.json). BRCA DyKo retains its class descriptions and authored knowledge bank. Text banks are separate from encoded concept tensors, model weights, and patient data.
