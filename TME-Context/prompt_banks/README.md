# Prompt banks

This folder contains text inputs for ViLa-MIL, MGPATH, FOCUS, MUSE, HiVE-MIL,
DyKo, and MSCPT across TCGA-NSCLC, TCGA-BRCA, TCGA-CRC, and TCGA-BLCA.

| Path | Contents |
|---|---|
| `native_tcga/text_prompts/` | NSCLC and BRCA prompts for the native models |
| `native_tcga/train_data/gpt/description/` | NSCLC MSCPT descriptions and patch-selection text |
| `dyko_brca_v1/` | BRCA DyKo class descriptions and morphology concepts |
| `crc_blca_gpt6_astra_v1/` | CRC and BLCA prompt banks, except superseded HiVE hierarchies |
| `hive_crc_blca_20261003_v2/` | Current CRC and BLCA HiVE hierarchies |

The original relative file structure is retained so model-specific loaders can
map to these files. DyKo knowledge banks retain their concepts but omit
machine-specific source paths and authorship metadata. Provenance manifests,
generating scripts, encoded concept tensors, and model weights are not included.
These text files are optional; the TME-Context conditioner uses only numeric
measurements.
