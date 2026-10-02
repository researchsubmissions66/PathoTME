# Data inventory

## CPTAC-BRCA external histology (2026-09-07)

Public NCI PDC `PDC000120` clinical and biospecimen annotations were downloaded
and content-pinned with the study catalog (version 2). Raw responses and the
frozen external cohort are in `${PATHOTME_DATA_ROOT}/external/cptac_brca/v1/`.

- Clinical candidates: 94 unambiguous IDC and 11 ILC patients. One apparent
  ductal diagnosis has a conflicting `8500/2` in-situ code and is excluded.
- Final exact TCIA/PDC primary-tumor matches with both CONCH scales:
  **98 slides / 98 patients (88 IDC, 10 ILC)**.
- `clinical_labels.csv`, `slide_audit.csv` and `metadata.json` retain all
  exclusions, source queries, study version, raw hashes and mapping policy.
- `feature_audit.json` verifies all 196 HDF5 headers: CONCH-v1, width 512,
  low 5x/512px and high 10x/256px. Historical extraction checkpoint hashes
  were not recorded; this is disclosed in the external contract.
- No CPTAC TME data were downloaded or fabricated. The external pathway
  evaluates exported WSI-only models. See [CONTROLLED_VILA.md](CONTROLLED_VILA.md).

## OpenTME

### Breast quantitative panels (2026-09-07)

The local breast release contains 1,125 slide rows in each of its five tables.
Both `opentme_brca_morph64_v1.csv` and `opentme_brca_shared62_v1.csv` are
materialized under the private `PathoTME-data/processed` root, with adjacent
source-hash metadata. They match all 960 BRCA benchmark slides (900 patients).
The 64-value primary panel differs from the lung panel; the 62-value reference
reuses its categories with breast values only. See [the BRCA contract](TME_GUIDED_BRCA.md)
for ordered token groups, derivations and the reference-only TLS assumption.
Missing individual measurements remain distinct from missing slide rows.

- **Source:** `Aignostics/OpenTME` on Hugging Face
- **Access:** gated, non-commercial academic research
- **Pinned revision:** `9262bc0cd0cd7774d0f7bcbfe6ae5f4665898b25`
- **Release inventory:** 28,203 files, approximately 6.16 GB
- **Core analysis payload:** 45 CSV/settings/documentation files,
  approximately 582 MB
- **Local source directory:**
  `/path/to/shared/PathoTME-data/OpenTME`
- **Project link:** `data/OpenTME`

The data directory is intentionally ignored and must never be committed or
redistributed. Users must obtain individual access from Aignostics, comply
with the OpenTME license, and comply with the original TCGA data-use policies.

### Local status (2026-09-03)

The core quantitative payload is complete and validated against the pinned
Hugging Face inventory:

- 45/45 expected core files are present;
- 40 CSV files cover bladder, breast, colorectal, liver, lung, pancreatic,
  prostate, and stomach cancer;
- total validated core size is 581,630,941 bytes;
- no files are missing and no file-size mismatches were found.

The initial all-file attempt downloaded a partial thumbnail cache before the
Hugging Face API rate limit was reached. The optional thumbnail collection is
therefore not yet complete; this does not affect the quantitative analysis
payload. Resume the same pinned transfer with:

```bash
/path/to/shared/envs/pgvl-gym/bin/python \
  scripts/download_opentme.py
```

The command above verifies or restores the complete quantitative analysis
payload while excluding per-slide visualization thumbnails. Fetch or resume
the thumbnails separately only when they are needed:

```bash
/path/to/shared/envs/pgvl-gym/bin/python \
  scripts/download_opentme.py --include-thumbnails --max-workers 2
```

The low worker count is intentional: the thumbnail release contains tens of
thousands of small files, and higher concurrency can exceed Hugging Face's API
request quota even though the byte volume is modest.
