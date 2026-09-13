# PathoTME

**Anonymous research submission**

PathoTME studies quantitative tumor-microenvironment (TME) conditioning for
whole-slide pathology models. A trainable cross-attention conditioner encodes
biological measurements into 16 semantic tokens and adds a gated residual to
an architecture's visual centers or encoded semantic queries. Native model
parameters remain frozen during adaptation. Guided models use TME at inference.

## Study design

The primary study uses 16 shots per class, five patient-disjoint folds, and
PLIP and CLIP-RN50 features for TCGA-NSCLC (LUAD/LUSC) and TCGA-BRCA (IDC/ILC).
ViLa-MIL, MGPATH, FOCUS, MUSE, HiVE-MIL, and DyKo share matched native, zero-TME,
actual-TME, and shuffled-TME conditions. The NSCLC core62 and BRCA morph64 panels
are fixed biological hypotheses; imputation and normalization use training
slides only. The paired encoder ports include architecture-specific extensions.

This repository presents the method and study design. Experimental results
are not included on the project website.

## Code and data

- `pathotme/`: biological panels, conditioners, and architecture integration.
- `configs/`: study configurations.
- `scripts/`: data preparation and research entry points.
- `text_prompts/`: separate text assets and their provenance.
- `tests/`: focused research checks.
- `docs/`: static project website and study protocol.

The models use the PGVL-Gym implementations and paired visual/text encoders.
Quantitative phenotypes are available through
[Aignostics / OpenTME](https://huggingface.co/datasets/Aignostics/OpenTME).
Datasets, model weights, patient measurements, and local experiment outputs
are not distributed here.

Local installation paths and scheduler accounts have been replaced with
`/path/to/...` and `YOUR_SLURM_ACCOUNT` for anonymous review. Configure these
for your environment before running the research scripts. Archived launch
bindings refer to the original local experiment setup; they are not executable
submission records for this anonymized copy.

## Project website

Serve the standalone website with:

```bash
python3 -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open `http://localhost:8000`. The site includes an interactive architecture
explorer, both biological panels, and the matched study design. No build step
or training dependencies are needed to view it.
