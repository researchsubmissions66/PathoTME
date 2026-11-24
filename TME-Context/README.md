<p align="center">
  <img src="assets/logo.png" alt="TME-Context logo" width="220">
</p>

<h1 align="center">TME-Context</h1>

<p align="center">
  <strong>Whole-slide learning, grounded in tumor biology.</strong><br>
  <em>Anonymous CVPR supplementary material</em>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#method">Method</a> ·
  <a href="#study-variants">Study variants</a> ·
  <a href="#numeric-baselines">Baselines</a> ·
  <a href="#supported-architectures">Architectures</a> ·
  <a href="#study-design">Study design</a> ·
  <a href="#biological-panels">Panels</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#repository-layout">Layout</a> ·
  <a href="#acknowledgements">Acknowledgements</a>
</p>

---

<a name="overview"></a>

## 🔬 Overview

**TME-Context** studies how quantitative **tumor microenvironment (TME)**
measurements contribute to whole-slide pathology prediction. Tissue
organization, cell composition, and spatial relationships provide biological
context for visual patch features. The study evaluates that context as a
numeric-only predictor, as a probability-fusion input, and as a learned
conditioner inside a frozen native model.

The central question is: **when does slide-specific biological context add
information beyond a native whole-slide model, a simple TME-only classifier,
and the capacity of an added conditioner?**

This small supplement contains the reusable biological panels and conditioner,
text prompt banks, five patient-disjoint BRCA split definitions, and an
executable **BRCA · MUSE · CLIP-RN50 · actual-TME** adapter recipe. The method
definitions below cover the broader comparison; the [run instructions](#getting-started)
identify what this folder executes. No whole-slide images, patch features,
measured TME rows, model weights, previous results, or test suite are bundled.

<a name="method"></a>

## 🧠 Method

All TME-based methods begin with the same ordered quantitative panel and
preprocessing fitted on training slides. The **TME-only baseline** fits an L2
logistic classifier on those measurements. **Probability fusion** combines
its output with a native model's probabilities. The **adapter** instead
arranges the measurements into **16 biological tokens**; trainable
cross-attention adds a gated residual to the native model's visual centers or
semantic queries while the native parameters remain frozen.

```mermaid
flowchart LR
    W["Whole-slide features"] --> N["Native model"]
    N --> PN["Native probabilities"]
    T["TME measurements"] --> P["Training-fold preprocessing"]
    P --> L["TME-only logistic regression"]
    L --> PL["TME probabilities"]
    PN --> F["Probability fusion"]
    PL --> F
    P --> B["16 biological tokens"]
    B --> C["Gated cross-attention"]
    N --> Q["Native centers or queries"]
    Q --> C
    C --> A["Frozen native aggregation"]
    A --> PA["Adapter probabilities"]
    classDef biological fill:#f3e8ff,stroke:#9333ea,color:#1e1b4b
    classDef frozen fill:#f8fafc,stroke:#94a3b8,color:#1e1b4b
    class T,P,L,PL,B,C,F biological
    class W,N,PN,Q,A,PA frozen
```

The TME-only classifier and actual-TME adapter need measurements at inference.
Fusion needs them whenever its TME weight is nonzero. Text prompts remain
separate from the numeric biological measurements.

<a name="study-variants"></a>

## 🧭 Study variants

| Variant | Definition | TME at inference? |
|---|---|---|
| **Native** | Matched underlying whole-slide model | No |
| **TME-only LR** | L2 logistic regression on the matching quantitative panel | Yes |
| **Fusion50** | Equal average of Native and TME-only LR class probabilities | Yes |
| **FusionVal** | Same probability mixture with its TME weight selected on validation patients | If selected weight is nonzero |
| **TME adapter** | Slide's own TME tokens condition a frozen native checkpoint | Yes |
| **Zero-TME adapter** | Same adapter with standardized TME values set to zero | Control input |
| **Shuffled-TME adapter** | Same adapter with another patient's row from the same split | Control input |

<a name="numeric-baselines"></a>

## 📊 Numeric and fusion baselines

The TME-only model fits an L2 logistic classifier on the selected panel. Its
fixed unit transforms, median imputation, and population standardization use
training slides only. Regularization is chosen from
`C = {0.001, 0.01, 0.1, 1, 10}` on validation-patient AUROC, then NLL, then
stronger regularization.
The selected model is evaluated on held-out patients without a train/validation
refit. One TME fit is shared when cohort, panel, fold, labels, and policy match;
it has no visual encoder.

Fusion combines saved native and TME-only class probabilities:
`p_fused = (1 - alpha) * p_native + alpha * p_TME-LR`.
Fusion50 fixes `alpha = 0.5`. FusionVal chooses from
`alpha = {0, 0.25, 0.5, 0.75, 1}` using validation-patient AUROC, then NLL,
then the smaller TME weight.
Both native-only (`alpha = 0`) and TME-only (`alpha = 1`) remain eligible.
Neither selection uses test labels. Native probabilities must come from the
matching saved checkpoint and the same held-out slides.

<a name="supported-architectures"></a>

## 🧩 Supported architectures

The wider study compares seven native architectures. TME-only LR does not use
an architecture; fusion uses each architecture's saved native probabilities.
For the adapter, the 16-token conditioner is inserted at the architecture's
native query location. This folder includes the reusable core and prompt
assets; the **MUSE/CLIP-RN50 BRCA actual-TME runner** is its one executable
architecture integration.

| Architecture | Visual scales | TME-conditioned representation |
|---|---|---|
| ViLa-MIL | 5× + 10× | Image-center queries before patch aggregation |
| MGPATH | 5× + 10× | Image-center queries before graph aggregation |
| FOCUS | 20× | Encoded semantic queries before patch selection and final attention |
| **MUSE** | **10×** | **Inference class semantics before sparse expert routing** |
| HiVE-MIL | 5× + 20× | Hierarchical text nodes before filtering and graph construction |
| DyKo | 20× | Class queries after concept retrieval |
| MSCPT | 5× + 20× | Multiscale text graph representations |

The other native architecture implementations are outside this small folder.
Their text inputs are retained in [the prompt banks](prompt_banks/README.md).

<a name="study-design"></a>

## 🧪 Study design

The wider comparison uses **16 shots per class**, **five patient-disjoint
folds**, paired **PLIP / CLIP-RN50** encoders, and matched cohort manifests.
The bundled protocol is the five-fold BRCA IDC-versus-ILC subset with
CLIP-RN50 feature filenames.

| Wider-study cohort | Classification task | Slides | Patients | Panel |
|---|---|---:|---:|---|
| TCGA-NSCLC | LUAD vs. LUSC | 1,041 | 944 | core62 |
| TCGA-BRCA | IDC vs. ILC | 960 | 900 | morph64; shared core62 in matched extensions |
| TCGA-CRC | Adenocarcinoma NOS vs. mucinous | 544 | 536 | core62 |
| TCGA-BLCA | Non-papillary vs. papillary | 451 | 380 | core62 |

The quantitative panel and native checkpoint are matched within each
cohort/encoder/fold comparison. Fixed transforms precede median imputation
and population standardization fitted on training slides. TME-only LR
regularization and FusionVal's weight use validation patients only; no
training/validation refit follows selection. The actual, zero, and shuffled
adapter arms use the same capacity and selection rules. Shuffling stays
within a split and uses another patient's TME row.

Held-out patient probabilities average that patient's slide probabilities.
The wider analysis reports AUROC, NLL, Brier score, accuracy, macro-F1, and
calibration alongside slide-level metrics. This folder contains no previous
performance scores or predictions.

<a name="biological-panels"></a>

## 🌿 Biological panels

The fixed panels use quantitative measurements from
[Aignostics / OpenTME](https://huggingface.co/datasets/Aignostics/OpenTME).

| Panel | Measurements | Biological token groups | Use here |
|---|---:|---:|---|
| `core62` | 62 | 5 tissue/TLS + 11 cell/spatial | Reusable schema |
| `morph64` | 64 | 6 tissue/morphology + 10 cell/spatial | BRCA MUSE run |

The [panel schema](tme_context/panels.json) lists every feature in its required
order, fixed percentage and `log1p` transforms, and token membership.
`core62` expects its four lymphoid presence/log-count variables to have been
derived; `morph64` expects two region-density variables to have been derived.
Missing measurements may be `NaN`; the fold's training rows determine median
imputation and scaling. These fixed definitions encode biological hypotheses,
not an optimized feature selection result.

<a name="getting-started"></a>

## 🚀 Getting started

The study variants above describe the full comparison. This folder executes
the **actual-TME BRCA MUSE/CLIP-RN50 adapter**. The [TME-only LR and fusion
runners](https://github.com/researchsubmissions66/PathoTME/tree/main/campaigns/baselines_20260914),
other architecture integrations, and control arms are in the larger research
repository and require their own inputs.

Use Python 3.10 or newer. Install from inside this folder:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[hdf5]'
```

The BRCA MUSE runner needs **real inputs supplied separately**:

1. A directory of 10×, 224-pixel CLIP-RN50 HDF5 patch bags. Exact basenames
   are in [`splits/brca/16shot/`](splits/README.md). Each file has `features`
   shaped `[patches, 1024]` and matching `coords`.
2. A quantitative BRCA CSV with `slide_id` followed by the exact raw
   `morph64` columns in the [panel schema](tme_context/panels.json).
3. A trained **native MUSE CLIP-RN50** `foldN_best.pt` state dictionary and
   its `config.json`. The config may be supplied with `--native-config` if it
   is not beside the checkpoint.

Point the variables at your existing files and choose a **new output directory
outside this source folder**:

```bash
FEATURE_ROOT=/path/to/clip_rn50_h5_files
TME_CSV=/path/to/opentme_brca_morph64_v1.csv
NATIVE_CHECKPOINT=/path/to/native/fold0_best.pt
RUN_OUTPUT=/path/to/new/brca_muse_fold0

python -m tme_context.run_muse \
  --feature-root "$FEATURE_ROOT" \
  --tme-csv "$TME_CSV" \
  --native-checkpoint "$NATIVE_CHECKPOINT" \
  --fold 0 \
  --device cuda:0 \
  --output-dir "$RUN_OUTPUT"
```

Use `--device cpu` if CUDA is unavailable; full training will be slower.
Change both the fold number and native checkpoint for folds 1–4. The runner
checks the native cohort, class order, encoder, and feature width; keeps the
native model frozen; and trains only the conditioner. It uses Adam with learning
rate 0.0001 and weight decay 0.00001, at most 200 epochs, validation-error
selection, and patience 20 after epoch 80.

New `adapter.pt`, `history.csv`, `predictions.csv`, and `summary.json` files
are written **only** to `RUN_OUTPUT`. The summary includes slide- and
patient-level metrics. The trained native checkpoint and feature tensors are
not distributed with this supplement.

<a name="repository-layout"></a>

## 🗂️ Repository layout

```text
TME-Context/
├── assets/logo.png          # Project illustration
├── tme_context/             # Panel, conditioner, data loader, MUSE runner
├── splits/brca/16shot/      # Five patient-disjoint fold definitions
├── prompt_banks/            # Native text inputs for seven architectures
├── pyproject.toml           # Minimal Python package
└── README.md
```

<a name="acknowledgements"></a>

## 🙏 Acknowledgements

We thank **Aignostics and the OpenTME contributors** for the quantitative
microenvironment profiles and tools used by the wider study.
We also acknowledge the researchers who developed ViLa-MIL, MGPATH, FOCUS,
MUSE, HiVE-MIL, DyKo, and MSCPT; cite the corresponding original method papers
when using those architectures.

- **GitHub:** [TME Studio — the official OpenTME exploration toolkit](https://github.com/aignostics/tme-studio).
- **Hugging Face:** [Aignostics / OpenTME dataset](https://huggingface.co/datasets/Aignostics/OpenTME).
- **Reference:** [Galama et al., *OpenTME: An Open Dataset of AI-powered H&E Tumor Microenvironment Profiles from TCGA* (2026)](https://arxiv.org/abs/2604.12075).

Raw OpenTME measurements and WSI feature arrays are not redistributed here.
Obtain them separately and follow their access and use terms.

## Additional research tools

The [companion full-study repository](https://github.com/researchsubmissions66/PathoTME)
also contains other model integrations, attribution utilities, and the wider
experimental setup. Those tools and their results are outside this minimal
folder.

## Text prompt banks

The [prompt-bank index](prompt_banks/README.md) covers ViLa-MIL, MGPATH,
FOCUS, MUSE, HiVE-MIL, DyKo, and MSCPT across NSCLC, BRCA, CRC, and BLCA.
Text banks are included as source assets; numeric TME conditioning does not
read them. Encoded tensors and model weights are not included.
