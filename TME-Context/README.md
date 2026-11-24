# TME-Context

TME-Context is an anonymous, compact implementation of quantitative tumor
microenvironment (TME) conditioning for whole-slide models. This supplement
contains the 62- and 64-measurement panel schemas, fold-fitted preprocessing,
16 semantic TME tokens, gated cross-attention, text prompt banks, and five
patient-disjoint BRCA 16-shot split definitions. It contains no images, patch
features, TME values, model weights, prior results, or tests.

## Install

Use Python 3.10 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[hdf5]'
```

## Run BRCA MUSE with CLIP-RN50

The executable recipe trains the **actual-TME** conditioner on a frozen native
MUSE CLIP-RN50 model. Supply three real inputs:

1. A directory of 10×, 224-pixel CLIP-RN50 HDF5 patch bags. The filenames are
   listed in [`splits/brca/16shot/`](splits/README.md); each file must contain
   `features` with shape `[patches, 1024]` and aligned `coords`.
2. A quantitative BRCA CSV with `slide_id` followed by the exact raw `morph64`
   columns in [`tme_context/panels.json`](tme_context/panels.json).
3. The trained native MUSE CLIP-RN50 `foldN_best.pt` state dictionary and its
   `config.json`. Keep `config.json` beside the checkpoint or pass
   `--native-config` explicitly. The runner checks the native cohort, class
   order, encoder, and feature width before training.

Set the paths below to those existing inputs and a **new output directory
outside this folder**:

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

Use `--device cpu` if CUDA is unavailable; a full fold will be slower. Change
both `--fold` and the native checkpoint for folds 1–4. The runner fits TME
imputation and standardization on training slides only, keeps the native MUSE
model frozen, and trains only the conditioner. It uses Adam, learning rate
0.0001, weight decay 0.00001, at most 200 epochs, validation error selection,
and patience 20 after epoch 80. Outputs are written only to `RUN_OUTPUT`:
`adapter.pt`, `history.csv`, `predictions.csv`, and `summary.json`. The summary
includes slide- and patient-level metrics; patient probabilities are the mean
of that patient's slide probabilities.

The runner needs a native checkpoint because native MUSE training includes a
large text encoder and is outside this small supplement. The code for the
numeric TME module is in [`tme_context/core.py`](tme_context/core.py). The
[`prompt_banks/`](prompt_banks/README.md) directory includes the text inputs
for ViLa-MIL, MGPATH, FOCUS, MUSE, HiVE-MIL, DyKo, and MSCPT. Their native
model runners and the native, zero-TME, and shuffled-TME comparison arms are
not bundled here.
