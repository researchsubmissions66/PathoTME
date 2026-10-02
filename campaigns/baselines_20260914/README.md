# Matched PathoTME baselines

This addition measures how much the numeric TME panel predicts by itself and
whether simple probability fusion explains the gain from a TME conditioner.
It supports ViLa, MGPATH, FOCUS, MUSE, HiVE-MIL, DyKo and MSCPT, PLIP and
CLIP-RN50, in the main 16-shot/five-fold TCGA study. Existing training files,
checkpoints, prompt banks and Slurm allocations are unchanged.

The three comparator outputs use the [canonical method names](../../README.md#-method-names):

- **PathoTME-LR:** L2 logistic regression on the matching panel.
- **PathoTME-Fusion50:** equal average of native and TME class probabilities.
- **PathoTME-FusionVal:** the same average with TME weight selected
  from 0, 0.25, 0.5, 0.75 and 1. The endpoints remain eligible.

Fixed panel transforms precede training-fold median imputation and population
standardization; no missingness indicators or additional features are added.
Regularization C is selected from 0.001, 0.01, 0.1, 1 and 10 on validation
patient AUROC, then NLL, then stronger regularization. Fusion chooses its weight
on validation patient AUROC, then NLL, then the smaller TME weight. The selected
TME model is independent of architecture and encoder. Identical panel bytes,
split memberships, labels and policies share one fit. No training/validation
refit follows selection. The classifier and preprocessing use float64 CPU
arithmetic; neural adapter buffers use float32 with the same preprocessing
equations. This is an exploratory addition after prior outcomes were inspected.

Saved test probabilities are reused exactly. Missing native validation
predictions are exported from the exact saved native checkpoint on CPU, with
strict checkpoint loading and one saved test prediction replayed as a
compatibility check. This replay does not tune anything or select a checkpoint.
All expected slide/patient IDs and labels must match; missing slides are errors.
Patient probabilities average all held-out slides. Reports include AUROC, NLL,
Brier, accuracy, balanced accuracy, macro-F1 and ECE, at slide and patient levels.
Selections, coefficients, preprocessing, predictions and hashes are saved.

Historical NSCLC uses core62 and BRCA uses morph64. CRC and BLCA use common62.
The baseline always matches its corresponding neural experiment; this addition
does not harmonize or relabel BRCA's existing panel. No raw feature data are
distributed with this code.

Use `run_baselines.py --prepare --launch <launch.json> ... --output <private-results>`
to create an immutable manifest. Later launch arguments supersede the same
cohort/method/encoder/fold in earlier retry history. Inspect the manifest with
`--manifest <manifest.json>`. Add `--execute --tme-only` for the small linear fits,
or `--execute --watch` to run available native prediction exports serially and
wait for the remaining native fits. Limit OMP, MKL and OpenBLAS to two CPU
threads. There are no GPU or Slurm requests. Failed exports remain explicit in
`status.json`; missing native results are recorded as `waiting_native`.

The fusion comparator tests simple multimodal prediction combination. It is
not a learned embedding-concatenation network or a capacity-matched neural
adapter. Those would be separate ablations.

These are standard baseline algorithms under the PathoTME study umbrella.
Presentation names do not rename stored condition keys or change the frozen runtime.
