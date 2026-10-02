# TME feature maps and held-out prediction overlays

This standalone research workflow creates publication figures and an offline
interactive viewer from the fixed PathoTME panels and saved predictions.
It does not train models or modify any original checkpoint, split or result.

Use a private results directory outside both code repositories. The existing
all-shot ablation manifest supplies exact patient memberships and LR fits;
its bound baseline manifest supplies native/adapter/fusion recipe identities.

```bash
python campaigns/tme_maps_20260915/prepare_maps.py --allshots "$ALLSHOT_MANIFEST" --output "$TME_MAP_OUTPUT"
python campaigns/tme_maps_20260915/embed_maps.py --output "$TME_MAP_OUTPUT"
python campaigns/tme_maps_20260915/render_maps_v2.py --output "$TME_MAP_OUTPUT"
```

Preparation/rendering use NumPy, pandas, SciPy, scikit-learn and Matplotlib.
Embedding can run in a separate existing environment with umap-learn; it reads
only the label-free covariate NPZ and the prespecified parameter policy.
No package upgrade of the training environment is needed. Set BLAS/OpenMP to
two threads and NUMBA_NUM_THREADS to two; UMAP itself uses one worker.

The embedding is descriptive and transductive: fixed-transformed slide values
are averaged per patient, then imputed/standardized across cohort patients
for display only. No labels or probabilities are supplied to UMAP/PCA.
Prediction overlays use the exact saved held-out fold for each patient.
All shots/methods share the same cohort coordinates. Cohort maps are separate,
and historical BRCA uses morph64 while the others use core62.

Native and adapter probabilities are verified against their completion hashes;
legacy plans without encoder metadata use the audited baseline task's explicit
encoder binding. Plot metrics consistently average saved probabilities in
float64. Small historical float32 aggregation differences must replay exactly
in float32, and are disclosed in the private manifest. This does not alter the
original metric reports or training results.

The main paired figure is prespecified as ViLa/CLIP-RN50/16-shot/Adapter, using
only fully completed cohorts; the viewer retains other available architectures,
shots, fusion and shuffled/zero controls, with explicit partial coverage.
Corrected/worsened counts use a fixed 0.5 decision threshold. Probability changes
and AUROC measure different properties. No seeds, plot bounds or parameters are
selected to exaggerate class separation or prediction gains.

Outputs include vector PDF/SVG, 600-DPI PNG, PCA controls, LR shot panels, and
an offline HTML viewer. Raw covariates, patient IDs, manifests and per-patient
arrays remain private. The HTML carries anonymous point numbers and saved
probabilities, without private paths or raw measurements. Nothing is added to
the public project website. Local preflight source history is ignored by Git.
