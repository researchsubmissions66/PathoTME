# TME feature redundancy

Standalone descriptive analysis of the existing fixed patient feature panels.
This implements clustered Spearman heatmaps, correlation groups, full PCA
variance spectra and effective linear dimension summaries for all four cohorts.
It does not add classifiers, feature selection, other feature-space analyses,
or training jobs. The same cohort matrix applies at every shot count.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python campaigns/feature_redundancy_20260915/compute.py --maps-manifest "$TME_MAP_MANIFEST" --allshot-manifest "$ALLSHOT_MANIFEST" --output "$REDUNDANCY_OUTPUT"
python campaigns/feature_redundancy_20260915/render_v2.py --output "$REDUNDANCY_OUTPUT"
```

Use the existing research Python environment with NumPy, pandas, SciPy,
scikit-learn and Matplotlib. Limit BLAS/OpenMP to two threads. Outputs must be
outside both code repositories. No GPU or new package is required.

Exact transformed patient matrices are replayed against the frozen TME-map
inputs. Primary Spearman correlations use those same imputed values. Pairwise
observed correlations and patient counts assess imputation sensitivity.
Complete linkage uses 1 − |rho|; a prespecified 0.85 cut produces groups whose
every pair meets the absolute-correlation threshold. Both signs are retained
in the figures. Thresholds 0.80 and 0.90 are descriptive sensitivity checks.

PCA uses standardized covariates, not rank correlations. Effective rank is the
exponentiated entropy of normalized covariance eigenvalues; participation ratio
is their inverse squared sum. These are summaries of effective linear dimension,
not counts of independent biological signals or a proposed minimum feature set.
Whole-cohort descriptions cannot select a replacement panel using test data.
Historical BRCA remains morph64; NSCLC/CRC/BLCA use core62.

Exports include six figure sets (vector PDF/SVG and 600-DPI PNG), exact named
correlations, PCA loadings/spectra, an aggregate-only offline viewer and a ZIP
package. Raw patient measurements and IDs never enter public-facing exports.
Private manifests record source/input hashes. Original models, results, maps,
and the public project website are unchanged.

Use `render_v2.py` for publication exports. It moves the detailed heatmap colorbar label above the bar to avoid clipping. The original source-bound renderer and its outputs are retained; no feature matrix, correlation, clustering, or PCA calculation changes.
