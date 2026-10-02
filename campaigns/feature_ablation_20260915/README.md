# Within-panel feature ablations

This isolated research workflow investigates reduced subsets of the existing
PathoTME panels. It fits PathoTME-LR on all four cohorts at 16-shot/five-fold,
reusing exact parent memberships, fixed transforms, training-only imputation
and standardization, and validation selection. It adds no OpenTME measurements,
prompt banks, cohorts, Bayesian model, GPU requests, or neural training.

The subset-size curve contains 8, 16, 32 and the full 62/64 features. Nested
training-only recursive feature elimination (RFE) uses L2 logistic regression
at fixed C=1 and removes one smallest absolute standardized coefficient at a
time. Exact ties remove the lexicographically first feature. Ten deterministic
random orders provide nested controls at each size; the same random panel is
used across folds and cohorts with identical feature names. All masks are
retained, never selected by test outcomes. RFE is a standard selection method:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

Every retained feature set receives a fresh L2 logistic fit for each C in
0.001/0.01/0.1/1/10. Validation patient AUROC, NLL and smaller C determine the
model, matching the original baseline. RFE never receives validation/test
features, labels, saved parent coefficients, or post-hoc attribution scores.
The feature design is saved before validation processing, and each C selection
is saved before that condition's test predictions. There is no train/validation
refit. All 32 training and 32 validation slides are preserved. Full-panel refits
must reproduce the original baseline's selected C and both sets of saved
probabilities within 1e-9, or execution fails.

| Panel | Tissue architecture | Cell composition | Spatial interactions | Fourth family |
|---|---:|---:|---:|---|
| core62 (NSCLC/CRC/BLCA) | 14 | 28 | 16 | TLS: 4 |
| morph64 (historical BRCA) | 16 | 28 | 12 | Tissue geometry: 8 |

These four broad families exactly partition each panel. They aggregate the
existing finer tokenizer groups; group ablation trains each group alone and
re-trains after excluding it. Historical BRCA is not harmonized or relabeled.
These are model refits, separate from checkpoint-based feature replacement
attribution. A dropped group can be redundant without being biologically
irrelevant. Counts are 42 retained conditions per cohort/fold and 840 selected
LR fits, including 20 full-panel parity checks. The five-C search performs
4,200 candidate classifier fits plus 1,090 small selector fits. These are
serial CPU operations, not Slurm jobs or 840 GPU runs.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python run_ablation.py --prepare --parent-manifest "$BASELINE_MANIFEST" --output "$PRIVATE_OUTPUT"
python run_ablation.py --manifest "$PRIVATE_OUTPUT/manifest.json" --execute
python report.py --output "$PRIVATE_OUTPUT"
```

The runner verifies source/input hashes, patient-disjoint memberships and the
few-shot budget. It saves immutable panel/subset designs, coefficients, every
C candidate's validation metrics, selected C, predictions and complete test
metrics. The CPU controller has an exclusive lock and resumes compatible
completed conditions. Do not modify its source files while executing.

Reports include patient AUROC, NLL and Brier plus secondary metrics, paired
fold differences, selection frequencies, standalone PDF/PNG/SVG curves and
group figures. Fold SD is not a confidence interval. Random panels' spread
is descriptive and is not treated as independent patient replication.
Selection frequency is not a Bayesian posterior probability. All null and
negative outcomes remain visible; no globally best test subset is exported.

Outputs and patient IDs remain in private result storage, never Git or the
public website. The source is separate from all frozen parent training and
baseline files. Neural-adapter refits require an explicitly scoped follow-up
using the same fold-specific masks; this CPU runner does not establish neural
adapter performance. This exploratory study measures sufficiency/redundancy
within the selected panel, not optimality against excluded OpenTME features.
