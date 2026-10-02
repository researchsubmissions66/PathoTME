# Feature ablations at every supported shot

The user requested the subset-size and biological-group studies for every
PathoTME cohort and every shot. This extension runs PathoTME-LR at canonical
4/8/16/32/64 training examples per class wherever all five frozen outer folds
can support equally sized, patient-disjoint training and validation sets.

| Cohort | Supported shots | Limiting development class per fold |
|---|---|---|
| TCGA-NSCLC | 4, 8, 16, 32, 64 | At least 372 patients |
| TCGA-BRCA | 4, 8, 16, 32, 64 | At least 140 patients |
| TCGA-CRC | 4, 8, 16 | 58–59 patients; 32 shots would need 64 |
| TCGA-BLCA | 4, 8, 16, 32 | 98–99 patients; 64 shots would need 128 |

All eligible shot memberships nest within the original outer test folds.
NSCLC/BRCA replay the original max64 permutation, preserving existing 4/8/16
memberships. CRC/BLCA replay their original max16 permutation. For BLCA32,
the original 16 training and 16 validation patients per class remain in their
original phases; training gains unused permutation positions 32:48 and
validation gains 48:64. Existing splits are not overwritten or replaced.
This new phase-preserving extension is recorded separately. No patient reuse
or smaller validation set is used to manufacture an unsupported shot count.

The scientific selector, fits, preprocessing, C selection, biological groups,
random-mask generation and metrics are imported unchanged from the completed
16-shot ablation. RFE at fixed C=1 selects nested 8/16/32 feature sets using
only that fold/shot's training data. Ten random panels per size are shared
across matching panels, folds and shots. All 42 feature conditions receive
fresh fits on the same training samples and the original five-C validation
selection. All masks, selected C values and coefficients are saved before
any test feature is evaluated. The feature count axis is distinct from shots.

There are 17 supported cohort/shot settings, 85 cohort/shot/folds and 3,570
condition/fold results. All 840 completed 16-shot results are reused without
refitting. The 65 new cohort/shot/folds comprise 2,730 new retained LR fits
(13,650 C candidates), in one two-thread login-CPU process. No GPU/Slurm
allocation is created. Neural adapters and probability fusion are outside
this LR extension's execution scope.

Each new fold saves `design.json`, `selection.json`, `predictions.npz` and
`results.json`. Probability arrays are indexed by condition and phase-specific
slide/patient IDs; the exact axis identities and all coefficients are retained.
Compact per-fold storage avoids thousands of small shared-filesystem files.
The runner resumes verified completed folds under an exclusive controller
lock. Reused 16-shot result files are hash-bound references, never rewritten.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python run_allshots.py --prepare --parent-manifest "$ABLATION16_MANIFEST" --output "$PRIVATE_OUTPUT"
python run_allshots.py --manifest "$PRIVATE_OUTPUT/manifest.json" --execute
python report_allshots.py --output "$PRIVATE_OUTPUT"
```

The preflight validates every input/source binding, exact 16-shot replay,
existing 4/8-shot compatibility, every new shot count, patient disjointness,
cross-shot nesting, complete original test membership and training-only
imputation validity. Source files are immutable while running. Original
training, baseline, attribution and 16-shot ablation files are unchanged.

All outcomes are retained, including null/negative effects and random panels
that outperform selected subsets. Reports include five-fold means/SD, paired
fold group-removal differences, selection frequencies, calibration metrics
and PDF/PNG/SVG figures. SD and random-panel spread are not confidence
intervals. This is an exploratory study of predictive sufficiency/redundancy
within current core62/historical BRCA morph64 panels; it does not establish
original-panel optimality, causal necessity, a universal selected panel or
neural-adapter effects. Patient-level outputs stay outside Git; no results
or raw features are added to the public website.
