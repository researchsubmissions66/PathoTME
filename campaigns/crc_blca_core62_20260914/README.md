# CRC and BLCA common62 campaign

This isolated task extension adds TCGA-CRC and TCGA-BLCA to the seven-model
PathoTME study. It imports the existing model equations without modifying
the original PGVL or previously submitted PathoTME runtime files.

Each cohort has fourteen model/encoder groups: ViLa-MIL, MGPATH, FOCUS, MUSE,
HiVE-MIL, DyKo and MSCPT, each with PLIP and CLIP-RN50. Five patient-disjoint
outer folds use 16 training patients and 16 validation patients per class.
The patient folds, representative training/validation slides and whole-row
shuffled donor maps are identical across every group within a cohort.

CRC uses the frozen clinical adenocarcinoma NOS versus mucinous diagnosis;
BLCA uses non-papillary versus papillary urothelial carcinoma. These are
clinical-label task extensions, not independently reviewed slide diagnoses.
NOS does not establish absence of mucin, and papillary does not establish
low grade or non-invasive disease. Missing or other diagnoses are excluded
before splitting, with private exclusion evidence.

All arms consume the same cohort-specific prompt-bank bytes, annotated
GPT 6 ASTRA. Conversation authorship and the unverified endpoint/revision
identity are disclosed in the separate bank manifest. MUSE retrieval rows
are deterministic combinations of authored clauses. DyKo visual-concept
embeddings were encoded using the pinned TITAN text encoder on CPU.

The numeric TME panel is the same ordered core62 schema for both cohorts.
Its units, transforms and semantic grouping are fixed before training;
imputation and normalization use only the training fold. Each allocation
fits a new native baseline and then freezes that exact checkpoint for its
zero, actual and shuffled TME arms. Adapter initializations are matched.
Native and adapter objectives and validation selection follow each parent
architecture's existing study recipe. TME-guided inference requires TME.

The feature audit checks all six stores (two encoders at 5x/10x/20x), exact
slide UUID and encoder headers, payload finiteness, file stability, geometry,
cross-encoder coordinates and HiVE hierarchy. Historical feature-extraction
checkpoint digests were not supplied; header and supplied-encoder bindings
do not establish those missing digests. Private feature files stay outside Git.

`prepare_data.py` writes the final manifest, folds and donor maps only after
the feature audit passes. `prepare_campaign.py` freezes configurations,
prompt files, current native encoder weights and source hashes. The focused
CPU preflight checks real training-bag loading and small numerical probes.
Those probes are not cohort results or live GPU compatibility evidence.

`launch_campaign.py` plans by default. Submission requires a passing CPU
preflight and exact Slurm test-only requests. Each group has a training-bag
GPU smoke followed by five jobs depending on its successful completion.
The smoke exercises a native training step and all three adapter arms;
it is not a fitted paper baseline. Submission is recorded incrementally.

The complete bounded scope is 28 smokes plus 140 fold allocations across
both cohorts: 140 native fits and 420 adapter fits. There is no external
cohort evaluation in this campaign. BRCA common62 harmonization and proposed
TME-only, simple-fusion, feature-group and robustness experiments are outside
this launch. Existing TOP holds remain in effect.

Each completed adapter exports predictions, slide/patient metrics, its best
checkpoint, training history and a resumable optimizer/RNG checkpoint.
Completed native baselines are reused on retry. An interrupted native fold
cannot resume at the optimizer epoch because the inherited native trainer
does not save that state. All planned contrasts, including negative results
and calibration metrics, must be retained.
