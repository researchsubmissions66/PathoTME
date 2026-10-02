# PathoTME 4- and 8-shot follow-up

The user selected 4 and 8 shots for the completed ViLa/MGPATH study. The scope is
TCGA-NSCLC and TCGA-BRCA, PLIP and CLIP-RN50, five folds, with native, zero,
actual and shuffled conditions. This gives eight bundled smoke allocations and
eighty fold allocations. Each smoke checks both shot values; all ten matching
fold allocations require that smoke to succeed. Submission and current states
are recorded separately from preparation.

The shared campaign root is
`/path/to/shared/PathoTME-results/tcga_4_8shot_20260912_v1`.
`campaign.json` seals the validated plan; `submission.json` records each exact
Slurm command and ID. Do not repeat a partial or completed submission without
inspecting that ledger. The PGVL records are
`benchmarks/pathotme_4_8shot_prepare_20260912.json` and, after submission,
`benchmarks/pathotme_4_8shot_launch_20260912.json`.

## Scientific comparison

The original max-shot-64 sampling permutation is replayed against the frozen
common cohort and outer test patients. Exact ordered 16-shot membership must
reproduce before selecting 4 and 8 shots. The new training and validation
patient sets are nested; ordered test membership is unchanged. All methods and
encoders share each cohort/shot/fold's split and phase-local, label-blind donor
map. Normalizers are fitted using only the selected training patients. The
original core62/morph64 panels, prompts, model/encoder ports, selection rules and
hyperparameters are preserved.

Thirty existing BRCA native baseline folds have matching scientific recipes,
ordered split membership and validated predictions and checkpoints. These are
reused for ViLa/RN50, ViLa/PLIP and MGPATH/PLIP. Fifty other native fits are new;
all 240 zero/actual/shuffled adapters are new. Native NSCLC baselines must be
refitted because the frozen common cohort differs from the original PGVL
cohort. MGPATH/RN50 remains an explicit PathoTME encoder extension.

This is an exploratory follow-up after inspecting 16-shot results. Report every
model/cohort/encoder/shot group, all actual-versus-control comparisons and
null/negative outcomes, including NLL. The guided models require TME at
inference. The study does not establish WSI-only distillation or external-cohort
generalization.

## Runtime and validation

The isolated worker calls the unchanged native and cross-encoder training
loops. It applies the previously validated BRCA MGPATH indexing repair in its
own process and uses shot-aware validation for reused native baselines. New
configs are native YAML so scientific-notation optimizer values remain numeric.
No original queued runtime source or completed result is edited.

Completed 16-shot native fold-zero checkpoints are used only as structural
smoke fixtures. They are never substituted for a new shot's experimental native
baseline. Each smoke fits the new shot's training normalizer, checks native
zero-residual parity, zero/label independence, actual/shuffled decision-margin
response and finite nonzero TME gradients, and performs a finite adapter step
with a frozen base. The margin check avoids probability-saturation false
failures identified in the MUSE investigation. Both shots must pass before the
matching folds can start.

The focused CPU validation covers 80 exact runtime configs, twenty donor/split
and normalization groups, and four real method optimizer constructions with
tiny tensors. The 8,004 inherited feature-header audit entries have current
size/mtime checks. No broad test suite or new encoder-weight test is required;
the eight live GPU smokes gate execution. CPU validation and Slurm resource
checks do not establish live GPU success or research gains.

Each allocation requests one A100, eight CPUs and 48 GiB. Bundled smokes request
two hours. Fold limits use completed 16-shot wall times with shot scaling, a
minimum 35% retained cost, 50% margin and fifteen minutes overhead, rounded up
to thirty minutes with a one-hour minimum. The resulting limits are one hour
for ViLa and 1.5–2.5 hours for MGPATH. These are estimates, not guarantees.

The 72 TOP holds remain in place. This campaign does not restart v8/Astra or
add other models, 32/64-shot runs, or additional cohorts.
