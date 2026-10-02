# Standalone TME teacher: controlled BRCA follow-up

## Scientific question and status

Can a **standalone quantitative subtype classifier** teach a WSI-only student
information that visual-only distillation does not provide? This is not the
old `kd_real` condition: that teacher was a CONCH–ViLa model plus a TME adapter.

Implemented for BRCA IDC/ILC, 16-shot, five patient-disjoint folds. No jobs are
submitted by implementation or by a dry-run. No improvement is claimed.
The authoritative contract is
`configs/vila_conch_brca_tme_teacher_16shot.yaml`.

### Authorized launch: 2026-09-08

- Ledger: `standalone_teacher_launch_20260908.json`, including exact commands,
  resolved input provenance, identities, and all Slurm job IDs.
- Smoke: `21893080`, one A100, 20 minutes; running at 20:04 CDT.
- New student folds: `21893081`–`21893090`, one A100 and 45 minutes each;
  all ten pending on `afterok:21893080` at that snapshot.
- Each fold allocation also evaluates its exported WSI-only student on CPTAC.
  Ten completed control folds are reused, not resubmitted.
- Existing results and TOP holds remain untouched. These are submission/status
  records, not evidence of successful training or improved performance.

## Four matched conditions

| Condition | Teacher | Execution |
| --- | --- | --- |
| `student_ce` | None; classification-only continuation | Reuse five validated completed controls |
| `kd_visual` | Frozen original CONCH–ViLa | Reuse five validated completed controls |
| `kd_tme_only` | Completed standalone TME logistic regression | Five new student folds |
| `kd_tme_only_shuffled` | Same classifier family fitted on shuffled TME rows | Five new student folds |

The student remains the paired-CONCH ViLa **extension**, not an upstream RN50
reproduction. ABMIL is not introduced here: changing both teacher and student
would prevent this experiment from isolating the teacher change. An ABMIL
student would require a separately named experiment and matched baselines.

The existing control contract supplies the exact source checkpoint, prompt
bank, class order, feature spaces and scales, splits, seeds, optimizer,
learning rate, weight decay, stopping and checkpoint-selection rules.
Students retain the same unused auxiliary-head scaffold to match initialization;
it receives no gradients and is removed at export. Classification CE + KL uses
temperature 2 and KL weight 1, retaining temperature-squared scaling. There is
no extra labeled or unlabeled patient data. This is continuation of an existing
source-validation-selected checkpoint, not training a new student from scratch.

## Teacher construction

The real teacher reuses the identity-checked `brca_morph64_v1` 64-feature
logistic-regression checkpoint. Its fixed panel transforms, training-fold median
imputation and standardization remain unchanged. Its C was selected on source
validation error over the existing grid, with first-grid-entry tie breaking.

The shuffled teacher uses exactly that pipeline/grid. A deterministic,
label-blind, split-local slide-row bijection assigns each slide another
patient's entire feature vector **before fitting and before validation**.
Train and validation have separate donor permutations; they never exchange
rows. Labels stay attached to their original slide. This is a refitted shuffled
teacher, not merely shuffled outputs from a real-data teacher. Missing-value
patterns travel with each entire row. Imputation and scaling are still fitted
on training rows only. One permutation per fold is an attribution control,
not a permutation significance test.

Ordered classifier classes must be `[0, 1]` (IDC, ILC). Binary decision score
`z` is converted to symmetric logits `[-z/2, z/2]`, preserving the classifier's
probabilities without clipping or applying the distillation temperature twice.
The teacher is frozen; saved logits contain **exactly the few-shot train slide
IDs**. These are in-sample teacher predictions, not cross-fitted predictions;
the limitation must remain disclosed. No teacher logits are requested during
student validation or test inference. The exported state is native ViLa only.

## Source-validation complementarity diagnostic

Before student optimization, record:

- both models correct / both wrong;
- teacher correct where the original visual classifier is wrong;
- visual classifier correct where the teacher is wrong;
- individual validation accuracies and the diagnostic either-correct oracle.

Visual predictions are reused from the completed fusion experiment's native
validation columns, with exact slide/patient/label checks. The real teacher is
not selected based on disagreement; the diagnostic does not automatically
change configurations, select a winning fold, or gate submission. The oracle
is not an achievable classifier score. The 32-slide validation set is small.
The runner transforms quantitative targets for train/validation only, not test.

## External evaluation and interpretation

The existing `eval_wsi_only.py` accepts the new native-only exports without any
code change. Optional CPTAC inference follows each new fold in the same GPU
allocation. The existing two controls' CPTAC predictions are reused in reports.
There are ten new external evaluations if requested, not a new external cohort.

Neither target labels nor target TME are used to fit, select or adapt these
students. The frozen 98-patient CPTAC manifest and prior geometry/checkpoint
attestation limitations remain in force. **Both TCGA and CPTAC outcomes have
already been inspected**, and this follow-up was motivated by those findings;
it is exploratory reuse, not a fresh confirmatory external test. Do not tune
hyperparameters against CPTAC. The same 98 patients across source folds are
not independent replications; report each source model and the fixed equal-weight
ensemble separately. Retain all controls, including negative findings.

## Commands

From the PathoTME root, with the configured environment:

```bash
python scripts/launch_standalone_tme_teacher.py \
  --with-external --report /tmp/pathotme_standalone_teacher_dry.json

python scripts/run_standalone_tme_teacher.py \
  --config configs/vila_conch_brca_tme_teacher_16shot.yaml \
  --fold 0 --condition kd_tme_only \
  --teacher-diagnostics /tmp/pathotme_teacher_fold0.json

# Only after explicit launch authorization:
python scripts/launch_standalone_tme_teacher.py --submit \
  --with-external --report standalone_teacher_launch.json

python scripts/report_standalone_tme_teacher.py \
  --output /tmp/pathotme_standalone_teacher_report --bootstrap 1000
```

Dry-run: **10 reused controls, 10 new fold jobs + one two-branch smoke job**.
Each new fold requests one A100, six CPUs, 24 GiB and 45 minutes; smoke requests
20 minutes. Jobs use offline caches and depend on successful smoke. If only
external inference remains, the launcher schedules that alone. Completed
source folds are skipped only after exact identity and artifact verification;
partial outputs are preserved for explicit recovery, never silently overwritten.

## Provenance and artifacts

New results live under `${PATHOTME_RESULTS_ROOT}/vila_standalone_tme_teacher/`
`brca/16shot/brca_morph64_v1/v1/`. No old results, run contracts, or source-hashed
runners are rewritten. Readiness verifies old controls, teachers and fusion
artifacts; stale/missing references block readiness rather than trigger reruns.

Each new result includes resolved source configuration, inherited training
contract, exact control/teacher identities, source hashes, donor maps,
`teacher_classifier.joblib`, `teacher_diagnostics.json`,
`teacher_validation_predictions.csv`, `training_teacher_logits.pt`, training
history with separate KD loss, validation predictions, test predictions, and
native-only inference weights. Teacher artifacts are provenance/training assets,
not deployment inputs. Source code and prediction reports may be shared subject
to applicable terms; do not redistribute private OpenTME data or derived teacher
artifacts without checking the data-use terms.

The reporting script checks artifact identities, pairs exact patients, and
compares real-teacher students with CE, visual-teacher and shuffled-teacher
controls. Patient-bootstrap intervals are exploratory, unadjusted for multiple
comparisons and conditional on fitted models. Fold means and pooled OOF scores
are different summaries and must not be interchanged.

## Verification (2026-09-08)

- Thirteen distinct regression tests passed: six existing controlled-preflight
  tests, six new dependency-light tests, and one synthetic end-to-end test.
- The end-to-end test ran all four student conditions on tiny CPU bags. The
  new runner's CE and visual-KD controls produced **bit-for-bit identical native
  weights after one epoch** to the immutable original runner. Both new teacher
  conditions completed; caches contained only train IDs and exports contained
  no auxiliary head, conditioner, or standardizer. This is not a convergence test.
- Real-data preflight validated all five folds and their completed comparators.
  The final dry-run is `/tmp/pathotme_standalone_teacher_20260908_dry.json`:
  ten reused controls, ten new folds plus smoke, and ten optional external
  evaluations. No jobs were submitted.
- All ten real/shuffled CPU teacher diagnostics completed on the actual fold
  data and produced exactly 32 finite training-logit pairs each. Diagnostics:
  `/tmp/pathotme_standalone_teacher_diagnostics_20260908.json`.
- Syntax, Ruff F checks, and source-hash agreement with the final plan passed.
  Planning imports do not load Torch, Transformers or sklearn. Installed
  dependency loading from Lustre was slow; the numerical test completed using
  the configured environment outside the sandbox with its `lib` directory on
  `LD_LIBRARY_PATH` and pytest plugin autoload disabled.

The real-feature GPU smoke has not run. It remains a required dependency before
any full student job in the launcher.
