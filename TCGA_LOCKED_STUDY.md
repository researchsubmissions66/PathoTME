# Locked TCGA PathoTME follow-up, 9 September 2026

The user authorized launch for **both TCGA-NSCLC and TCGA-BRCA**, with no
CPTAC evaluation. The authored source is
`configs/tcga_locked_16shot_20260909.json`. Do not expand the study without
new authorization or modify source-hashed runtime while jobs are active.

The study tests quantitative TME-conditioned aggregation in native
ViLa/CLIP-RN50 and the partial PLIP-only MGPATH implementation, at 16 shots,
five patient-disjoint folds, with native, zero-TME, real-TME and shuffled-TME
arms. These guided models require TME measurements at inference. This is
not the separate CONCH visual-distillation study, nor upstream PLIP-G.

The common cohort is the intersection of original slides and fixed TME
coverage. All folds are generated with the unchanged PGVL split algorithm,
seed 1 and its existing maximum-shot-64 sampling offset. Only 16-shot splits
are published and launched. BRCA uses the existing morphology64 panel;
NSCLC uses the existing core62 panel. Cross-cohort tokenizer capacities
differ; within each cohort the three adapter arms have identical capacity.

Native results are reused only after config validation and exact ordered
train/validation/test membership checks, plus matching saved predictions.
Otherwise the unchanged native recipe is trained in a new output directory.
Previously fitted adapters are not mixed into the new five-fold result.
Historical checkpoint producer-source attribution remains limited to the
available original records. Feature headers establish encoder, geometry and
width with registry-bound representation; original extraction checkpoint
digests are unavailable and are explicitly disclosed.

Only adapter parameters train, with fixed Adam 1e-4, weight decay 1e-5 and
200 maximum epochs. ViLa uses strict validation-error improvement with
patience 20 after epoch 80; MGPATH uses validation macro-F1 with improving
ties and patience 100 after epoch 160. Adapter learning rate is distinct
from native MGPATH's 9e-6 recipe. All arms reset to the same seed and initial
conditioner parameters. Median imputation and standardization fit training
rows only. The shuffled control uses a frozen, label-blind, whole-row
bijection separately within train/validation/test, excluding same-patient
donors. This preserves row marginals and missingness; it is not a patient
block permutation for multi-slide test patients.

The primary endpoint is mean five-fold patient AUROC. Secondary endpoints,
three planned contrasts per model/cohort group, paired patient bootstrap,
and simultaneous intervals for the 12 primary contrasts are specified in
the protocol. Report every group and fold, including null/negative results.
This is exploratory follow-up: previous TCGA and CPTAC outcomes were already
inspected. Restricting scope to TCGA does not create an untouched validation.

Launch is bounded to **24 Slurm jobs**: four training-bag-only smokes followed
by twenty fold allocations. Each fold allocation runs its native model if
needed, then the three adapters sequentially. Every fold has an `afterok`
dependency on its own model/cohort smoke, plus a matching artifact gate at
runtime. Smokes test all three adapters and use historical checkpoints only
as structural fixtures, not as eligible experimental baselines. One A100,
8 CPUs and 48 GiB are requested per allocation; smokes have one hour and
folds four hours (ViLa) or six hours (MGPATH). The allocation ceiling is
104 GPU-hours, not a measured runtime prediction.

`scripts/prepare_locked_tcga.py` generates immutable split/config snapshots,
feature-header inventories, donor maps and `launch.json` under the shared
PathoTME-results root. `scripts/seal_locked_tcga.py` adds exact cached encoder
digests, tests and documentation in a fresh `launch.sealed.json`, preserving
the prepared parent contract. Launch uses this sealed file.
`scripts/launch_locked_tcga.py` defaults to dry-run;
explicit submission requires focused tests and exact Slurm test-only checks.
`submission.json` is the durable job-to-config lookup and captures every
returned ID immediately. Never relaunch blindly after a partial submission.
The adapter runner saves epoch optimizer/RNG state for an explicitly
authorized continuation; native training resumes only at fold boundaries.
