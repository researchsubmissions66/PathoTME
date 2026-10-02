# PathoTME: both TCGA cohorts, both encoders

The user's correction on 9 September 2026 expands the study to TCGA-NSCLC
and TCGA-BRCA, each with ViLa and MGPATH under both PLIP and CLIP-RN50.
The source is `configs/tcga_cross_encoders_16shot_20260909.json`.

The original sealed batch, jobs 21933657–21933680, supplies ViLa/CLIP-RN50
and MGPATH/PLIP. This separately prepared batch adds ViLa/PLIP and
MGPATH/CLIP-RN50. It uses the parent's exact common-cohort manifests,
16-shot/five-fold splits, fixed TME panels, donor permutations and adapter
recipe. No original runtime or running-job source is edited. These additions
bring the full design to eight model/cohort/encoder groups, 160 fold/arm
comparisons and **48 Slurm jobs total**, including eight smokes. Each fold
allocation runs its baseline if needed and the three adapters sequentially.

ViLa/PLIP reuses the registered paired-feature implementation. Its frozen
native PLIP projection maps cached 768D features to the paired 512D space
before TME conditioning; the native checkpoint and conditioner see the same
projected bags. The old RN50-only TME batch bridge is not used for raw PLIP
features. Eligible BRCA PLIP baselines are reused only after config, split
and prediction identity checks. NSCLC needs five new matched PLIP baselines.

MGPATH/CLIP-RN50 is a **new PathoTME-owned encoder extension**, not a registered
PGVL native configuration or the upstream PLIP-G model. It declares its own
strict RN50 contract and uses the existing graph, 64 centers, four prompt
views, pooling and Sinkhorn implementation at CLIP's paired 1024D boundary.
Four learned feature-context vectors replace PLIP's token-context learner,
following the existing paired-feature MGPATH ports. Full original prompt
bank bytes/order are preserved. Its explicit native CLIP tokenizer policy
consumes at most 75 BPE content tokens plus SOT/EOT, analogous to native
PLIP's existing 77-position prefix policy. Original token counts, consumed
token IDs, tokenizer code and vocabulary digests are recorded. This context
replacement and tokenizer difference limit interpretation as a pure encoder
swap; within each pairing the native/zero/real/shuffled comparison is matched.

The RN50 MGPATH baseline uses the unchanged native trainer fold loop through
the task-owned adapter, with the existing Adam 9e-6/200-epoch validation-F1
recipe. No allowlist or existing benchmark source is changed. Its smoke
constructs a fresh RN50 graph model, checks one training-bag update, and
uses that state only as a structural fixture for the three conditioners.
The five-fold experimental baselines start independently from their fixed
seeds. ViLa/PLIP smokes use historical checkpoints only as structural fixtures.

The copied adapter training loop differs from the frozen parent only in its
explicit method factory. Train-only normalization, identical initial seeds,
checkpoint selection, epoch continuation, native residual equivalence,
frozen-base gradients, test membership checks and attention role exports
are retained. Every added fold depends on a successful model/cohort smoke
and verifies its report at runtime. New GPU checks do not attest convergence.

The combined analysis now includes **24 prespecified primary contrasts**
(real minus native, zero and shuffled within each of eight groups), with
ordinary 95% and Bonferroni simultaneous 99.7916666667% patient-bootstrap
intervals. This prospective combined-analysis amendment is separate from the
original batch's immutable 12-contrast subset specification. Primary outcome
is mean five-fold patient AUROC; secondary outcomes and reporting rules remain
as in the parent. Report every group and negative/null result. Earlier TCGA
and CPTAC outcomes were inspected, so this remains exploratory. Guided models
require TME at inference; this is separate from WSI-only visual distillation.

New outputs, immutable plans, config snapshots, test evidence, exact dry-run,
Slurm test-only checks and the per-ID submission ledger belong under shared
`PathoTME-results/tcga_cross_encoders_16shot_20260909_v1/`. Preserve every launch
ledger and source-hashed file while jobs are queued or running. This addition
does not resume the stopped PGVL v8/Astra prompt campaign.
