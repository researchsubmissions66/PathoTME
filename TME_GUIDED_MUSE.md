# MUSE with quantitative TME guidance

The user selected MUSE as the fourth PathoTME architecture on September 12,
2026, authorizing this addition to TCGA-NSCLC and TCGA-BRCA with PLIP and
CLIP-RN50, 16-shot and five patient-disjoint folds. This is exploratory after
the earlier ViLa/MGPATH outcomes were inspected. No CPTAC evaluation is added.

## Model and controls

The registered MUSE paired encoder extensions consume one **10x** patch bag.
PLIP supplies 768-dimensional preprojection features to MUSE's learned visual
adapter and 512-dimensional paired text features; RN50 uses 1024-dimensional
image/text features. Both keep eight experts, two selected experts, eight
attention heads and the top-20-percent patch aggregation rule. These ports use
frozen encoded class semantics plus a learned feature context, replacing native
CONCH token prompting. They are not upstream CONCH MUSE reproductions.

Native baselines retain the existing SFSE/SMMO recipe: every training slide
retrieves and exhausts its 20 semantic views with 20 optimizer updates. Class
descriptions, original CSV order and the registered EOT-preserving 77-token
description-encoding policy are unchanged. `prompt_audit.json` records the
original token counts, consumed IDs, class bindings and truncated-row counts.
Inference class-template strings are separately audited. Source strings are
never rewritten; the long description bank serves native training only.

The TME adapter adds a gated residual to MUSE's inference class-semantic vectors
**before sparse expert routing**. The existing visual adapter, experts, patch
aggregation and classifier are frozen and remain in evaluation mode, including
during adapter training. The conditioner reuses the existing 128-dimensional,
four-head attention over the 16 semantic TME groups, with dropout 0.1 and an
initial residual gate of 0.1. All groups are available jointly for this single
visual scale. Discrete expert IDs and top-patch selection have no gradients;
selected expert weights and visual cross-attention provide differentiable paths.

Adapters follow the shared PathoTME classification-only protocol: one CE update
per training slide using MUSE's label-free inference branch. They do not run
ground-truth-conditioned description retrieval. Labels enter the loss only and
cannot change predictions. This adapter training is intentionally distinct from
the full native SMMO training described above. Inference requires TME for the
actual arm; these are not WSI-only distillation models.

Each fold has native, zero-TME, actual-TME and shuffled-TME results. Native
parameters are identical for all three adapters, and conditioner initialization
is reset immediately after native model loading so encoder cache warmup cannot
alter the matched initial state. Zero replaces standardized values with zeros
while preserving learned group identities. Shuffled TME reuses the parent's
phase-local, label-blind whole-row donor bijections, excluding the recipient's
patient. It is not a patient-block permutation. Imputation and population
mean/std are fitted only to the 32 training slides.

## Fixed data and execution

`configs/muse_tcga_16shot_20260912.json` fixes the experiment. The parent study's
common-cohort manifests, exact ordered folds, core62/morph64 panels and donor
maps are reused. NSCLC has 1,041 slides/944 patients, BRCA 960/900. The two
original NSCLC slides without matching TME are `TCGA-44-7661-01Z-00-DX1` and
`TCGA-50-6590-01Z-00-DX1`; the previously trained 1,043-slide PGVL baselines have
different split memberships and cannot serve as matched controls.

Historical baselines are eligible only with matching configuration, all three
ordered split memberships, complete metrics/checkpoints and exact held-out
prediction membership. Preparation records reuse/exclusion reasons, rather
than selecting by outcome. Otherwise the same native recipe is trained on the
common folds. No existing PGVL or queued FOCUS runtime file is edited.

Preparation audits all 10x feature headers, encoder identities, widths and
224-pixel geometry. It binds source/config/prompt/model/split assets by SHA256
and feature files by size/mtime after header audit. Historical extraction
checkpoint digests remain unavailable; header/registry identity does not
independently attest extraction weights. Files named in a submitted launch
must not change while its jobs are active.

Native and adapter selection use strict validation-error improvement, patience
20 after epoch 80, with at most 200 epochs. Both use Adam 1e-4, weight decay
1e-5 and no scheduler. YAML exports preserve numeric types through the actual
training loader. Adapters resume with optimizer/RNG state at epoch boundaries;
an interrupted native fold restarts its unfinished fold under the PGVL trainer.
Completed baselines and adapter arms are retained after identity checks.

The bounded campaign is **four smoke allocations and twenty fold allocations**.
Each fold allocation runs its matched native baseline if needed, then its
three adapters. This is 80 condition/fold evaluations, not 80 Slurm jobs.
Every fold has an `afterok` dependency on its cohort/encoder smoke and verifies
that smoke's matching artifact identity at runtime. No TOP holds are changed.

Resources are one A100, eight CPUs and 48 GiB per allocation, with one-hour
smokes and provisional two-hour fold limits. Historical native NSCLC paired
PLIP/RN50 jobs 21823608/21823601 completed all five 16-shot folds in 64m04s and
85m41s respectively. These observations inform the limit but do not guarantee
completion of a new native-plus-three-adapter allocation. Native stage and
adapter epoch wall times are recorded for any later resource adjustment.

## Verification and outputs

Eight focused CPU checks cover the real MUSE model at both embedding widths
and both TME panels, zero-residual native parity, finite TME gradients, frozen
base parameters, zero-input independence, label/retrieval independence,
checkpoint restoration, train-only statistics and the bounded launch plan.
These synthetic checks and saved-checkpoint construction are not live GPU
smokes or research results. Each GPU smoke runs a complete 20-update native
training-bag step and all three adapter arms on training data only. It tests
native equivalence, finite gradients, TME dependence and matching initialization.
Its checkpoint is a structural fixture, never an experimental baseline.

Results include native and adapter predictions, slide/patient metrics, best and
resume checkpoints, group attention per class-semantic query, sparse expert
weights and measured wall times. Patch attention retains original bag rows.
Attention and expert weights are model diagnostics, not causal effects or
word/token attribution. PGVL's existing text-attribution command does not yet
load standalone PathoTME adapter checkpoints.

Report all four MUSE groups and all twelve actual-minus-native/zero/shuffled
contrasts, including null/negative outcomes and secondary metrics such as NLL.
The combined ViLa/MGPATH/FOCUS/MUSE family has 48 contrasts; uncertainty claims
must account for that family rather than reuse intervals for a narrower study.

```bash
python /path/to/PathoTME/scripts/prepare_muse_tcga.py --output /path/to/shared/PathoTME-results/muse_tcga_16shot_20260912_v1 --prepare
python -m pytest -q /path/to/PathoTME/tests/test_muse_tme.py -p no:cacheprovider
python /path/to/PathoTME/scripts/launch_muse_tcga.py --launch /path/to/shared/PathoTME-results/muse_tcga_16shot_20260912_v1/launch.json --validation /path/to/shared/PathoTME-results/muse_tcga_16shot_20260912_v1/validation.json
```

The launcher plans by default. `--test-only` validates the exact resource
requests; `--submit` requires passing source-bound validation and those checks.
Its durable ledger records each returned job ID and refuses duplicate or
uninspected partial submissions. Execution is GPU-Slurm-only and opt-in through
`run_muse_tcga.py --execute`; preparation and syntax checks never start training.
