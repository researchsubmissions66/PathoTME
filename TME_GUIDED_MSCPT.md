# MSCPT with measured TME conditioning

The seventh PathoTME architecture uses TCGA-NSCLC and TCGA-BRCA, PLIP and
CLIP-RN50, 16 shots, and five frozen common-cohort folds. Native, zero, actual,
and shuffled arms reuse the parent core62/morph64 panels and phase-local,
label-blind donor bijections. No cohort expansion or external evaluation.

## Architecture

The PLIP baseline retains MSCPT's vendored deep text prompt learner, graph
learners, three-branch CE, selector ensemble and top-k arithmetic. Both 5x and
20x preprojection bags use PLIP's frozen native 768-to-512 projection. RN50 is
an explicitly owned PathoTME port: its native 512D text transformer implements
the same shared and description-derived layer contexts, with final 1024D paired
text/image features and 1024D GCN layers. This is not a global PGVL allowlist
change or a claim of an upstream RN50 implementation.

Both ports use cached images. Deep visual prompting is inactive; neither port
is a full RGB MSCPT reproduction. Text retains native MSCPT 64-position
consumption: 2 shared plus 10 description context slots, native EOT pooling and
disclosed truncation. PLIP's existing embedding/position behavior is retained.
RN50 applies the corresponding native positional embeddings before context
replacement, and uses local causal masks without mutating transformer state.
All original prompt and selector file bytes are preserved. The NSCLC released
bank includes documented pathology/class-binding limitations; this addition
does not silently rewrite those descriptions. BRCA uses the existing separate
IDC/ILC bank, not the released High/Low task bank.

A single gated TME residual conditions the frozen low-scale descriptions and
the deep-prompted high-scale descriptions before affinity construction and
classification. One shared conditioner attends each description to the same
16 biological TME groups. Frozen layer-wise description caches and a separately
specified selector bank remain fixed. When the native selector falls back to
mean low-scale descriptions, those conditioned descriptions also affect its
selection. Cross-scale aggregation remains native. The inherited forward also
retains upstream affinity/graph quirks; no claim that every computed affinity
weight is consumed is made.

Adapter training freezes the complete native model and uses CE on its combined
label-independent inference logits. Standardization is fitted on training rows
only, using the project's exact feature transforms. Zero sets standardized TME
inputs to zero while retaining adapter capacity; shuffled uses the parent's
whole-row donors. All adapters share initialization and the native validation
criterion: maximize the mean of validation macro F1 and AUROC. Fifty epochs,
patience ten, no scheduler. Adapter epoch checkpoints retain optimizer and RNG
state; unfinished native folds may restart because PGVL saves final/best weights,
not an optimizer-resumable native training checkpoint.

## Execution and interpretation

Preparation and launching plan by default. The bounded campaign comprises four
encoder/cohort GPU smokes and twenty dependent fold allocations. Each fold fits
one new matched native control and three adapters (80 condition/fold evaluations,
not 80 Slurm jobs). A matching successful smoke is required in both Slurm and
the worker. Launch ledgers bind code, prompts, configs, folds, donor maps, TME
files, model weights and feature inventories. Feature geometry coverage reuses
the existing verified 5x/20x inventory with fresh size/mtime checks; no extra
patch selection or patient exclusion is introduced. Native PGVL metrics from
other cohorts/folds cannot substitute for these common-cohort native fits.

Resources: one A100 and four CPUs; PLIP uses 32 GiB and three hours per fold,
RN50 48 GiB and four hours. Smokes request 45 minutes. A related historical
CONCH MSCPT five-fold NSCLC run took 1,196 seconds and 23.624 GiB MaxRSS;
new-port native-plus-adapter runtimes remain to be measured. Preserve measured runtime evidence before
further reductions. TOP holds and existing jobs remain untouched.

The campaign is exploratory following inspection of prior results. Report all
four groups and twelve actual-minus-control contrasts, including null/negative
outcomes, uncertainty and calibration metrics. TME is required at inference.
No performance gain, statistical significance or full GPU compatibility follows
from implementation or CPU validation alone.

TME feature/group attribution and thumbnail region-ablation overlays use the
same saved-checkpoint interface as the other PathoTME architectures. Interpret
those as model sensitivity to reference replacements, not causal biology or
independent cell-level scores. No real MSCPT attribution result exists before
trained checkpoints complete.

Primary implementation reference: https://github.com/Hanminghao/MSCPT
