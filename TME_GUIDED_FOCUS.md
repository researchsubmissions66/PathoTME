# FOCUS with quantitative TME guidance

This PathoTME-owned extension adds FOCUS to TCGA-NSCLC and TCGA-BRCA with
PLIP and CLIP-RN50, 16 shots per class and five patient-disjoint folds. It is
an exploratory addition after earlier PathoTME outcomes were inspected.
Implementation and preparation do not authorize job submission.

## Model and controls

The native single **20x** patch bag enters the existing FOCUS feature encoder,
adaptive token selection, spatial compression, cross-attention and classifier.
A 128-dimensional, four-head TME conditioner adds a gated residual to FOCUS's
encoded high-resolution semantic class queries. Those queries participate in
patch relevance ranking and final visual cross-attention. Discrete top-k and
compression decisions have no gradient; the final attention supplies the
differentiable training path. The original FOCUS equations are imported without
editing the PGVL implementation.

The NSCLC core62 panel and BRCA brca_morph64_v1 panel are unchanged. All 16
semantic groups are available jointly to each query because FOCUS has one
visual scale. The tokenizer's low/high group labels describe biological roles,
not two image bags. This is a declared architectural extension; it is not a
feature-selection experiment or a claim that these hand-selected panels are optimal.

Each fold compares the matched native baseline, zero-TME, actual-TME and
shuffled-TME adapters. Native parameters, including learned prompt context,
remain frozen and in evaluation mode during adapter training. The three
adapters have identical initial parameters and capacity. Zero-TME replaces
standardized values with zeros but keeps learnable group identities. Shuffled
TME uses the parent's phase-local, label-blind, whole-row donor bijections,
excluding the recipient's patient; this is not a patient-block permutation.
Median imputation and population mean/std use only the 32 training slides.
Inference requires the slide's TME measurements for the real arm; this is
distinct from WSI-only visual distillation.

PLIP uses the existing **paired_feature_context_v1** FOCUS condition, with the
checkpoint's frozen 768-to-512 visual projection and paired text tower. It is
distinct from the separately trained PLIP token16 variant. RN50 is a new
**pathotme_focus_clip_rn50_feature_context_v1** port with 1024-dimensional shared
features and the matching text tower. Both use one trainable feature context
in the baseline. Neither reproduces upstream CONCH token prompting. The RN50
port is local to PathoTME; the PGVL encoder allowlist is unchanged.

The original two-scale prompt CSVs remain byte-preserved, with their ordered
class/scale hashes and provenance (upstream NSCLC, generated BRCA). FOCUS uses
the high-resolution half. Preparation checks the full bank against both native
tokenizers and rejects overflow rather than truncating or rewriting it.

## Fixed experiment and provenance

`configs/focus_tcga_16shot_20260911.json` is the source of truth for this addition.
Preparation derives YAMLs from the registered PGVL FOCUS recipe and protocol
feature sources, then reuses the parent study's exact common-cohort manifests,
ordered split memberships and donor files. Both feature stores must cover all
parent slides; missing files cannot silently change the cohort. HDF5 headers
must match encoder, width, 20x magnification, 224-pixel patches and coordinates.
Historical extraction checkpoint digests are absent; registry binding and
header identity do not independently attest the original extraction weights.
Runtime paired weights, code, source protocols, TME assets, split files, prompt
banks and config snapshots are SHA256-bound in `launch.json`; audited feature
files are additionally checked by size/mtime at execution.

Existing PLIP baselines are reused only with matching configuration, complete
metrics/checkpoints, exact train/validation/test membership and complete test
predictions. The reuse audit records exclusions. Otherwise a new matched
baseline is trained. Historical producer code is not independently re-attested.
Never use token16 or a different cohort split as a replacement baseline.

Native FOCUS and adapter selection use validation error, strict improvement,
patience 20 and stopping after epoch 40, at most 200 epochs. Adapter optimization
is Adam, learning rate 1e-4, weight decay 1e-5; no scheduler. Adapter checkpoints
resume at epoch boundaries with optimizer/RNG state. An interrupted new native
baseline restarts its unfinished fold, following the existing PGVL trainer;
completed native folds and completed adapter arms are retained.

The bounded plan has **4 smoke allocations and 20 fold allocations**, each fold
running its native baseline if needed and the three adapters. There are 80
arm/fold evaluations, not 80 separate Slurm allocations. Six-hour fold limits
are provisional and must be profiled; completion within an allocation is not
guaranteed. Do not reduce them using the runtime of an adapter-only fold.

Every fold requires its own cohort/encoder smoke to pass. The smoke uses only
a fold-zero training bag, checks the baseline and all three adapters, verifies
native equivalence with a zero residual, TME dependence/zero independence,
finite gradients and frozen parameters. A smoke fixture is never a paper
baseline. CPU tests are not a live GPU smoke.

## Commands

Use `/path/to/shared/envs/pgvl-gym/bin/python` from the PGVL root with
offline model settings. No command below submits a job.

```bash
python /path/to/PathoTME/scripts/prepare_focus_tcga.py --output /path/to/shared/PathoTME-results/focus_tcga_16shot_20260911_v1 --prepare
python -m pytest -q /path/to/PathoTME/tests/test_focus_tme.py -p no:cacheprovider
python /path/to/PathoTME/scripts/launch_focus_tcga.py --launch /path/to/shared/PathoTME-results/focus_tcga_16shot_20260911_v1/launch.json --validation /path/to/shared/PathoTME-results/focus_tcga_16shot_20260911_v1/validation.json
```

`--test-only` performs Slurm resource validation without submission. A later,
explicitly authorized `--submit` additionally requires that exact check and a
source/launch-bound passing validation record. The launcher writes a durable
submission ledger after each response, refuses duplicates, and attaches every
fold to its matching successful-smoke dependency. A partial submission requires
inspection rather than blindly rerunning the launcher.

`run_focus_tcga.py` is read-only by default. Execution requires `--execute` in
a GPU Slurm allocation. Supply `--cohort nsclc|brca --encoder plip|clip-rn50
--fold 0..4`, and `--smoke` only for fold zero. Runtime output includes native
metrics/predictions, per-arm configs, best/resume checkpoints, epoch wall times,
slide and patient metrics, class-query TME group attention and a fold completion
record binding every result to its native checkpoint and launch. Detailed
evaluation also exposes patch attention with original patch-row indices.
Attention weights are model explanations, not causal effects or word attribution.

Report all four FOCUS groups and all folds, including null and negative results.
Patient AUROC is the primary contrast; retain other metrics, including NLL.
Combining this addition with ViLa/MGPATH expands the family from 24 to 36
actual-minus-native/zero/shuffled contrasts. Do not reuse the narrower family's
simultaneous intervals for that combined claim.
