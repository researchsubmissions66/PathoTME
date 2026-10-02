# DyKo with quantitative TME guidance

DyKo is the sixth PathoTME architecture. The addition retains TCGA-NSCLC and
TCGA-BRCA, PLIP and CLIP-RN50, 16-shot/five-fold, native/zero/actual/shuffled.
The parent common-cohort manifests, exact ordered splits, donor maps and
core62/morph64 panels remain unchanged. One 20x bag is used per slide.

## Architecture and encoder boundaries

The native model is PGVL's unchanged equation-level DyKo implementation.
PLIP reuses its existing paired-feature-context extension: the frozen native
768-to-512 image projection, paired text tower, shared learned feature context,
and learned 768-to-512 concept bridge. RN50 is a separately owned PathoTME port
with its frozen paired 1024D image/text space and learned 768-to-1024 concept
bridge. Neither condition is an upstream TITAN reproduction. No PGVL allowlist
or queued runtime file is modified by these PathoTME additions.

The class prompt learner returns two class-ordered queries. A temporary forward
hook adds the existing gated TME residual immediately before both native visual
and semantic cross-attention streams. It is removed even if a forward fails.
The native WAKI clustering and concept retrieval have already run at that point:
TME does not change the concept tensor, selected concepts, visual/semantic
adapters, structural probabilities or clustering procedure. The continuous dual
attention supplies conditioner gradients; discrete retrieval is not trained by
TME. Each process owns one fold model, with no concurrent forwards on that model.

The baseline is frozen and held in eval mode while the adapter learns. The
conditioner uses the same sixteen biological groups, hidden width 128, four
heads, dropout 0.1 and gate 0.1 as the other single-scale PathoTME adapters.
All groups are available to both class queries; no handpicked mapping ties a
particular TME feature to a particular phrase. TME is required at inference.
Group attention is a diagnostic, not causal or word-level attribution.

## Knowledge and class text

NSCLC preserves its exact released two-row class CSV and 1000x768 TITAN concept
tensor. Runtime reorders the CSV's LUSC/LUAD rows into benchmark LUAD/LUSC order.
BRCA has no released DyKo bank. At the user's explicit request, the new
`text_prompts/dyko_brca_v1/knowledge_bank.json` contains 64 individually authored
H&E morphology concepts in eight descriptive categories. It includes tumor
architecture, discohesion, cytology, stroma, inflammation, vessels and background.
These categories are catalog metadata, not class labels or TME token groups.
All 64 concepts are available to every visual prototype. No duplicated padding
or Cartesian text combinations inflate the bank to 1000 rows.

Existing PGVL breast vocabulary and public RCPath/SEER morphology guidance inform
the selection. Per-entry source categories and all local source hashes are saved.
IHC/receptor/mutation status and clinical outcomes are excluded. No patient text,
images, split membership, TME values or performance results are used to author or
select concepts. Independent pathology review has not been performed.
See the bank README for linked sources and limitations. This bank is a research
extension, not a validated diagnostic vocabulary or an upstream DyKo asset.

The exact pinned TITAN text tower at revision
`dac6773d9961cfc75503440676ff157a2c6e8d2e` encodes the bank once on CPU into
normalized float32 64x768 vectors, preserving the concept interface. Its encoding
record binds model/source/tensor hashes and native token IDs. Both paired BRCA
encoders and all four conditions use the same frozen tensor. The class CSV is
separate: it preserves the two existing FOCUS high-resolution IDC/ILC descriptions
exactly and explicitly binds their order. All class descriptions encode in full
within the native 77-position text context; no token prefix is added by the
feature-context learner. Original prompts elsewhere remain unchanged.

The 64-versus-1000 concept count differs across cohorts and is disclosed. Compare
TME effects within each cohort/encoder; the study does not establish bank-size
parity, optimal concept selection or an upstream BRCA reproduction.

## Controls, fitting and outputs

Each fold fits its normalizer using its 32 training slides only. Zero sets the
standardized measurements to zero, retaining learned group identities. Shuffled
uses the exact parent phase-local, label-blind, whole-row donor bijections,
excluding the recipient's patient. These are not patient-block permutations.
Actual uses the recipient's measurements. Conditioners are reseeded after base
loading so all three arms have identical initialization.

Native training retains Adam 1e-3, weight decay 1e-5, four-slide gradient
accumulation, CE plus unit-weight structural KL, 1000 epochs maximum, no scheduler,
and strict validation-loss improvement with patience forty after epoch zero.
Adapter training retains the PathoTME Adam 1e-4/weight-decay 1e-5 per-slide CE
update, with the same epoch limit, strict validation-CE selection and patience.
Structural KL is constant with respect to the query conditioner and is excluded
from its objective and validation monitor. Labels enter only CE and not prediction.

All twenty native baselines are new. Existing NSCLC PLIP PGVL results use different
train, validation and test memberships; they are not a valid common-cohort
comparator. BRCA and RN50 conditions are new extensions. Twenty native fits plus
sixty adapters give eighty condition/fold evaluations, allocated as four smokes
and twenty folds. Each fold requires its matching successful smoke and verifies
that smoke's launch and artifact identities. Adapter checkpoints preserve
optimizer/RNG/epoch state for resumption; unfinished native folds restart under
the existing PGVL trainer. Completed identity-validated outputs are retained.

Requests are one A100, eight CPUs and 48 GiB. One-hour smokes and provisional
six-hour folds retain the full native training recipe; limits are estimates,
not completion guarantees. Native and adapter wall times are recorded.
The launch tool plans by default and requires source-bound focused validation
and all 24 Slurm resource checks before submission. The user's standing launch
instruction applies after those gates; preserve TOP holds and inspect the
submission ledger before any recovery.

## Verification and reporting

Eight focused CPU checks cover all four cohort/width combinations, native
zero-residual logits and attention parity, invariant structural outputs, finite
TME-to-decision-margin gradients, parameter updates with a frozen baseline,
zero controls, intervention cleanup, checkpoint restoration, train-only
normalization, donor validity, numeric configs and the bounded launch graph.
Actual cached PLIP/RN50 constructors and the real BRCA concept tensor are checked
separately with synthetic image bags. GPU smokes use training bags only, retain
native CE+KL prompt gradients and inspect pre-softmax margins to avoid the prior
MUSE saturated-probability false failure. None of these structural checks is a
research performance result.

Single-scale feature inventories reuse the source-bound FOCUS header audits with
current size/mtime verification for all 4002 files. They retain 1041 NSCLC slides
(944 patients) and 960 BRCA slides (900 patients). Extraction checkpoint digests
remain historically unavailable. Original header evidence is retained.

Report all four groups, all folds and all twelve actual-minus-native/zero/shuffled
contrasts, including negative or null results and NLL/calibration. Six architectures
produce 72 planned within-cohort/encoder contrasts. Previous outcomes were seen
before this addition, so all new conclusions remain exploratory. The generic
PGVL text attribution command does not load standalone PathoTME adapter checkpoints.
