# HiVE-MIL with quantitative TME guidance

The user selected HiVE-MIL as PathoTME's fifth architecture on September 12,
2026. The addition retains TCGA-NSCLC and TCGA-BRCA, PLIP and CLIP-RN50,
16-shot/five-fold and native/zero/actual/shuffled conditions. All original
common-cohort manifests, ordered folds, donor maps and core62/morph64 panels
are reused. No CPTAC or additional TCGA cohort is added by this implementation.

## Paired encoder ports

The new PathoTME-owned ports replace the CONCH-specific text constructor with
frozen native PLIP/RN50 text towers. Sixteen shared learned prefix tokens keep
HiVE's normal(0, 0.02) initialization and its original `X ... X term : explanation`
format. Token positions 1–16 are replaced by context parameters. Each paired
tower consumes its native 77-position prefix/EOT sequence. Longer descriptions
retain their source wording on disk, with explicit truncation at encoding time;
the original and consumed token counts/IDs are recorded in `prompt_audit.json`.
Runtime token hashes must match preparation. This is an encoder extension,
not an upstream CONCH reproduction or a new PGVL allowlist entry.

PLIP restores its frozen native 768-to-512 visual projection. RN50 uses its
1024-dimensional paired image/text space. The heterogeneous graph widths follow
the paired space; graph topology, filtering, hierarchical text contrastive loss
(HTCL), original logit-scale arithmetic and pooling are inherited directly from
the vendored `CustomCLIP` implementation at commit
`fa5ccec1a99db510e9add85b318e6241acb1aecd`. Native training learns only the
token context and graph prompt learner with CE + 0.5 HTCL. Frozen text towers
remain in eval mode. Original 4 coarse plus 12 fine descriptions per class,
class order, bank hashes and all wording remain unchanged.

## TME adapter and controls

A shared gated residual conditions each of the 32 text nodes immediately after
text encoding and before text-guided filtering and graph construction. The
existing 128-dimensional/four-head conditioner attends to all 16 biological
groups for each node, at either magnification. It uses dropout 0.1 and initial
gate 0.1. It does not introduce new manually selected input features or map
individual TME measurements to specific prompt phrases.

The native baseline is frozen and kept in eval mode during adapter training.
Adapters use a single CE update per training slide through the label-free
inference branch. Labels only enter CE; native HTCL is excluded from adapter
optimization. Continuous text-node and graph computation supplies gradients;
discrete graph edges, filtering masks and top-k indices have no gradients.
The scoped text intervention is removed on success or error. Each process
owns one fold model, with no concurrent forwards through the same tower.

Zero replaces standardized TME values with zero while preserving learned group
identities. Actual uses the recipient's TME. Shuffled uses the exact parent
phase-local, label-blind whole-row donor maps, excluding the recipient's patient.
These maps are not patient-block permutations. Imputation/mean/std are fitted
only on the 32 training slides. Conditioner initialization is reseeded after
base model loading so all three arms start identically. Actual TME is required
at inference; this model is not WSI-only distillation.

## Data and execution

NSCLC retains 1,041 slides/944 patients; BRCA retains 960 slides/900 patients.
Both visual magnifications are required for every existing study slide. The
CPU audit checks 8,004 feature headers and 4,002 slide/encoder hierarchies.
It validates paired coordinate frames and magnification/224-pixel geometry,
rejects duplicate coordinates or more than sixteen children, and records exact
parent/child map hashes. An x-sorted search used in the audit is checked against
the existing runtime compiler's stored row order.

Training uses the unchanged PGVL HiVE loader. It retains parents with at least
one child and zero-pads partial groups to sixteen children, preserving native
graph behavior. Empty parents are dropped within a slide; slides are not
removed from the common cohort. The returned validity mask is metadata, as in
the current PGVL HiVE implementation; the native graph still consumes padded
zero rows. Feature size/mtime must remain unchanged after audit. Historical
extraction checkpoint digests remain unavailable and are disclosed.

Every paired native baseline is new: no PLIP/RN50 HiVE baseline exists for reuse.
The bounded campaign contains four smokes and twenty fold allocations. Twenty
native fits plus sixty adapters yield eighty condition/fold evaluations. Every
fold depends on successful completion of its matching cohort/encoder smoke and
also checks that smoke's artifact and launch identities at runtime.

Native and adapter training use Adam 1e-4, weight decay 1e-5, no scheduler,
fifty epochs maximum and strict validation-error improvement, patience ten
after epoch zero. Adapter resume checkpoints preserve optimizer/RNG/epoch state;
an interrupted native fit restarts its unfinished fold under the existing PGVL
trainer. Completed native/adapter outputs are retained after identity checks.

Requests are one A100, eight CPUs and 48 GiB, with one-hour smokes and provisional
three-hour fold limits. Historical CONCH native five-fold jobs 21426323/21426320
took 25m10s NSCLC / 37m11s BRCA. These do not establish PLIP/RN50 wall times;
the new runner records smoke steps, native stage and adapter epoch times.

## Verification and reporting

Eight focused CPU tests cover both graph widths and both TME panels, exact
zero-residual native graph/output parity, finite TME gradients, frozen baseline
parameters, controls and checkpoint restoration, intervention cleanup, training-only
statistics, native hierarchy order/padding, numeric YAML and the 24-job boundary.
Separate cached-tower checks compare token-port outputs against actual native
PLIP/RN50 encoding and exercise prompt gradients. These checks are not live GPU
smokes or research results. GPU smokes use fold-zero training bags only and
never export their one-step fixtures as experimental baselines.

Adapter predictions include patient/slide metrics, TME attention averaged over
the four or twelve text nodes within each class and scale, and retained-parent
counts. Full per-text-node TME attention is available in detailed forward output.
These are attention diagnostics, not causal or word/token prediction attribution.
The general PGVL text-attribution CLI does not load standalone PathoTME adapters.

Report all four HiVE groups and all twelve actual-minus-native/zero/shuffled
contrasts, including null/negative results and NLL/calibration. Five architectures
yield sixty planned contrasts. Prior outcomes were inspected before selecting
this architecture, so the addition is exploratory.

```bash
python scripts/prepare_hive_tcga.py --output /path/to/shared/PathoTME-results/hive_tcga_16shot_20260912_v1 --hierarchy-audit AUDIT_DIR --prepare
python scripts/launch_hive_tcga.py --launch OUTPUT/launch.json --validation OUTPUT/validation.json
```

The launcher plans by default. `--test-only` checks exactly 24 requests;
`--submit` requires passing source-bound validation and those resource checks.
The user's standing launch instruction applies after these gates. Inspect the
durable submission ledger before any continuation. Existing PGVL, FOCUS and
MUSE runtime files are unchanged. TOP holds remain in place.
