# BRCA-specific TME-guided ViLa-MIL

## Scope and status

This is a **PathoTME extension**, not an upstream ViLa-MIL component. The task
is TCGA-BRCA **IDC (0) versus ILC (1)**, not PAM50, receptor status or grading.
The implementation preserves the existing native CLIP-RN50 5x/10x bags,
validation-selected ViLa checkpoints, class order and generated BRCA two-scale
prompt bank. No lung prompts, lung slide measurements, lung fitted statistics
or lung adapter checkpoints are substituted.

The first authored contracts cover **16-shot, folds 0–4**. Existing baselines
are reused. The raw breast release contains 1,125 slides; all 960 benchmark
slides (900 patients; 771 IDC and 189 ILC slides) have matching quantitative
rows. Native features are available at both scales for all 960. All 25 native
folds across 4/8/16/32/64 shots were config-valid at the availability audit;
only 16-shot PathoTME extensions are currently registered here.

Implementation and static readiness do not demonstrate performance gains.
The subsequent user-authorized [launch](BRCA_LAUNCH_STATUS.md) submitted two
native-equivalence smokes and 20 fold jobs dependent on their panel's smoke.
No smoke or training success is implied by submission.

### Validation completed (2026-09-07)

- **26 targeted tests passed**: feature contracts, quantitative transforms,
  cohort/order/duplicate rejection, strict preflight, native/zero-residual
  equivalence, real-versus-zero behavior, adapter-only gradients, checkpoint
  round trips and TME-only training-statistic isolation.
- **30/30 planned conditions passed static preflight** with 30 distinct run
  identities, recorded in `brca_vila_plan_20260907.json`.
- Syntax checks and Ruff undefined-name/unused-import checks passed.
- The six existing PathoTME source files hashed by the queued MGPATH campaign
  still match its launch ledger.
- **Not yet performed:** a live BRCA checkpoint/feature GPU smoke, training,
  convergence assessment or paired performance comparison.

## Why a different panel?

The primary hypothesis concerns carcinoma/stromal organization and their
spatial context. Breast research documents stromal differences between lobular
and ductal/NST carcinomas, including fibroblast-related expression patterns:
[Park et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27469595/) and
[Characterisation of the Stromal Microenvironment in Lobular Breast Cancer](https://pubmed.ncbi.nlm.nih.gov/35205651/).
These studies motivate **candidate biological roles**, not a claim that the
selected OpenTME variables reproduce molecular assays or predict IDC/ILC.

OpenTME tissue-region eccentricity, solidity and roundness are coarse segmented
region geometry. They are **not direct measurements of single-file invasion,
cell adhesion, E-cadherin loss, CAF molecular subtype or duct formation**.
Native visual bags remain necessary to test complementary information.

The feature list is authored from these roles and released column definitions,
without ranking quantitative features against class labels or held-out scores.
Native BRCA performance was previously inspected, so this is a development
benchmark, not a newly untouched external validation study.

## Two separately identified panels

### Primary: `brca_morph64_v1`

| Scale | Token groups | Values | Tokens |
| --- | --- | ---: | ---: |
| Low | Global carcinoma, stroma, epithelium, necrosis and vessel proportions | 5 | 1 |
| Low | Carcinoma and stromal region geometry | 8 | 2 |
| Low | Tumor-core and inner/outer margin extent | 3 | 1 |
| Low | Core and inner-margin carcinoma/stroma/epithelium/necrosis composition | 8 | 2 |
| High | Seven cell types: percentage and density in core and inner margin | 28 | 7 |
| High | Fibroblast, lymphocyte and macrophage proximity to carcinoma | 12 | 3 |
| **Total** | | **64** | **16 (6 low / 10 high)** |

Each geometry token contains average eccentricity, average solidity (the
release spells its column `SOLIDICITY`), largest-region roundness, and
segmented region count per mm² of valid tissue. The last quantity is derived
as `region_count * 1e6 / ABSOLUTE_AREA_VALID_TISSUE` (source area in µm²).
Invalid/zero denominators remain missing, not infinite or silently zero.

Cell types are carcinoma cells, lymphocytes, macrophages, granulocytes,
plasma cells, fibroblasts and endothelial cells. Each spatial token uses ratio
and average minimum distance at the release's 20/40 neighborhood radii.
TLS is omitted from this morphology-focused primary hypothesis; this is not a
finding that TLS is biologically irrelevant to breast cancer.

### Reference: `brca_shared62_v1`

This uses the exact legacy 62 measurement categories, deterministic transforms
and 5-low/11-high tokenizer, **populated entirely from breast-cancer files**.
It retains TLS and plasma-cell neighborhood measurements. For comparability,
it explicitly retains the legacy convention mapping missing TLS counts to
zero before deriving presence/log-count values. That convention is an inherited
assumption, not a new verification that every absent breast TLS value is a
true biological absence. The primary morphology panel does not rely on it.

The two panels have different widths, low/high routing and adapter parameter
counts. Their comparison is a **panel-and-tokenization comparison**, not a
parameter-matched feature-only ablation. Actual/zero controls *within* each
panel have identical architecture, initialization, parameter count and recipe.

## Preprocessing and model insertion

- Slide-level tissue percentages and cell percentages are divided by 100.
- WTR relative-area fields retain their release fraction scale.
- Cell densities, positive distances and derived region densities use `log1p`.
- Other values remain on their declared scale until standardization.
- Median imputation and mean/population-SD scaling are fitted from the **32
  training slides of each fold only**. Validation/test use frozen statistics.
- All-missing training columns, infinity, invalid negatives, duplicate slide
  barcodes, wrong-cohort inputs and incomplete source joins fail explicitly.
- No metadata, class names, project IDs or labels enter the quantitative vector.

Tokens are embedded in 128 dimensions. Four-head cross-attention updates the
frozen ViLa image-center queries before native center-to-patch attention.
Residual gates start at 0.10. The native visual/text operations and prompts
remain unchanged; only the added conditioner is trainable.

The constructor and runtime smoke check the original native bypass and the
zero-residual equation equivalence. A zero-TME **control is not a bypass**:
it retains the learnable adapter and token identities, zeroing only normalized
slide values before tokenization.

## Controlled comparisons

For **each panel and fold**:

1. `actual`: optimize the conditioner using the slide's quantitative TME.
2. `zero`: same optimizer, seed, checkpoint and capacity, with zeroed values.
3. `tme_only`: a CPU logistic-regression control on the same panel and splits.

Adapters use Adam, learning rate 1e-4, weight decay 1e-5, up to 200 epochs,
validation-error selection and the existing 20-patience/80-minimum-epoch policy.
The TME-only pipeline fits imputation/scaling on train and selects logistic C
from `[0.01, 0.1, 1, 10]` by validation error; ties retain the first/smallest C.
It is a complementary signal baseline, not a capacity-matched neural control.

The full setup is **20 GPU adapter conditions + 10 CPU controls**, excluding
separate pre-campaign smokes. No native baseline retraining is required.
On this allocation the ten CPU controls execute after their matching real-TME
adapters within the same job; thus the campaign uses 22 Slurm jobs including
smokes, not 32. See the launch ledger for both component identities per job.
Report actual versus zero, actual versus native, and actual versus TME-only.
An actual-versus-native gain alone cannot isolate the effect of quantitative
TME from added adaptation. Gains do not establish statistical significance;
use patient-level paired analysis after completion.

## Commands and provenance

Run from the PathoTME project, using the configured PGVL environment for model
execution. Compute nodes must remain offline. `--check-only` and the planner
are dependency-light and can use system Python with PyYAML.

```bash
python scripts/prepare_brca_features.py \
  --opentme-root /path/to/shared/PathoTME-data/OpenTME \
  --panel brca_morph64_v1

python scripts/run_brca_vila.py \
  --experiment-config configs/vila_mil_brca_morph64_v1_16shot.yaml \
  --fold 0 --tme-mode actual --check-only

python scripts/run_brca_tme_only.py \
  --experiment-config configs/vila_mil_brca_morph64_v1_16shot.yaml \
  --fold 0 --check-only

python scripts/plan_brca_vila.py --report /tmp/brca_plan.json
```

Use the `brca_shared62_v1` contract for the reference. Feature materialization
requires `--execute --output ...`; existing outputs are never overwritten.
The private tables and per-file source hashes are under
`/path/to/shared/PathoTME-data/processed/opentme_brca_*.csv` and adjacent
`.metadata.json` files. Source revision is
`9262bc0cd0cd7774d0f7bcbfe6ae5f4665898b25`. Never commit or redistribute them.

The planner **never calls sbatch**. It records exact config/fold/condition,
source hashes, expected runtime identity, output paths and suggested resources.
After explicit launch authorization, run each panel's `--smoke-only` on an
allocated GPU, then submit training using the corresponding `--expected-identity`.
Results belong under `PathoTME-results/vila_mil_tme_guided/brca/16shot/<panel>/`.
Adapter checkpoints, predictions, token-attention columns and metrics carry
panel-specific provenance. Matching completed runs are skipped; partial or
incompatible outputs are preserved and rejected, never reused silently.

## Modified versus preserved

- **Added:** breast-only feature builder, 64-variable panel, shared-reference
  contract, breast tokenizer/adapter, guarded runner, TME-only comparator,
  read-only planner, tests and this documentation.
- **Preserved:** original `features.py`, `guided_vila.py`, NSCLC runner,
  MGPATH code, native PGVL methods, benchmark protocols, prompts, splits,
  checkpoints and all existing results. New BRCA code reuses their interfaces
  without modifying their hashed source, avoiding invalidating queued runs.
