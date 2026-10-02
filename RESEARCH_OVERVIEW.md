# PathoTME

PathoTME is a research project on tumor-microenvironment-conditioned learning
from whole-slide pathology images. It studies how quantitative OpenTME
phenotypes can act as biological supervision, semantic knowledge, and
interpretable conditioning signals for MIL and pathology vision-language
models.

## 🧭 Method names

| Name | Definition |
|---|---|
| **Native** | The matched underlying model without TME. |
| **PathoTME-LR** | L2 logistic regression using only the matched numeric TME panel. |
| **PathoTME-Fusion50** | An equal average of Native and PathoTME-LR class probabilities. |
| **PathoTME-FusionVal** | Native and PathoTME-LR probability fusion with the TME weight selected on validation patients. |
| **PathoTME-Adapter** | A learned conditioner using the slide’s own TME measurements with a frozen native checkpoint. |
| **PathoTME-Adapter-Zero** | The adapter ablation with standardized numeric TME inputs set to zero. |
| **PathoTME-Adapter-Shuffled** | The adapter ablation with a different patient’s TME row from the same data split. |

PathoTME is the study umbrella; LR and probability fusion retain their roles as
standard baseline algorithms. Report the underlying architecture and encoder
separately, for example **PathoTME-FusionVal · MGPATH · PLIP**. PathoTME-LR
shares one fit per matching cohort/panel/fold and has no visual encoder.

Fusion uses `p = (1 - alpha) * p_native + alpha * p_tme_lr`. Fusion50 fixes
`alpha = 0.5`; FusionVal selects from `0, 0.25, 0.5, 0.75, 1` on validation
patient AUROC, then NLL, then smaller alpha. Both the Native-only and LR-only
endpoints remain eligible. PathoTME-Adapter and PathoTME-LR use TME at inference;
FusionVal needs TME whenever the selected weight is nonzero.

In compact tables, **PathoTME-Zero** and **PathoTME-Shuffled** may abbreviate
the two adapter ablations when the caption explicitly identifies them.
Stored condition keys, CLI options and historical run identifiers retain
their original spelling. The [canonical name mapping](docs/data/variant-names.json)
connects these identifiers to presentation labels.

## Central question

> Can quantitative tumor phenotypes serve as intermediate biological
> supervision for learning whole-slide image representations?

The main hypothesis is that OpenTME should not be treated merely as another
feature vector concatenated with a slide embedding. Its measurements can play
several distinct functional roles:

- privileged biological supervision during training;
- phenotype tokens that interact with WSI patch tokens;
- task-specific visual and textual prototype construction;
- phenotype-guided semantic retrieval;
- biologically directed patch selection;
- coarse-to-fine concept organization; and
- evidence used by an agentic pathology system.

## Prioritized directions

1. **Privileged TME supervision** — train a WSI model to predict biologically
   meaningful phenotypes alongside the clinical target, while requiring only
   the WSI at inference.
2. **TME phenotype tokens** — compress thousands of measurements into a small
   set of interpretable tokens and connect them to patch tokens through
   cross-attention.
3. **TME-conditioned prototypes** — ground Libra-MIL/ViLa-MIL-style prototypes
   in immune, stromal, tumor, TLS, and spatial-interaction phenotypes.
4. **Phenotype-guided retrieval** — use OpenTME to retrieve slide-specific
   biological descriptions in a MUSE-style semantic pathway.
5. **TME-grounded hierarchy** — organize coarse and fine concepts using ideas
   from HiVE-MIL, MAPLE, and MGPATH.
6. **TME-guided token selection** — retain WSI regions associated with the
   slide phenotype using a FOCUS-style visual compressor.
7. **TME-conditioned prompts** — use biologically grounded prompts with SLIP,
   SLDPC, or PathPT as a simpler baseline.

The complete motivation, method map, equations, examples, and initial
experimental roadmap are in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## Project status

### PathoTME-LR and probability-fusion baselines

The [matched baseline suite](campaigns/baselines_20260914/README.md) adds
PathoTME-LR (TME-only L2 logistic regression), PathoTME-Fusion50 (fixed
50/50 native–TME probability fusion), and PathoTME-FusionVal
(validation-selected probability fusion). It covers the seven PathoTME
architectures with PLIP and CLIP-RN50 across NSCLC, BRCA, CRC and BLCA,
using the corresponding 16-shot patient folds and feature panel. Preprocessing
is fitted on training slides; regularization and fusion weights use validation
patients only. Identical panel/fold inputs share one TME fit. Computation runs
on CPU and reuses saved native checkpoints. Reports include patient AUROC,
NLL and Brier alongside slide-level metrics. Historical BRCA comparisons
retain morph64; this does not relabel them as common62 experiments.

### Standalone TME teacher follow-up

The [standalone-teacher experiment](STANDALONE_TME_TEACHER.md) isolates whether
a quantitative-only subtype classifier can improve a WSI-only CONCH–ViLa
student. It reuses ten completed classification-only/visual-distillation control
folds and adds ten real/shuffled standalone-teacher student folds. Source-only
validation diagnostics measure teacher complementarity; exported students
require no TME input. Existing runners and results remain unchanged. The new
launcher defaults to dry-run; implementation does not submit jobs or establish
a performance gain. Both TCGA and CPTAC outcomes were previously inspected, so
this is an explicitly exploratory follow-up.

### Controlled CONCH–ViLa and WSI-only transfer

The [controlled study](CONTROLLED_VILA.md) adds real/zero/shuffled adapters,
validation-selected prediction fusion, training-only TME auxiliary supervision,
and real/shuffled/visual-teacher distillation. Matched continuation controls
separate TME effects from extra training. WSI-only student exports require no
TME data at inference. A source-frozen CPTAC BRCA evaluation is prepared on 98
verified primary-tumor patients (88 IDC, 10 ILC), with audited dual-scale CONCH
features and explicit historical feature-provenance limitations. The launcher
dry-runs first, uses single-fold jobs, and records every condition and job ID.
The [authorized campaign](CONTROLLED_LAUNCH_STATUS.md) has submitted all 51
jobs, with training gated on GPU smoke and CPTAC inference following source
training in the same allocations. Submission is not a performance result.

### TCGA-BRCA extension

The [BRCA-specific ViLa-MIL setup](TME_GUIDED_BRCA.md) adds a 64-variable
carcinoma/stromal morphology panel and a separate breast-populated shared
62-variable reference. Both cover all 960 IDC/ILC benchmark slides. Authored
16-shot contracts use all five existing folds, matched real/zero-TME adapters
and CPU TME-only controls. Native BRCA prompts, checkpoints and splits remain
unchanged. The read-only planner binds each condition to its config and input
hashes. The user-authorized [BRCA campaign](BRCA_LAUNCH_STATUS.md) is now queued:
two panel smokes plus 20 dependent fold jobs. The ten TME-only controls share
their matching actual-TME allocations because no CPU allocation is available.

### Existing NSCLC extensions

The pinned OpenTME quantitative release is available locally; see
[DATA.md](DATA.md) for its access conditions, storage location, revision, and
validation status. Two explicitly non-upstream ViLa-MIL extensions are set up:

- a leakage-safe late-fusion signal canary; and
- a TME-guided prototype experiment in which 5 low-scale and 11 high-scale
  quantitative phenotype tokens condition ViLa's learned image-center queries
  through parameter-efficient cross-attention.

The native ViLa-MIL prompts, dual-scale feature bags, and checkpoint remain
unchanged. The guided experiment freezes that checkpoint and trains only the
PathoTME adapter. See [TME_GUIDED_VILA.md](TME_GUIDED_VILA.md) for the precise
architecture, negative control, provenance boundary, and launch contract. No
performance improvement is claimed until the scheduled experiments complete.

The next extension is [TME-guided MGPATH](TME_GUIDED_MGPATH.md): quantitative
tokens condition its 64 image centers while freezing the native PLIP model,
graph and text/transport paths. Matched actual/zero controls use existing
16-shot NSCLC checkpoints on exact-coverage folds 0, 2, and 3. The runner has
hash-bound launch provenance, a separate GPU smoke, and adapter-only outputs.
This setup does not yet establish a performance improvement for MGPATH.
