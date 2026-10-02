# Controlled CONCH–ViLa study and WSI-only deployment

## Scope

TCGA-BRCA IDC/ILC, 16 labeled training slides and 16 validation slides per
class, all five existing patient-disjoint folds, seed 1 + fold. The primary
panel is the breast-specific `brca_morph64_v1`, not the lung panel. The source
is PGVL's **paired-feature CONCH ViLa extension**, not upstream RN50 ViLa.
The existing prompt bank, native visual equations and source splits are retained.

## Conditions

| Condition | What is trained | TME required at inference? |
| --- | --- | --- |
| Existing CONCH ViLa | Reused source-selected checkpoint | No |
| Existing TME-only logistic regression | Reused source-selected classifier | Yes |
| `actual`, `zero`, `shuffled` | Matched adapters over frozen CONCH ViLa | Yes / zero-vector control / shuffled measurements |
| `fusion` | No classifier retraining; validation-selected probability blend | Yes |
| `student_ce` | Matched classification-only continuation of ViLa | No |
| `aux_real`, `aux_shuffled` | ViLa + training-only TME regression head | No |
| `kd_visual`, `kd_real`, `kd_shuffled` | ViLa distilled from the corresponding frozen teacher | No |

Fifty new fold conditions plus one ten-branch GPU smoke suite. The two existing
baselines are not retrained. Every student has the same auxiliary module and
initialization; it is active only for auxiliary-supervision variants. Thus the
capacity scaffold matches, but unused heads do not receive gradients.

## Auxiliary supervision

The auxiliary head consumes the concatenation of ViLa's native low/high
attention-pooled visual representations. A temporary hook observes the native
LayerNorm outputs; pooling uses the same native attention layers. No text or
visual forward equations are replaced. Regression gradients reach the visual
prototype, attention and pooling parameters shared with classification.

- Loss: subtype cross-entropy + 0.1 × observed-value-only Huber regression.
- Targets: quantitative breast measurements after the panel's fixed percentage
  and log transforms, then training-fold median imputation and z-scoring.
- Missing values are masked out of regression; all-missing slides contribute
  zero auxiliary loss. No validation/test TME targets enter the student loss.
- Head: `2 × CONCH width → 128 → GELU → 64`; no head dropout.
- Export: native ViLa state only. The regression head and TME standardizer
  are absent from `foldN_inference.pt`.

All students continue the same existing source-selected checkpoint for up to
200 epochs with Adam, learning rate 1e-4, weight decay 1e-5, validation-error
selection, patience 20 and minimum epoch 80. `student_ce` controls for this
additional training; comparison solely to the original checkpoint is inadequate.

## Distillation

Teachers are frozen and evaluated once on the exact few-shot **training**
slides. Students optimize subtype cross-entropy plus KL(teacher || student),
temperature 2, temperature-squared scaling and weight 1. No additional patients
or teacher predictions from validation/test enter training. `kd_visual` controls
for distillation itself. `kd_real` and `kd_shuffled` depend on successful matching
adapter folds. Smoke uses untrained teacher fixtures for those two branches;
the full runs require identity- and artifact-valid completed teachers.

## Attribution controls

- Shuffling is a deterministic, label-blind, split-local **slide-row bijection**
  with donors required to belong to another patient. Every quantitative row is
  used exactly once per split. It is not whole-patient block shuffling when test
  patients have multiple slides. Donor maps are serialized in run provenance.
- Zero inputs match adapter parameter count, but not functional effective
  capacity; the shuffled control is therefore also necessary.
- Fusion uses `(1-alpha) * visual + alpha * TME`, alpha 0–1 in increments of
  0.1, selected on source validation error with ties favoring the visual model.
  Saved source test predictions are reused. A fixed 0.5 blend is secondary.
- Auxiliary supervision is privileged information, not TME-token conditioning.
  Existing measured-TME results do not establish WSI-only performance.

## CPTAC external transfer

The downloaded PDC study `PDC000120` (catalog version 2) supplies histology and
biospecimen types. Exact TCIA patient/sample matching and both CONCH feature
scales yield **98 primary-tumor slides / 98 patients: 88 IDC, 10 ILC**. Mixed,
other, unreported, conflicting morphology, normal and unidentified specimens
are excluded with individual audit reasons. This is the explicitly eligible
external subset, not all CPTAC breast patients. Raw responses, queries, hashes,
clinical labels and the frozen manifest are under
`${PATHOTME_DATA_ROOT}/external/cptac_brca/v1/`.

All 196 HDF5 headers verify CONCH-v1, width 512, low 5x/512px and high
10x/256px. Historical extraction checkpoint hashes were **not recorded**;
runtime checkpoint identity does not attest the old extraction. This limitation
is serialized rather than replaced with a fabricated provenance claim.

Thirty WSI-only student evaluations plus five original native-baseline
evaluations follow the source folds in their existing GPU allocations. No extra
GPUs are reserved for external inference. All source models are evaluated;
CPTAC does not select a checkpoint, blend, threshold, seed or condition.

`scripts/eval_wsi_only.py` has no TME loading path and never calls an optimizer.
Target labels are used for scoring, not passed to model forward (a dummy label
serves the native loss-bearing API). Frozen target hashes and exact source
identities bind each evaluation. Report patient-level metrics and paired
intervals; only ten ILC patients make uncertainty particularly important.
Five source checkpoints evaluated on the same CPTAC patients are not five
independent external cohorts.

## Commands and provenance

```bash
python scripts/launch_controlled_vila.py --with-external --report /tmp/controlled_dry.json
# Only with explicit launch authorization:
python scripts/launch_controlled_vila.py --with-external --submit --report controlled_launch.json
```

Each fold saves resolved configuration, source hashes, donor maps, selected
validation predictions, test predictions, metrics and checkpoint hashes.
The launch ledger records exact commands, dependencies and job IDs. Partial
outputs are preserved and require explicit recovery; valid completed folds
are skipped. Native PGVL source and previous PathoTME runtime files are not
changed, preserving their historical identities. These are experimental
extensions, not established performance improvements.

After completion, `scripts/report_controlled_vila.py --output <fresh-directory>`
reports complete OOF comparisons and stratified paired patient-bootstrap
intervals. External results include each source fold and a predeclared
equal-weight probability ensemble across all five source models; the same 98
patients are not counted five times. These are exploratory comparisons without
multiplicity correction, and intervals are conditional on the fitted models.

## Data sources

- [NCI PDC public API](https://pdc.cancer.gov/pdc/publicapi-documentation)
- [CPTAC-BRCA TCIA collection](https://www.cancerimagingarchive.net/collection/cptac-brca/)
- [Privileged-information distillation](https://arxiv.org/abs/1511.03643)
