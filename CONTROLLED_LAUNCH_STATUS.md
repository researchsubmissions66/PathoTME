# Controlled CONCH–ViLa launch: 2026-09-07 evening

All **51 jobs** were submitted and verified in Slurm. The durable mapping of
every job to its exact configuration, source identity, fold, commands and
dependencies is [controlled_launch_20260908.json](controlled_launch_20260908.json).

- Smoke suite: **21871042**, pending Priority at the submission check.
- Fifty single-fold jobs: all pending Dependency at that check.
- Ten TCGA conditions × five BRCA 16-shot folds; six conditions deploy WSI-only.
- Thirty-five CPTAC evaluations (six students plus the original native baseline,
  each across five source folds) run within the student allocations afterward.
- Each allocation requests one A100, six CPUs and 24 GiB host RAM. Five fusion
  jobs request 30 minutes; 15 adapters request 45 minutes; 30 student jobs and
  the smoke suite request 60 minutes. These are limits, not runtime predictions.
- All smoke and teacher dependencies were checked against the live queue.
  A failed smoke cannot release training; a failed teacher cannot release its
  corresponding distillation condition.
- Existing source checkpoints, completed TME-only results and native test
  predictions are reused. No original PGVL jobs were replaced or canceled.
  All 77 TOP user holds remain unchanged.

Validation passed: six dependency-light contract/launch/external tests, five
new model/control tests, four existing breast-model tests, and one synthetic
end-to-end test executing all ten conditions. Syntax/lint checks passed.
**The real GPU smoke and performance results were still pending at submission.**

CPTAC labels and tumor-specimen annotations were downloaded from NCI PDC and
joined exactly to TCIA. The external cohort has **98 primary-tumor patients:
88 IDC and 10 ILC**. All 196 HDF5 headers passed feature-identity and geometry
checks. See [CONTROLLED_VILA.md](CONTROLLED_VILA.md) for exclusions, historical
extraction-checkpoint attestation limits and the source-only selection policy.

Results are stored under
`/path/to/shared/PathoTME-results/vila_conch_controlled/brca/16shot/brca_morph64_v1/v1/`.
Do not edit the queued source-hashed runtime, contracts or target cohort, and
do not submit duplicate jobs. Recheck the ledger and Slurm for live status.
