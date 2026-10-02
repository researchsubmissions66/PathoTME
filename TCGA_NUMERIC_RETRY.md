# TCGA numeric-config retry, 10 September 2026

The first launch exported typed configurations as JSON and passed those files
through the existing YAML training loader. YAML 1.1 resolves JSON's `1e-05`
and `9e-06` as strings. Fifteen baseline-training jobs and both RN50 MGPATH
smokes failed before their first optimizer step; ten dependent folds stayed
blocked. The fifteen completed BRCA folds remain separate and unchanged.

`pathotme/yaml_runtime_config.py` writes native YAML with SafeDumper and checks
that the full tree reloads exactly through the training configuration loader.
It emits numeric exponents such as `1.0e-05`. This fixes export at its source
without changing PGVL's shared runtime or the frozen original launch files.

`scripts/retry_tcga_numeric_configs.py --prepare` derives new YAML from the
original typed snapshots. Only each output directory changes; all numerical
values, seeds, splits, feature stores, prompts, recipes and analysis conditions
are preserved. Original failed output is retained. No selected failed/blocked
fold has a saved checkpoint or completed metrics. The new source-bound plans
live under shared `PathoTME-results/tcga_numeric_retry_20260910_v1/`, with
`native/` (NSCLC ViLa/RN50 and MGPATH/PLIP) and `cross/` (NSCLC ViLa/PLIP and
both RN50 MGPATH cohorts) groups. Existing runners consume the new YAML.

Run `scripts/check_tcga_numeric_retry.py` for the single focused regression:
all 25 saved configs through `train.load_yaml_config`, exact value/type
comparison against the intended snapshot, reproduction of the old bad types,
and real method optimizer construction/one update on a tiny parameter probe
for each of five pairings. Native conditions also pass preflight. This checks
config/optimizer integration; actual models and assets remain GPU-smoke gated.

The exact retry is **25 fold jobs plus five smoke jobs**, with the unchanged
one-A100/eight-CPU/48-GiB resource recipe. Each fold depends on its successful
smoke. The submit step checks those exact Slurm requests, cancels only the ten
unstarted jobs blocked on failed smokes, submits replacements, and immediately
records each returned ID. The three previously passed smoke pairings are
rechecked against the corrected runtime configurations. This is continuation
of the same exploratory study; it does not add conditions or restart completed
folds. Preserve the parent ledgers and use the new old-to-new mappings for
paper attribution. Do not rerun submission when its ledger exists.
