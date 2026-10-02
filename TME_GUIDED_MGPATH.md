# TME-guided MGPATH: NSCLC 16-shot diagnostic

This is a **PathoTME extension of PGVL-Gym's partial PLIP-only MGPATH
implementation**, not an upstream reproduction or the paper's PLIP-G condition.
The registered base recipe is pinned to upstream commit
`ed703fd6e1ae200b4821d611ec895dd75ecf8219` and uses cached 5x/10x 224px PLIP
preprojection bags, its frozen native projection, four prompt views, and 64
learned image centers. No PGVL method, prompt bank, protocol or baseline result
is modified by this experiment.

## Intervention

Reuse the existing 62 quantitative features, percentage/log1p preprocessing,
training-fold-only median imputation and z-standardization. Five tissue/TLS
tokens condition low-scale centers; eleven cell/spatial tokens condition
high-scale centers. These are slide-level quantitative measurements, not
patch-local labels or text embeddings of feature names.

For each scale, the 64 original centers cross-attend to the TME tokens in a
128-wide, four-head adapter space. Add the projected output through a learned
gate initialized to 0.1. Use the conditioned centers in **both** raw/graph
attention queries and the center residual. Retain the graph mixture (0.2),
normalization, original four-row text bank, and epsilon-0.1/100-iteration
Sinkhorn scoring. Frozen text features are cached once in eval mode, an exact
computation reuse rather than a changed text representation.

All base parameters stay frozen and in evaluation mode. Only the conditioner
is trained. `zero` retains exactly the same architecture and trainable
parameter count but zeros standardized values; `actual` receives the correct
slide's measurements. TME is required at inference; this is not privileged
training-only supervision.

## Comparison and limits

Use original 16-shot train/validation splits and exact-coverage folds 0, 2, 3.
Each fold has 32 training and 32 validation slides. Folds 1/4 are not silently
reduced for missing TME. These previously inspected holdouts are diagnostic,
not final untouched five-fold evidence.

Both adapters use Adam, LR 1e-4, weight decay 1e-5, up to 200 epochs, and
validation macro-F1 selection (ties replace the best). Patience is 100 with
minimum stopping epoch 160. The adapter LR follows our ViLa extension and is
an explicit departure from native MGPATH's 9e-6. It is fixed before inspecting
results and identical between actual and zero.

The native 16-shot checkpoints/predictions already exist and are reused.
The runner validates their configuration ownership and exact holdout IDs.
TME-only can be compared from the previous canary after verifying identical
splits; **MGPATH-specific late fusion is not implemented/launched here** and
must not be substituted with the existing ViLa fusion results.

## Validation and launching

From PGVL-Gym, with PathoTME's path available:

```bash
python /path/to/PathoTME/scripts/run_mgpath_guided.py \
  --experiment-config /path/to/PathoTME/configs/mgpath_tme_guided_nsclc_16shot.yaml \
  --fold 0 --tme-mode actual --check-only
python /path/to/PathoTME/scripts/launch_mgpath_guided.py --smoke-only
```

`--check-only` does not import Torch or open feature tensors, but verifies all
feature paths, patient separation, coverage and hashes (including the native
checkpoint). A GPU smoke verifies bypass/native logit equivalence and finite
adapter-only gradients on a training bag; it writes under `smoke/`, never a
completed training fold. Unit tests use tiny real MGPATH equations without
private weights or data.

The launcher is dry-run by default. `--submit --report NEW_LEDGER.json` submits
only with explicit authorization. Each condition/fold is one GPU job, 6 CPUs,
24G RAM, 45 minutes by default. A full campaign contains six jobs. Optional
`--dependency SMOKE_JOB_ID` gates them on successful smoke completion.

Launch ledgers bind job IDs to the complete contract and input/source hashes.
Jobs refuse changed launch identities. Results contain resolved configuration,
baseline hashes, source hashes, adapter-only checkpoints (including fitted
normalization), training history, matched native/guided slide and patient
metrics, and predictions with TME-group attention. Matching completed folds
are skipped; mismatched or partial directories are preserved and rejected.
No epoch-level resume is claimed.

Outputs: `/path/to/shared/PathoTME-results/mgpath_tme_guided/nsclc/16shot_v1/`.
