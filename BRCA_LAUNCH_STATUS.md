# BRCA launch — 2026-09-07

The user authorized launch after implementation. Fresh preflight passed and
all **22 Slurm jobs** were submitted successfully under `shared-delta-gpu` on
`gpuA100x4`: one A100, six CPUs, 24G host RAM and 45 minutes per job.

## Exact job mapping

| Panel | Smoke | Fold | Actual TME + CPU TME-only | Zero TME |
| --- | --- | --- | --- | --- |
| BRCA morphology 64 | 21853416 | 0 | 21853417 | 21853418 |
| BRCA morphology 64 | 21853416 | 1 | 21853419 | 21853420 |
| BRCA morphology 64 | 21853416 | 2 | 21853421 | 21853422 |
| BRCA morphology 64 | 21853416 | 3 | 21853423 | 21853424 |
| BRCA morphology 64 | 21853416 | 4 | 21853425 | 21853426 |
| Breast shared 62 | 21853427 | 0 | 21853428 | 21853429 |
| Breast shared 62 | 21853427 | 1 | 21853430 | 21853431 |
| Breast shared 62 | 21853427 | 2 | 21853432 | 21853433 |
| Breast shared 62 | 21853427 | 3 | 21853434 | 21853435 |
| Breast shared 62 | 21853427 | 4 | 21853436 | 21853437 |

Each of the 20 fold jobs has `afterok` on its own panel's smoke. A failed smoke
must be investigated; do not remove dependencies to bypass it. Submission is
not evidence of smoke success, convergence, completion or improved performance.

## CPU-only comparator scheduling

Slurm test-only submissions rejected both the GPU account on the CPU partition
and the `noalloc` association. The account list exposes no CPU allocation.
Therefore each actual-TME job runs its matching CPU-only logistic-regression
control **after** successful adapter training, within the same allocation.
There are no ten additional GPU requests for these small controls.

The two components have separate fingerprints and output directories but share
one Slurm job ID. If an actual-TME job fails, its subsequent TME-only condition
may not have run. Inspect each component's `metrics.json`; do not count all
components complete just because one produced output. Job-level cost is shared
and must not be counted twice. Recovery must preserve completed components.

## Provenance and recovery

- `brca_launch_20260907_dry.json`: fresh dry-run commands and full input contracts.
- `brca_launch_20260907.json`: exact submitted argv, job IDs, dependencies and
  component-level config/fold/panel/input hashes.
- `scripts/launch_brca_vila.py`: dry-run by default; duplicate-job guard,
  launch lock, partial-ledger persistence, matching-completion checks.
- Launcher tests: **3 passed**; the preceding feature/adapter suite: **26 passed**.
- Results: `PathoTME-results/vila_mil_tme_guided/brca/16shot/<panel>/`.
- Logs: `PathoTME-results/logs/ptme-brca-<panel>-<condition>-f<fold>-<job>.out`.

No existing native baseline or NSCLC/MGPATH result was rerun, replaced or
modified. Keep source-hashed runtime files stable while these jobs are queued.
TOP holds and other campaigns were not changed.
