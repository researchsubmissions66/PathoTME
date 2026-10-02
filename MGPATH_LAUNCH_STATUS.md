# MGPATH PathoTME launch status — 2026-09-06

- Implementation and contract: `TME_GUIDED_MGPATH.md` and
  `configs/mgpath_tme_guided_nsclc_16shot.yaml`.
- All six read-only plans passed baseline identity, exact split, TME coverage,
  native prompt and checkpoint hash checks. No native training is repeated.
- GPU smoke job: **21846044**, one A100, 6 CPUs, 24G, 45 minutes;
  submission ledger: `mgpath_smoke_launch_20260906.json`.
- The smoke is separate from training. Check its `smoke_report.json` under
  `mgpath_tme_guided/nsclc/16shot_v1/smoke/actual/fold0/` in PathoTME-results.
- User explicitly approved the six-job campaign on 2026-09-06. A fresh dry-run
  passed for all six conditions, and the smoke's current identity matched its
  launch identity. All six jobs were then **submitted** with
  `afterok:21846044`, one A100/6 CPUs/24G/45 minutes each (maximum 4.5 GPU-hours).
  Ledger: `mgpath_guided_launch_20260906.json`; do not submit duplicates.

  | Fold | Actual TME | Zero TME |
  | --- | --- | --- |
  | 0 | 21846218 | 21846219 |
  | 2 | 21846220 | 21846221 |
  | 3 | 21846222 | 21846223 |
- A successful smoke must precede training; retain the successful-smoke
  dependency when submitting. Do not bypass a failed or stale smoke.
- Targeted CPU tests passed: **6 passed** in 420.28 seconds. They cover native
  bypass equivalence, quantitative/zero-TME behavior, adapter-only gradients,
  checkpoint round trips, and train-only standardization. The sole warning was
  an unwritable pytest cache; no test failed.
- Smoke was **RUNNING** on `gpua018` during submission checks.
  No smoke report had been observed yet; this is not a smoke-pass claim.

Training is queued but cannot start before successful smoke completion. If the
smoke fails, investigate it; do not simply remove the dependency. Source or
input changes invalidate planned identities and require a new smoke/plan.
