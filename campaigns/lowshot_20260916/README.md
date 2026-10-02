# PathoTME missing 4/8-shot coverage

User-authorized priority: finish four-cohort, seven-model PLIP/CLIP-RN50 coverage
at both 4 and 8 shots. This campaign fills 96 missing settings with 480 fold
allocations and 48 bundled GPU smokes. Each fold trains matched Native and
zero/actual/shuffled frozen-native adapters (1,920 new condition/fold results).
The 80 completed ViLa/MGPATH NSCLC/BRCA folds are verified and excluded.

Splits reuse the completed all-shot LR study exactly, including patient-disjoint
train/validation phases and unchanged outer tests. Existing low-shot memberships
and donor maps must match. NSCLC/CRC/BLCA use core62; historical BRCA stays morph64.
Native prompts, recipes, epochs, losses, selection rules and architecture equations
are unchanged. CRC/BLCA retain their explicit task-specific class and panel bindings.
The isolated runtime inherits original training loops and restores scoped validators.

Resources use recorded 16-shot timings with a fixed-overhead floor, native-reuse
adjustment, measured-RSS margins and HiVE hierarchy-cache bounds. Flexible walltime
ranges preserve a larger allowance; successful completion is not guaranteed.
GPU smoke dependencies gate every fold. CPU checks do not constitute GPU results.

Pending jobs in the separate 32/64-shot campaign are held by a scheduling amendment;
running jobs and all 76 TOP holds are preserved. No original campaign source is edited.
Existing LR fits are reused as controls; this GPU campaign does not automatically
compute probability-fusion results or new attribution exports. No full-shot runs.

Private outputs: shared PathoTME-results/tcga_4_8shot_allmodels_20260916_v1/.
