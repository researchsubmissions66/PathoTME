# PathoTME 32/64-shot follow-up

Authorized scope: seven models, PLIP and CLIP-RN50, five folds. NSCLC and BRCA
use 32 and 64 shots; BLCA uses 32. CRC cannot support these balanced, disjoint
train/validation sizes. Full-shot is explicitly excluded by the user.

350 fold allocations fit fresh matched native models and zero, actual-TME and
shuffled-TME frozen-native adapters. 42 bundled GPU smoke allocations gate the
corresponding groups, exercising both shot values where applicable.

Splits exactly reuse the completed all-shot LR analysis memberships, nesting
the original 16-shot train/validation sets and preserving every outer test fold.
Donors use the original label-blind, phase-local patient-excluding bijection.
BRCA remains morph64; NSCLC and BLCA retain their existing core62 panels.
All prompts, losses, epochs, early stopping, model equations and preprocessing
are inherited unchanged. Every fold fits its own native baseline. The fresh
one-bag smoke fixture is only a structural check and never a study baseline.

The isolated runtime scopes exact configuration validators and restores them
on success or failure. It inherits original native train_one_fold and the
original per-method adapter arm_run functions. Existing source files are not
edited. Previous numerical invariance and DyKo directional checks are retained;
the margin-resolution gate is not lowered.

Resource requests use saved matching 16-shot accounting, shot scaling and
explicit native-reuse correction where needed. Flexible TimeMin/TimeLimit
requests retain safety margins but cannot guarantee completion. GPU smoke
success remains necessary; CPU constructors and dataset checks are not GPU
execution or research results. Preserve all outcomes, including null/negative.

Private campaign records are under shared PathoTME-results
`tcga_32_64shot_20260915_v1/`. Raw features are not copied into the repository.
Existing TOP holds and all older running/queued source-bound campaigns remain
untouched. This campaign adds no neural feature-subset/group ablation and no
new probability-fusion CPU controller; high-shot LR controls already exist.
