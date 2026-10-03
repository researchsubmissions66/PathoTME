# Refreshed HiVE-MIL CRC and BLCA hierarchies

This v2 bank replaces the generic clause-based HiVE descriptions for new runs. Each class has four explicit morphological parent concepts followed by three related children per parent, in native HiVE order. The banks contain 32 unique descriptions per cohort and have no shared class-prefixed template. Descriptions express possible H&E appearances rather than mandatory diagnostic criteria.

- [CRC bank](crc/hierarchy.json): adenocarcinoma NOS versus mucinous adenocarcinoma.
- [BLCA bank](blca/hierarchy.json): non-papillary versus papillary urothelial carcinoma.
- [Authored source](source.py): structured parent/child descriptions, preserving the approved draft NOS branch.
- [Provenance and hashes](MANIFEST.json): assistant-authored; no external generation API or verified model revision is claimed.

The upstream NSCLC and BRCA HiVE banks remain the corresponding cohort references. Historical v1 banks remain under `../crc_blca_gpt6_astra_v1/` for reproducing earlier results. New banks require freshly trained native checkpoints; old native checkpoints cannot be combined with these banks as a matched comparison.

The refreshed campaign uses the established few-shot grid: NSCLC/BRCA at 4, 8, 16, 32, and 64 shots; CRC at 4, 8, and 16; BLCA at 4, 8, 16, and 32. Both PLIP and CLIP-RN50 use five folds and native, zero, actual, and shuffled conditions. Full-shot remains cancelled. GPU smoke gates check text-embedding uniqueness, adapter gradients, and invariants before training; submission alone is not a successful smoke or performance result.
