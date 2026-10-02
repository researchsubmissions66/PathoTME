# DyKo BRCA morphology bank

64 separate, assistant-authored H&E descriptions; eight descriptive groups of eight.
The vocabulary covers breast architecture and cytology plus shared tissue context.
Every concept is available to every visual prototype regardless of the slide label.
Descriptions name observable patterns without declaring them subtype-exclusive.
No biomarkers, clinical outcomes or case data enter this bank.

The sources and selection rationale are in `knowledge_bank.json`. Existing PGVL
breast banks informed the vocabulary but were not copied as class-specific rules;
for example, molecular E-cadherin expression is excluded from an H&E concept bank.
Architecture is informed by the [RCPath breast dataset](https://www.rcpath.org/asset/D255F34C-176A-490D-9B5A7D58AC85F3A6/)
and grading axes by [NCI SEER](https://training.seer.cancer.gov/breast/evaluation/grade.html).
Independent pathology review has not been performed.

`class_descriptions.csv` separately preserves the two existing high-resolution
FOCUS IDC/ILC strings exactly, reformatted into DyKo's two-row CSV. It is not
upstream DyKo text. The ordered knowledge concepts are encoded once by the pinned
TITAN tower into a 64x768 tensor. A learned concept bridge maps that shared frozen
asset into each paired encoder space. Both native and TME-guided BRCA models use
this same bank. No interpolation or duplicated padding inflates it to 1000 rows.

This is an explicitly different BRCA knowledge condition from NSCLC's released
1000-concept bank. Compare TME effects within each cohort/encoder; this does not
establish bank-size equivalence or an upstream DyKo BRCA reproduction.
