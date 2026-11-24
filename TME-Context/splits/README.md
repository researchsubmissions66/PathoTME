# BRCA split protocol

`brca/16shot/fold0` through `fold4` contain the real patient-disjoint
TCGA-BRCA IDC/ILC 16-shot train, validation, and test assignments. Each CSV
has `slide_id`, `case_id`, `label`, and `feature_file`. The `feature_file` value
is an HDF5 basename; supply its directory with `--feature-root`.

These files contain split metadata only. They contain no feature vectors,
quantitative TME measurements, model weights, predictions, or scores.
