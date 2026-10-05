# 16-shot PathoTME-LR attribution figures

These figures show the **ten largest biological-group scores** and the **ten
largest individual-feature scores** for each cancer. All use the TME-only
PathoTME-LR variant. They do not depict the learned image-aware PathoTME
adapter. The completed fusion scores are scaled LR scores, so separate fusion
plots would repeat the same feature effects at different fixed TME weights.

| Cancer | Preview | Editable vector | Print vector |
|---|---|---|---|
| NSCLC | [PNG](nsclc_lr_16shot_attribution.png) | [SVG](nsclc_lr_16shot_attribution.svg) | [PDF](nsclc_lr_16shot_attribution.pdf) |
| BRCA | [PNG](brca_lr_16shot_attribution.png) | [SVG](brca_lr_16shot_attribution.svg) | [PDF](brca_lr_16shot_attribution.pdf) |
| CRC | [PNG](crc_lr_16shot_attribution.png) | [SVG](crc_lr_16shot_attribution.svg) | [PDF](crc_lr_16shot_attribution.pdf) |
| BLCA | [PNG](blca_lr_16shot_attribution.png) | [SVG](blca_lr_16shot_attribution.svg) | [PDF](blca_lr_16shot_attribution.pdf) |

Bars show the five-fold average of mean absolute patient-level changes in
predicted class probability after one TME measurement—or all measurements in a
group—is replaced with its training-fold reference. White circles show the
five individual fold means. The horizontal scales differ between the group
and individual-measurement panels. Group scores are joint interventions, not
sums of individual scores. These are model-sensitivity values in probability
percentage points, not accuracy changes, causal effects, or confidence
intervals.

Display labels are shortened for legibility. The exact feature identifiers,
five-fold scores, and fold ranks are in the per-cancer [top-10 files](../README.md)
and the full [fold table](../fold_feature_scores.csv). The charts come only
from those published aggregate CSVs; no patient-level data is used. The PNGs
are rendered at 300 dpi, while the SVG and PDF files preserve vector text and
graphics.

To regenerate with Python and Matplotlib, run from the repository root:

```bash
python docs/data/attribution-16shot-20261004/figures/make_figures.py
```
