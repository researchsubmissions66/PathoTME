# 16-shot TME feature attribution figures

Each compact bar chart shows the **top 10 biological feature groups** and
**top 10 individual TME measurements** for one cancer and variant. Bars show
the five-fold mean absolute patient-level probability change, in percentage
points. The right column gives that mean ± the **sample SD across the five fold
means**. Group and individual panels use separate horizontal scales. Exact
fold values and canonical feature names are in the [per-cancer top-10 CSVs](../README.md).

| Cancer | TME-only PathoTME-LR | PathoTME-Fusion50 |
|---|---|---|
| NSCLC | [PNG](nsclc_lr_16shot_attribution.png) · [SVG](nsclc_lr_16shot_attribution.svg) · [PDF](nsclc_lr_16shot_attribution.pdf) | [PNG](fusion50/nsclc_fusion50_16shot_attribution.png) · [SVG](fusion50/nsclc_fusion50_16shot_attribution.svg) · [PDF](fusion50/nsclc_fusion50_16shot_attribution.pdf) |
| BRCA | [PNG](brca_lr_16shot_attribution.png) · [SVG](brca_lr_16shot_attribution.svg) · [PDF](brca_lr_16shot_attribution.pdf) | [PNG](fusion50/brca_fusion50_16shot_attribution.png) · [SVG](fusion50/brca_fusion50_16shot_attribution.svg) · [PDF](fusion50/brca_fusion50_16shot_attribution.pdf) |
| CRC | [PNG](crc_lr_16shot_attribution.png) · [SVG](crc_lr_16shot_attribution.svg) · [PDF](crc_lr_16shot_attribution.pdf) | [PNG](fusion50/crc_fusion50_16shot_attribution.png) · [SVG](fusion50/crc_fusion50_16shot_attribution.svg) · [PDF](fusion50/crc_fusion50_16shot_attribution.pdf) |
| BLCA | [PNG](blca_lr_16shot_attribution.png) · [SVG](blca_lr_16shot_attribution.svg) · [PDF](blca_lr_16shot_attribution.pdf) | [PNG](fusion50/blca_fusion50_16shot_attribution.png) · [SVG](fusion50/blca_fusion50_16shot_attribution.svg) · [PDF](fusion50/blca_fusion50_16shot_attribution.pdf) |

The [FusionVal model index](fusionval/README.md) links to **56 more charts**:
one for every cancer, image architecture, and encoder. Each chart labels its
saved TME fusion weight for folds 0–4. Fusion50 needs only one chart per cancer
because all seven architectures and both encoders have identical TME scores.
The Native image-only variant has zero TME attribution by construction, so a
Native bar chart would contain only zeros.

PathoTME-LR reads TME measurements only. Fusion50 and FusionVal use an image
prediction, but each TME feature score is a fixed weight times the matching LR
score; the image prediction stays fixed during feature replacement. These
figures do **not** measure image-dependent TME effects. The learned PathoTME
adapter is still being scored and is absent from this complete-fold gallery.
Attribution describes model sensitivity, not accuracy, confidence intervals,
or a causal effect. Biological-group scores jointly replace their member
measurements and are not sums of individual scores.

The figures are generated entirely from the published aggregate
[rankings](../feature_rankings.csv) and [fold scores](../fold_feature_scores.csv),
without patient-level data. PNGs are 300 dpi; SVGs and PDFs are vector files.
To reproduce them with Python and Matplotlib, run from the repository root:

```bash
python docs/data/attribution-16shot-20261004/figures/make_figures.py
python docs/data/attribution-16shot-20261004/figures/make_fusion_figures.py
```
