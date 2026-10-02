# TME-guided ViLa-MIL

## Scientific question

This experiment tests whether slide-specific quantitative TME phenotypes can
guide ViLa-MIL's internal visual aggregation, beyond merely contributing an
independent prediction. It is a **PathoTME extension**, not a reproduction of
an upstream ViLa-MIL component.

Native ViLa-MIL is loaded from the validation-selected checkpoint for the same
fold and frozen. Its CLIP-RN50 5x/10x patch bags, two-scale NSCLC prompt bank,
16 learned image centers, label order, and visual/text attention blocks are
unchanged. Only the new TME adapter is optimized.

## Quantitative tokenization

The input is the actual 62-value OpenTME vector for the current slide. Feature
names determine a fixed biological schema but are not embedded as text.

The vector is transformed into 16 semantic tokens:

| ViLa scale | Tokens | Quantitative content |
| --- | ---: | --- |
| Low, 5x | 5 | global tissue, compartment extent, tumor-core composition, invasive-margin composition, TLS |
| High, 10x | 11 | seven cell types and four cell-to-carcinoma spatial-interaction groups |

Each cell token contains percentage and density in tumor core and inner
invasive margin. Each spatial token contains interaction ratio and average
minimum distance at 20 and 40 units. Encoders are shared within the cell and
spatial families and use learned identity embeddings to retain the biological
entity. This gives finer, auditable attention than either one global TME token
or four coarse group tokens while keeping capacity small for 16-shot training.

Percentages are first converted to fractions; density and positive-distance
measurements receive `log1p`. Median imputation and z-standardization are fitted
using the 32 training slides of each fold only. Validation and diagnostic
holdout values use those frozen statistics. The in-model width remains 62;
missingness indicators are deliberately omitted from the primary guided test.

## Insertion point

The token embeddings and ViLa image centers are projected to a 128-dimensional
adapter space. A shared four-head cross-attention block updates the low- and
high-scale center queries separately:

```text
62 quantitative values
       |
       +--> 5 low-scale TME tokens ----+
       |                                | cross-attention
       +--> 11 high-scale TME tokens ---+ into ViLa image centers
                                                |
                                      conditioned centers
                                                |
                                  native center-to-patch attention
                                                |
                                    native prompt-to-image attention
                                                |
                                             logits
```

The conditioned center for scale `s` is

```text
C'_s = C + sigmoid(g_s) * ProjectOut(
         Attention(ProjectIn(C), T_s, T_s))
```

and `C'_s`, rather than `C`, becomes the query for ViLa's existing patch
attention. The two learned residual gates start at 0.10 to avoid overwhelming
the checkpoint at initialization.

## Controlled comparison

Two matched conditions are defined:

- `actual`: the correct slide-specific TME vector is supplied;
- `zero`: the identical adapter and parameter count are retained, but the
  standardized values are zeroed before tokenization.

Both begin from the exact same frozen ViLa checkpoint. The zero condition tests
whether any gain comes from slide-specific phenotype information rather than
the extra trainable adapter, global token identities, or changed optimization.
Existing native ViLa predictions are also included in every result file.

The immediate diagnostic uses exact-coverage folds 0, 2, and 3. Folds 1 and 4
are not reduced silently because each includes a slide unavailable in OpenTME.
A final five-fold result requires regenerating both native and guided runs over
the common 1,041-slide universe.

## Validation and launch

Validate one condition without constructing a model:

```bash
python scripts/run_vila_guided.py \
  --experiment-config configs/vila_mil_tme_guided_nsclc_16shot.yaml \
  --fold 0 --tme-mode actual --check-only
```

Preview all six independent jobs:

```bash
scripts/launch_vila_guided.sh --dry-run
```

Submission requires an explicit launch decision:

```bash
scripts/launch_vila_guided.sh
```

Each job requests one A100 for 20 minutes and has no dependency on the existing
late-fusion canary. Results and compact adapter-only checkpoints are written to
`/path/to/shared/PathoTME-results/vila_mil_tme_guided/`.
