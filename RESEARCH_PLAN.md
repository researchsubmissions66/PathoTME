# PathoTME research plan

## 1. Motivation

Beyond MAPLE and HiVE-MIL, OpenTME can fit into several distinct parts of the
pathology vision-language and multiple-instance-learning pipeline. The central
design principle is to stop treating OpenTME only as a source of extra features
and instead identify the *function* that tumor-microenvironment knowledge
performs.

| Role of OpenTME | Natural methods | Basic idea |
| --- | --- | --- |
| Semantic prompting | SLIP, SLDPC, PathPT | Use biologically grounded prompts. |
| Dynamic semantic retrieval | MUSE | Retrieve concepts based on the phenotype of each slide. |
| Prototype construction | Libra-MIL, ViLa-MIL | Build TME-grounded visual and textual prototypes. |
| Patch/token selection | FOCUS | Select regions relevant to the slide's TME phenotype. |
| Hierarchical concepts | MAPLE, HiVE-MIL, MGPATH | Organize coarse and fine TME-aware concepts. |
| Privileged supervision | Almost any MIL/VLM | Use OpenTME as a teacher during training only. |
| Agentic evidence | CPathAgent-style system | Query OpenTME when biological evidence is needed. |

These roles are not mutually exclusive. A general PathoTME model could combine
intermediate phenotype supervision with tokens, retrieval, prototypes, or
evidence localization.

## 2. MUSE + OpenTME: phenotype-guided semantic retrieval

MUSE is a particularly natural integration point because it is motivated by
the limitations of static class descriptions. It introduces sample-wise
semantic adaptation and retrieves from a pathological-description knowledge
base.

Conceptually, the original flow is:

$$
\mathrm{WSI}
\rightarrow
\text{sample-specific semantic representation}
\rightarrow
\text{relevant descriptions}.
$$

PathoTME could introduce a quantitative biological retrieval signal:

$$
\mathrm{OpenTME}_i
\rightarrow
\text{TME-relevant descriptions}.
$$

For example:

```text
Slide phenotype:
  high lymphocyte density
  high TLS score
  high tumor-immune proximity

                    ↓

Retrieved concepts:
  prominent lymphoid aggregates
  tumor-infiltrating lymphocytes
  organized lymphoid structures
  immune-rich tumor interface
```

The retrieved TME semantics can then interact with the visual patches:

$$
\text{retrieved TME semantics}
\leftrightarrow
\text{visual patches}.
$$

This creates a cleaner hypothesis than simple feature concatenation:

> Can quantitative phenotype-guided semantic retrieval outperform visual-only
> semantic retrieval?

## 3. Libra-MIL + OpenTME: phenotype-grounded prototypes

Libra-MIL constructs task-specific textual pathological-entity prototypes,
learns corresponding visual prototypes, and aligns the two. OpenTME provides a
natural source of biological structure for those prototypes.

Instead of relying only on:

$$
\text{LLM description}
\rightarrow
\text{text prototype},
$$

PathoTME could define explicit phenotype-grounded prototypes such as:

$$
P_{\mathrm{immune}},\quad
P_{\mathrm{stromal}},\quad
P_{\mathrm{tumor}},\quad
P_{\mathrm{TLS}},\quad
P_{\mathrm{tumor\text{-}immune}}.
$$

Visual patches would be encouraged to organize around these biologically
meaningful concepts. OpenTME could also determine slide-specific prototype
mixture weights. For slide $i$:

$$
z_i =
0.5P_{\mathrm{immune}}
+0.2P_{\mathrm{stromal}}
+0.1P_{\mathrm{TLS}}
+\cdots.
$$

This direction can be framed as **phenotype-grounded multimodal prototype
learning**. Its prototypes should support both quantitative evaluation and
biological interpretation.

## 4. FOCUS + OpenTME: biological patch selection

FOCUS progressively removes visually redundant patches and uses language
prompts to identify semantically relevant regions before neighbor-aware token
filtering. OpenTME could specify which kinds of regions are biologically
important for a particular slide.

The generic pathway:

$$
\text{class prompt}
\rightarrow
\text{important patches}
$$

would become:

$$
\text{slide TME phenotype}
\rightarrow
\text{phenotype-specific patch importance}.
$$

For example:

```text
OpenTME phenotype:
  high tumor-immune interaction

                    ↓

Search the WSI for:
  tumor/immune boundaries
  dense lymphocyte infiltration
  lymphoid aggregates

                    ↓

Retain phenotype-relevant patches
```

This yields a **TME-guided visual compressor** and creates a direct way to test
whether the patches retained by the model correspond to quantitative spatial
phenotypes.

## 5. ViLa-MIL + OpenTME: dual-scale TME concepts

ViLa-MIL provides two useful ingredients: dual-scale pathology prompts and
prototype-guided patch aggregation. OpenTME can ground both scales.

Candidate low-resolution concepts include:

- tumor-stroma organization;
- TLS distribution; and
- invasive-margin organization.

Candidate high-resolution concepts include:

- lymphocyte density;
- tumor-cell morphology; and
- immune-cell composition.

The resulting conditioning can be written as:

$$
\mathrm{OpenTME}_i
\rightarrow
\begin{cases}
T_{\mathrm{low},i},\\
T_{\mathrm{high},i},
\end{cases}
$$

where the two sets of concepts guide the corresponding prototype decoders and
patch aggregation pathways.

## 6. PathPT, SLIP, and SLDPC: prompt supervision

Prompt-oriented methods offer a simpler integration path:

- **SLIP** uses pathology prior knowledge to identify local tissue types and
  learns prompts from few slide labels.
- **PathPT** learns task-adaptive prompts and creates tile-level pseudo-labels
  from slide-level labels.
- **SLDPC** uses two-stage slide-level dual-prompt tuning with a largely frozen
  vision-language model.

OpenTME could contribute an explicit TME prompt component:

$$
P = P_{\mathrm{learnable}} + P_{\mathrm{TME}},
$$

or supervise prompt learning:

$$
P_{\mathrm{learned}}
\approx
\phi(\text{TME phenotype}).
$$

This is a useful baseline and may be straightforward to prototype. However, it
is less scientifically distinctive than privileged supervision, phenotype
tokens, or biologically grounded prototype learning.

## 7. Architecture-independent OpenTME teacher

The most general direction is to treat OpenTME as privileged biological
supervision. A WSI model learns a phenotype representation:

$$
z_{\mathrm{TME}} = f_{\theta}(\mathrm{WSI}),
$$

and a phenotype head predicts the OpenTME targets:

$$
\widehat{t}_i = g(z_{\mathrm{TME}}),
\qquad
t_i = \mathrm{OpenTME}_i.
$$

The training objective is:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{classification}}
+
\lambda_{\mathrm{TME}}\mathcal{L}_{\mathrm{TME}}.
$$

At inference time, only the WSI is required:

$$
\boxed{\text{WSI only at inference}}.
$$

This formulation can be attached to ABMIL, TransMIL, CLAM, ViLa-MIL, MAPLE,
HiVE-MIL, FOCUS, and many other slide models. The larger scientific question
is:

> Can quantitative TME phenotypes provide biological intermediate supervision
> for WSI learning?

### Important ablations

1. Classification only versus classification plus TME supervision.
2. Direct multi-target regression versus phenotype-factor supervision.
3. All OpenTME targets versus immune, stromal, tumor, and spatial subsets.
4. Shared versus separate slide representations for the task and phenotype
   heads.
5. OpenTME available during training and inference versus training only.
6. Randomized or cohort-matched phenotype controls to test whether gains come
   from meaningful biological supervision.

## 8. TME-aware pathology agent

An agentic pathway has a different primary goal from predictive performance:

$$
\text{agent}
\rightarrow
\text{query OpenTME}
\rightarrow
\text{form a biological hypothesis}
\rightarrow
\text{locate WSI evidence}
\rightarrow
\text{verify the hypothesis}.
$$

For example:

```text
Agent hypothesis:
  "This slide has a high TLS phenotype; locate supporting evidence."

Evidence tools:
  MAPLE entity search
  HiVE-MIL hierarchical search
  FOCUS phenotype-guided region selection
```

This direction targets **evidence-grounded reasoning**, not merely AUROC. It
could be layered on top of the predictive model after phenotype learning and
spatial localization have been validated.

## 9. Unifying architecture: TME phenotype tokens

OpenTME contains approximately 4,500 measurements. Converting every variable into an
independent text prompt would be unwieldy and would ignore their correlation
structure. Instead, a phenotype encoder could compress them into a small set
of TME tokens:

$$
t_{\mathrm{OpenTME}} \in \mathbb{R}^{4500}
\xrightarrow{E_{\mathrm{TME}}}
\{p_1,p_2,\ldots,p_K\}.
$$

The tokens could be structured or weakly supervised so that they correspond to
major biological axes:

$$
\begin{aligned}
p_1 &= \text{immune phenotype},\\
p_2 &= \text{stromal phenotype},\\
p_3 &= \text{tumor phenotype},\\
p_4 &= \text{spatial-interaction phenotype},\\
p_5 &= \text{invasive-margin phenotype}.
\end{aligned}
$$

These tokens can interact directly with WSI patch tokens through
cross-attention:

$$
\boxed{
\text{TME phenotype tokens}
\leftrightarrow
\text{WSI patch tokens}
}.
$$

This is a general **TME-conditioned WSI learner**, rather than a narrow
modification of one existing method. It can borrow:

- hierarchical reasoning from HiVE-MIL;
- entity reasoning from MAPLE;
- prototype learning from Libra-MIL;
- token pruning from FOCUS; and
- semantic retrieval from MUSE.

### Two inference modes

The architecture should explicitly distinguish two scientifically different
settings:

1. **Privileged-supervision mode:** OpenTME is used only during training; a WSI
   encoder predicts latent phenotype tokens at inference.
2. **Conditioned mode:** measured OpenTME variables are provided at both
   training and inference.

Results from these settings must not be mixed. The privileged setting tests
representation learning, whereas the conditioned setting tests multimodal
prediction with additional patient information.

## 10. Ranked research directions

### 1. OpenTME as privileged biological supervision

$$
\mathrm{WSI}
\rightarrow
\text{learn TME phenotypes}
\rightarrow
\text{classification}.
$$

This provides the strongest general machine-learning story and applies across
architectures.

### 2. TME phenotype tokens plus visual tokens

$$
\mathrm{TME}
\leftrightarrow
\mathrm{WSI}.
$$

This offers the greatest opportunity to design a genuinely new architecture.

### 3. TME-conditioned prototype learning

This Libra-MIL/ViLa-MIL-style direction is clean, interpretable, and easy to
connect to explicit biological concepts.

### 4. TME-guided semantic retrieval

The MUSE-style direction is natural and likely relatively easy to prototype.

### 5. TME-grounded hierarchy

HiVE-MIL/MAPLE-style reasoning is strong but more closely resembles an
extension of existing architectures.

### 6. TME-guided FOCUS/token selection

This is a promising secondary direction with a direct spatial-interpretability
story.

### 7. Plain TME-conditioned prompts

This is probably the easiest initial baseline, but the least distinctive
scientific contribution.

## 11. Initial execution roadmap

### Phase 0: data and leakage audit

- Define the unit of analysis and join keys between WSIs, patients, clinical
  outcomes, and OpenTME measurements.
- Inventory missingness, repeated slides, multiple samples per patient, cohort
  coverage, and phenotype distributions.
- Freeze patient-disjoint train/validation/test partitions before fitting any
  phenotype encoder, normalization, target factorization, prompt retrieval, or
  prototype selection.
- Fit all target preprocessing on training patients only.
- Decide which OpenTME measurements are targets, covariates, or unavailable at
  inference.

### Phase 1: privileged-supervision baseline

- Start from ABMIL and TransMIL with a shared WSI encoder.
- Add multi-task heads for clinical classification and grouped TME targets.
- Compare direct targets with low-dimensional phenotype factors.
- Establish whether TME supervision improves held-out classification,
  calibration, and phenotype prediction.

### Phase 2: phenotype-token architecture

- Learn a small token set from grouped OpenTME targets.
- Predict those tokens from WSI features for WSI-only inference.
- Add bidirectional or phenotype-to-patch cross-attention.
- Test token disentanglement, stability, and correspondence to spatial regions.

### Phase 3: specialized integrations

- Add TME-grounded prototypes inspired by Libra-MIL and ViLa-MIL.
- Add phenotype-guided retrieval inspired by MUSE.
- Add phenotype-guided patch selection inspired by FOCUS.
- Treat each integration as a separately named ablation rather than silently
  changing an existing published method.

### Phase 4: interpretation and agentic evidence

- Generate phenotype-specific attention or relevance maps.
- Quantify whether selected regions correspond to measured TME phenotypes.
- Build an agent that states a phenotype hypothesis, retrieves supporting WSI
  regions, and reports both positive and contradictory evidence.

## 12. Evaluation principles

- Use patient-disjoint splits and keep the test set final.
- Never use validation or test OpenTME measurements to fit normalization,
  factors, prototypes, prompts, retrieval banks, or stopping rules.
- Report WSI-only and OpenTME-conditioned inference as separate settings.
- Compare against equal-capacity multi-task and random-target controls.
- Report predictive metrics, calibration, phenotype fidelity, localization,
  missing-modality robustness, and subgroup performance.
- Test whether phenotype-token meaning is stable across folds and cohorts.
- Distinguish extensions from faithful reproductions of MAPLE, HiVE-MIL, MUSE,
  Libra-MIL, ViLa-MIL, FOCUS, PathPT, SLIP, and SLDPC.

## 13. Central thesis

The broad contribution is not simply “OpenTME + MAPLE” or “OpenTME +
HiVE-MIL.” It is:

$$
\boxed{
\textbf{Quantitative tumor phenotypes as intermediate biological supervision}
\\
\textbf{for learning WSI representations}
}.
$$

This framing leaves room for a new general method while allowing established
architectures to serve as baselines, ablations, or sources of reusable design
ideas.

## References

1. [MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide
   Image Classification](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_MUSE_Harnessing_Precise_and_Diverse_Semantics_for_Few-Shot_Whole_Slide_CVPR_2026_paper.html)
2. [Libra-MIL: Multimodal Prototypes Stereoscopic Infused with Task-specific
   Language Priors for Few-shot Whole Slide Image
   Classification](https://arxiv.org/abs/2511.07941)
3. [FOCUS: Knowledge-enhanced Adaptive Visual Compression for Few-shot Whole
   Slide Image Classification](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_FOCUS_Knowledge-enhanced_Adaptive_Visual_Compression_for_Few-shot_Whole_Slide_Image_CVPR_2025_paper.html)
4. [ViLa-MIL](https://github.com/Jiangbo-Shi/ViLa-MIL)
5. [Slide-Level Prompt Learning with Vision Language Models for Few-Shot
   Multiple Instance Learning in Histopathology](https://arxiv.org/abs/2503.17238)
6. [Boosting pathology foundation models via few-shot prompt-tuning for rare
   cancer subtyping](https://doi.org/10.1038/s41467-026-71715-2)
7. [SLDPC: Slide-Level Dual-Prompt Collaboration for few-shot whole slide image
   classification](https://www.sciencedirect.com/science/article/pii/S0895611126000716)
