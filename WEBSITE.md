# PathoTME project website

The static website lives in `docs/`. It presents the TME conditioner, six
architecture-specific entry points, the core62/morph64 biological panels,
and the two-cohort study design. It intentionally contains no experimental
results, outcome tables, run-status summaries, or performance downloads.

## Preview

From the PathoTME project directory:

```bash
python3 -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open `http://localhost:8000`. When serving on the cluster, forward port 8000
over your existing SSH connection to preview in your local browser. The main
page also works by opening `docs/index.html` directly; it does not fetch data
at runtime.

## Files and editing

- `docs/index.html`: project landing page and accessible default content.
- `docs/protocol.html`: study design and methodological boundaries.
- `docs/style.css`: shared responsive styles; purple/navy colors and
  Outfit/Inter typography inspired by the reference-project project website.
- `docs/script.js`: navigation and keyboard-operable architecture/panel tabs.
- `docs/data/variant-names.json`: canonical condition-to-display-name mapping shared
  by the project documentation and research result exports. Keep the static
  variant definitions in both HTML pages consistent with this catalog.
- `docs/data/site-content.js`: editorial descriptions used by the explorers.
- `docs/data/panels.json`: ordered panel definitions exported from
  `pathotme/features.py` and `pathotme/brca_features.py`. No patient rows.
- `docs/assets/PathoTME_logo.png`: user-provided project logo, preserved
  unchanged and used in the title, navigation, footer, and favicon. The tissue
  illustration is a separate conceptual inline SVG, not a patient image or
  attention heatmap.
- `docs/assets/fonts/`: locally hosted Latin subsets of Inter and Outfit,
  with their SIL Open Font License files.

Keep the initial ViLa-MIL and core62 HTML descriptions in sync with the
editorial data when changing those defaults. The protocol supplies descriptions
of every architecture and both panels when JavaScript is disabled. No package
installation, build step, external font service, analytics, or training runtime
is required to serve the site.

## Hosting

The complete deployable directory is `docs/`. All local URLs are relative, so
the site can be hosted at a repository subpath. GitHub Pages can serve the
`main` branch's `/docs` directory; enabling Pages is a repository setting,
separate from pushing the files. Repository visibility is not changed by the
website. `.nojekyll` keeps the published files unchanged.

The repository URL is
https://github.com/researchsubmissions66/PathoTME.

## Content provenance

The website summarizes the current TCGA study and the architecture-specific
project notes. Cohort sizes describe the common study manifests, not measured
model performance. Panel definitions retain the pinned OpenTME revision and
source-column names. Earlier WSI-only distillation and external-cohort studies
remain documented elsewhere in the research project.

The local GitHub token is excluded by `.gitignore` and is never needed by the
site. Do not include it in a deployment directory.

## Browser validation

Checked in Chromium 148 at 320, 390, 768, 1024, and 1440 pixels. All six
architecture tabs, both panel totals, keyboard navigation, mobile menu, local
links, and the panel download passed. Reduced motion and the no-JavaScript
fallback were checked. No browser errors or horizontal page overflow remained.
Research runtime files were not changed by the website implementation.
