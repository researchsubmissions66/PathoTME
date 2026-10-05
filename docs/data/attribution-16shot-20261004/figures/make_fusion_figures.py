"""Render compact 16-shot fusion charts from the published aggregate CSVs."""

import csv
from collections import defaultdict
from pathlib import Path

from make_figures import COHORTS, OUTPUT, ROOT, plt, render_chart


METHOD_NAMES = {
    "vila_mil": "ViLa-MIL",
    "mgpath": "MGPATH",
    "focus": "FOCUS",
    "muse": "MUSE",
    "hive_mil": "HiVE-MIL",
    "dyko": "DyKo",
    "mscpt": "MSCPT",
}
ENCODER_NAMES = {"plip": "PLIP", "clip-rn50": "CLIP-RN50"}
VARIANTS = ("PathoTME-LR", "PathoTME-Fusion50", "PathoTME-FusionVal")


def read_tables():
    rankings = defaultdict(list)
    with (ROOT / "feature_rankings.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] not in VARIANTS or row["class_index"] != "1":
                continue
            assert row["shots"] == "16" and row["complete_five_fold"] == "True"
            key = (row["variant"], row["cohort"], row["method"], row["encoder"], row["level"])
            rankings[key].append(row)

    folds = {}
    with (ROOT / "fold_feature_scores.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] not in VARIANTS or row["class_index"] != "1":
                continue
            key = (row["variant"], row["cohort"], row["method"], row["encoder"],
                   row["level"], row["feature_name"], int(row["fold"]))
            assert key not in folds
            folds[key] = row
    return rankings, folds


def model_conditions(rankings, cohort, variant):
    models = sorted({(method, encoder) for kind, cancer, method, encoder, level in rankings
                     if kind == variant and cancer == cohort})
    assert len(models) == 14 and all(method in METHOD_NAMES and encoder in ENCODER_NAMES
                                     for method, encoder in models)
    return models


def panel_tables(rankings, folds, variant, cohort, method, encoder):
    selected_rankings = {}
    selected_folds = {}
    for level in ("group", "feature"):
        rows = rankings[variant, cohort, method, encoder, level]
        assert len(rows) >= 10
        selected_rankings[cohort, level] = rows
        for row in rows:
            for fold in range(5):
                key = (variant, cohort, method, encoder, level, row["feature_name"], fold)
                selected_folds[cohort, level, row["feature_name"], fold] = folds[key]
    return selected_rankings, selected_folds


def alpha_by_fold(rankings, folds, cohort, method, encoder):
    """Recover the saved TME weight from public fusion/LR fold score ratios."""
    weights = []
    for fold in range(5):
        ratios = []
        for row in rankings["PathoTME-FusionVal", cohort, method, encoder, "group"]:
            feature = row["feature_name"]
            native_key = ("PathoTME-LR", cohort, "none", "none", "group", feature, fold)
            fused_key = ("PathoTME-FusionVal", cohort, method, encoder, "group", feature, fold)
            reference = float(folds[native_key]["mean_absolute_pp"])
            if reference > 1e-8:
                ratios.append(float(folds[fused_key]["mean_absolute_pp"]) / reference)
        assert ratios and max(ratios) - min(ratios) < 1e-8
        alpha = min((0.0, 0.25, 0.5, 0.75, 1.0), key=lambda value: abs(value - ratios[0]))
        assert abs(alpha - ratios[0]) < 1e-8
        weights.append(alpha)
    return tuple(weights)


def verify_fusion50_invariance(rankings, folds, cohort, models):
    reference_method, reference_encoder = models[0]
    for level in ("group", "feature"):
        reference_rows = {
            row["feature_name"]: row for row in
            rankings["PathoTME-Fusion50", cohort, reference_method, reference_encoder, level]
        }
        for method, encoder in models:
            current_rows = {
                row["feature_name"]: row for row in
                rankings["PathoTME-Fusion50", cohort, method, encoder, level]
            }
            assert current_rows.keys() == reference_rows.keys()
            for feature, row in current_rows.items():
                base = reference_rows[feature]
                for column in ("mean_absolute_pp", "fold_sd_absolute_pp", "mean_signed_pp"):
                    assert abs(float(row[column]) - float(base[column])) < 1e-10
                for fold in range(5):
                    current = folds["PathoTME-Fusion50", cohort, method, encoder, level, feature, fold]
                    baseline = folds["PathoTME-Fusion50", cohort, reference_method,
                                     reference_encoder, level, feature, fold]
                    assert abs(float(current["mean_absolute_pp"]) -
                               float(baseline["mean_absolute_pp"])) < 1e-10


def write_fusionval_index(items):
    output = OUTPUT / "fusionval" / "README.md"
    lines = [
        "# 16-shot FusionVal TME attribution charts",
        "",
        "Each row names the image architecture and encoder used for the saved",
        "prediction. The chart plots the top 10 TME groups and top 10 individual",
        "features for that exact condition. Its TME feature score equals the",
        "saved validation-selected TME weight (α) times the matched PathoTME-LR",
        "score. The image prediction stays fixed under the TME intervention;",
        "these charts do not measure image-dependent feature effects.",
        "",
        "The α values below are recovered from the published per-fold fusion/LR",
        "score ratios and checked across all groups. Fold values and sample SDs",
        "come from the published aggregate CSVs. Native has structural zero",
        "TME attribution; learned-adapter attribution is still incomplete.",
        "",
    ]
    for cohort, (title, _) in COHORTS.items():
        lines += [f"## {title}", "",
                  "| Architecture | Encoder | α, folds 0–4 | PNG | SVG | PDF |",
                  "|---|---|---|---|---|---|"]
        for method, encoder, weights, stem in sorted(items[cohort]):
            prefix = f"{cohort}/{stem}"
            alpha = ", ".join(f"{weight:.2f}" for weight in weights)
            lines.append(
                f"| {METHOD_NAMES[method]} | {ENCODER_NAMES[encoder]} | {alpha} | "
                f"[PNG]({prefix}.png) | [SVG]({prefix}.svg) | [PDF]({prefix}.pdf) |"
            )
        lines.append("")
    output.write_text("\n".join(lines).rstrip() + "\n")


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "pdf.fonttype": 42,
        "svg.fonttype": "none", "savefig.bbox": "standard",
    })
    rankings, folds = read_tables()
    index_items = defaultdict(list)
    for cohort in COHORTS:
        models = model_conditions(rankings, cohort, "PathoTME-Fusion50")
        assert models == model_conditions(rankings, cohort, "PathoTME-FusionVal")
        verify_fusion50_invariance(rankings, folds, cohort, models)
        method, encoder = models[0]
        selected_rankings, selected_folds = panel_tables(
            rankings, folds, "PathoTME-Fusion50", cohort, method, encoder)
        render_chart(
            cohort, selected_rankings, selected_folds,
            variant_label="PathoTME-Fusion50",
            scope_line="All seven image architectures and both encoders have identical TME scores",
            note_line="Fusion50 score = 0.5 × LR score; image prediction fixed. Separate x scales; no causal claim.",
            stem=f"{cohort}_fusion50_16shot_attribution", output_dir=OUTPUT / "fusion50",
        )
        print(f"Rendered Fusion50 {cohort}", flush=True)

        for method, encoder in models:
            selected_rankings, selected_folds = panel_tables(
                rankings, folds, "PathoTME-FusionVal", cohort, method, encoder)
            weights = alpha_by_fold(rankings, folds, cohort, method, encoder)
            stem = f"{cohort}_{method}_{encoder}_fusionval_16shot_attribution"
            render_chart(
                cohort, selected_rankings, selected_folds,
                variant_label="PathoTME-FusionVal",
                scope_line=f"{METHOD_NAMES[method]} · {ENCODER_NAMES[encoder]} · α across folds: "
                           + ", ".join(f"{weight:.2f}" for weight in weights),
                note_line="FusionVal score = saved α × LR score; image prediction fixed. Separate x scales; no causal claim.",
                stem=stem, output_dir=OUTPUT / "fusionval" / cohort,
            )
            index_items[cohort].append((method, encoder, weights, stem))
            print(f"Rendered FusionVal {cohort} {method} {encoder}", flush=True)
    write_fusionval_index(index_items)


if __name__ == "__main__":
    main()
