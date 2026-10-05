"""Render compact publication-ready 16-shot PathoTME-LR bar charts.

Run with a Python environment containing Matplotlib:
    python figures/make_figures.py

Only the published aggregate CSVs are read. No patient-level data is needed.
"""

import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
COHORTS = {
    "nsclc": ("NSCLC", "LUSC"),
    "brca": ("BRCA", "ILC"),
    "crc": ("CRC", "MUCINOUS"),
    "blca": ("BLCA", "PAPILLARY"),
}
GROUP_LABELS = {
    "low/global_tissue": "Global tissue",
    "low/tls": "Tertiary lymphoid structures",
    "low/carcinoma_geometry": "Carcinoma geometry",
    "low/stroma_geometry": "Stroma geometry",
    "low/compartment_extent": "Compartment extent",
    "low/tissue_composition": "Tissue composition",
    "low/tumor_core_composition": "Tumor-core composition",
    "low/invasive_margin_composition": "Invasive-margin composition",
}
CELL_LABELS = {
    "FIBROBLASTS": "Fibroblast",
    "LYMPHOCYTES": "Lymphocyte",
    "MACROPHAGES": "Macrophage",
    "PLASMA_CELLS": "Plasma-cell",
    "FIBROBLAST": "Fibroblast",
    "LYMPHOCYTE": "Lymphocyte",
    "MACROPHAGE": "Macrophage",
    "PLASMA_CELL": "Plasma-cell",
    "GRANULOCYTE": "Granulocyte",
    "ENDOTHELIAL_CELL": "Endothelial-cell",
    "CARCINOMA_CELL": "Carcinoma-cell",
}
TISSUE_LABELS = {
    "STROMA": "Stromal",
    "CARCINOMA": "Carcinoma",
    "NECROSIS": "Necrosis",
    "EPITHELIAL_TISSUE": "Epithelial-tissue",
    "OUTER_INVASIVE_MARGIN": "Outer invasive-margin",
    "INNER_INVASIVE_MARGIN": "Inner invasive-margin",
    "TUMOR_CORE": "Tumor-core",
    "VALID_TISSUE": "Valid-tissue",
    "VESSEL": "Vessel",
}
REGION_LABELS = {
    "INNER_INVASIVE_MARGIN": "inner margin",
    "TUMOR_CORE": "tumor core",
}


def display_label(name, level):
    """Short display label; exact feature names remain in the linked CSVs."""
    if level == "group":
        if name in GROUP_LABELS:
            return GROUP_LABELS[name]
        part, kind = name.split("/", 1)
        if part == "high" and kind.startswith("spatial_"):
            cell = kind.removeprefix("spatial_").upper().removesuffix("S")
            if kind == "spatial_plasma_cells":
                cell = "PLASMA_CELL"
            return f"{CELL_LABELS[cell]} spatial context"
        if part == "high" and kind.startswith("cell_"):
            cell = kind.removeprefix("cell_").upper()
            return f"{CELL_LABELS[cell]} measurements"
        raise ValueError(f"Unmapped group: {name}")

    match = re.fullmatch(
        r"AVG_MIN_DISTANCE_OF_(.+)_AROUND_CARCINOMA_CELL_IN_WHOLE_TUMOR_REGION_(\d+)",
        name,
    )
    if match:
        return f"{CELL_LABELS[match[1]]}–carcinoma distance ({match[2]})"
    match = re.fullmatch(
        r"RATIO_OF_(.+)_AROUND_CARCINOMA_CELL_IN_WHOLE_TUMOR_REGION_(\d+)",
        name,
    )
    if match:
        return f"{CELL_LABELS[match[1]]} proximity ratio ({match[2]})"
    match = re.fullmatch(r"CELL_(DENSITY|PERCENTAGE)_(.+)_IN_(.+)", name)
    if match:
        measure = "density" if match[1] == "DENSITY" else "percentage"
        return f"{CELL_LABELS[match[2]]} {measure} · {REGION_LABELS[match[3]]}"
    if name.startswith("RELATIVE_AREA_"):
        tissue = name.removeprefix("RELATIVE_AREA_")
        if "_IN_" in tissue:
            tissue, region = tissue.rsplit("_IN_", 1)
            return f"{TISSUE_LABELS[tissue]} area · {REGION_LABELS[region]}"
        return f"{TISSUE_LABELS[tissue]} area"
    if name.startswith("AVG_SOLIDICITY_"):
        return f"{TISSUE_LABELS[name.removeprefix('AVG_SOLIDICITY_')]} solidity"
    if name.startswith("AVG_ECCENTRICITY_"):
        return f"{TISSUE_LABELS[name.removeprefix('AVG_ECCENTRICITY_')]} eccentricity"
    if name.startswith("LARGEST_ROUNDNESS_"):
        return f"Largest {TISSUE_LABELS[name.removeprefix('LARGEST_ROUNDNESS_')].lower()} roundness"
    if name.startswith("LOG1P_COUNT_TLS_"):
        return f"{name.removeprefix('LOG1P_COUNT_TLS_').title()} TLS count (log1p)"
    if name.startswith("REGIONS_PER_MM2_VALID_TISSUE_"):
        return f"{TISSUE_LABELS[name.removeprefix('REGIONS_PER_MM2_VALID_TISSUE_')]} regions per mm²"
    if name.startswith("TLS_") and name.endswith("_PRESENT"):
        return f"{name.removeprefix('TLS_').removesuffix('_PRESENT').title()} TLS present"
    raise ValueError(f"Unmapped feature: {name}")


def read_results():
    rankings = defaultdict(list)
    with (ROOT / "feature_rankings.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] == "PathoTME-LR" and row["class_index"] == "1":
                assert row["shots"] == "16" and row["complete_five_fold"] == "True"
                rankings[(row["cohort"], row["level"])].append(row)

    folds = {}
    with (ROOT / "fold_feature_scores.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] == "PathoTME-LR" and row["class_index"] == "1":
                key = (row["cohort"], row["level"], row["feature_name"], int(row["fold"]))
                assert key not in folds
                folds[key] = row
    return rankings, folds


def plot_panel(axis, cohort, level, rankings, folds):
    top = sorted(
        rankings[(cohort, level)],
        key=lambda row: (-float(row["mean_absolute_pp"]), row["feature_name"]),
    )[:10]
    labels = [display_label(row["feature_name"], level) for row in top]
    assert len(set(labels)) == len(labels), (cohort, level, labels)
    means = [float(row["mean_absolute_pp"]) for row in top]
    standard_deviations = [float(row["fold_sd_absolute_pp"]) for row in top]
    fold_values = [
        [float(folds[(cohort, level, row["feature_name"], fold)]["mean_absolute_pp"])
         for fold in range(5)]
        for row in top
    ]
    for mean, standard_deviation, values in zip(means, standard_deviations, fold_values):
        assert abs(mean - sum(values) / 5) < 1e-10
        assert abs(standard_deviation - statistics.stdev(values)) < 1e-10

    accent = "#147A7E" if level == "group" else "#3A609B"
    pale = "#A5D5D2" if level == "group" else "#B9CBE5"
    axis.barh(range(10), means, height=0.66,
              color=[accent] + [pale] * 9, edgecolor="none", zorder=2)
    x_max = max(max(means) * 1.07, 0.4)
    axis.set_xlim(0, x_max)
    axis.set_yticks(range(10), labels=labels)
    axis.invert_yaxis()
    axis.tick_params(axis="y", length=0, pad=8, labelsize=8.7, colors="#26374A")
    axis.tick_params(axis="x", length=0, labelsize=8.5, colors="#536578")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    axis.grid(axis="x", color="#E3E9EF", linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.spines["bottom"].set_color("#C9D3DD")
    axis.text(1.018, 1.03, "MEAN ± SD", transform=axis.transAxes,
              ha="left", va="bottom", fontsize=8.2, fontweight="bold",
              color="#52687B", clip_on=False)
    for index, (mean, standard_deviation) in enumerate(zip(means, standard_deviations)):
        axis.text(1.018, index, f"{mean:.2f} ± {standard_deviation:.2f}",
                  transform=axis.get_yaxis_transform(), ha="left", va="center",
                  fontsize=8.5, fontweight="semibold", color="#21384E",
                  clip_on=False, zorder=5)
    axis.set_title(
        "A  Biological feature groups" if level == "group" else "B  Individual measurements",
        loc="left", fontsize=10.5, fontweight="bold", color="#172B41", pad=13,
    )
    axis.set_xlabel("Mean absolute probability change (pp)",
                    fontsize=8.8, color="#334D63", labelpad=6)


def render_chart(cohort, rankings, folds, *, variant_label, scope_line,
                 note_line, stem, output_dir=OUTPUT):
    title, positive_class = COHORTS[cohort]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 8.6))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.38, right=0.80, top=0.82 if scope_line else 0.85,
                        bottom=0.14, hspace=0.40)
    fig.text(0.055, 0.955, f"{title}  |  TME feature attribution",
             fontsize=16, fontweight="bold", color="#172B41")
    fig.text(0.055, 0.922,
             f"{variant_label} · 16-shot · predicted {positive_class} probability · five held-out folds",
             fontsize=9.7, color="#51677C")
    if scope_line:
        fig.text(0.055, 0.890, scope_line, fontsize=9.2, color="#536578")
    for axis, level in zip(axes, ("group", "feature")):
        plot_panel(axis, cohort, level, rankings, folds)
    fig.text(0.055, 0.075, "Bars show mean absolute patient-level scores; SD describes the five fold means (sample SD).",
             fontsize=8.1, color="#536578")
    fig.text(0.055, 0.053, note_line, fontsize=8.1, color="#536578")

    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        target = output_dir / f"{stem}.{extension}"
        fig.savefig(
            target, dpi=300, facecolor="white",
            metadata={"Title": f"{title} {variant_label} 16-shot TME feature attribution"},
        )
        if extension == "svg":
            # Matplotlib adds trailing spaces to multiline SVG path data.
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


def make_figure(cohort, rankings, folds):
    render_chart(
        cohort, rankings, folds, variant_label="PathoTME-LR", scope_line="",
        note_line="Separate x scales. TME-only model with no image input; sensitivity is not accuracy or causality.",
        stem=f"{cohort}_lr_16shot_attribution",
    )


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "standard",
    })
    rankings, folds = read_results()
    for cohort in COHORTS:
        make_figure(cohort, rankings, folds)
        print(f"Rendered {cohort}")


if __name__ == "__main__":
    main()
