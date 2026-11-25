#!/usr/bin/env python3
"""Render feature-localization heatmaps from real OpenTME slide thumbnails.

The spatial color is a smoothed thumbnail segmentation mask. The TME-LR model
selects and annotates a slide-level feature; it does not supply pixel scores.
Only Pillow is needed beyond the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


COHORTS = {
    "brca": "breast_cancer",
    "nsclc": "lung_cancer",
    "crc": "colorectal_cancer",
    "blca": "bladder_cancer",
}

# Chosen among WTR area features that remain visible at thumbnail resolution.
# Region and tissue colors are the swatches in the OpenTME thumbnail legends.
FEATURES = {
    "brca": ("RELATIVE_AREA_OUTER_INVASIVE_MARGIN", "outer invasive margin", "outer", None),
    "nsclc": ("RELATIVE_AREA_STROMA_IN_INNER_INVASIVE_MARGIN", "stroma in inner invasive margin", "inner", "stroma"),
    "crc": ("RELATIVE_AREA_INNER_INVASIVE_MARGIN", "inner invasive margin", "inner", None),
    "blca": ("RELATIVE_AREA_OUTER_INVASIVE_MARGIN", "outer invasive margin", "outer", None),
}
REGION_COLORS = {
    "core": (5, 255, 188),
    "inner": (0, 151, 111),
    "outer": (214, 0, 110),
}
TISSUE_COLORS = {"stroma": (255, 204, 102)}


def font(size: int, bold: bool = False):
    names = ("DejaVuSans-Bold.ttf", "Arial Bold.ttf") if bold else ("DejaVuSans.ttf", "Arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def rows_by_slide(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["slide_id"]: row for row in csv.DictReader(handle) if row["slide_id"] in wanted}
    if rows.keys() != wanted:
        raise ValueError(f"Missing feature rows: {sorted(wanted - rows.keys())}")
    return rows


def model_for_cohort(root: Path, cohort: str) -> dict:
    found = []
    for path in root.glob("*/model.json"):
        model = json.loads(path.read_text(encoding="utf-8"))
        task = model["task"]
        if task["cohort"] == cohort and task["fold"] == 0:
            found.append(model)
    if len(found) != 1:
        raise ValueError(f"Expected one fold-0 TME-LR model for {cohort}, found {len(found)}")
    return found[0]


def source_images(root: Path, cohort: str, filename: str):
    stem = Path(filename).stem
    if Path(filename).name != filename:
        raise ValueError("Manifest filename must be a basename")
    folder = root / "data" / COHORTS[cohort] / "thumbnails" / stem
    wsi = Image.open(folder / "wsi.png").convert("RGB")
    region = Image.open(folder / "whole_tumor_region.png").convert("RGB")
    tissue = Image.open(folder / "tissue_segmentation.png").convert("RGB")
    if region.size != tissue.size or wsi.width != region.width or abs(wsi.height - region.height) > 1:
        raise ValueError(f"Thumbnail layers have incompatible sizes: {folder}")
    # One OpenTME WSI thumbnail is a single row taller than its annotation layers.
    if wsi.size != region.size:
        wsi = wsi.crop((0, 0, *region.size))
    return wsi, region, tissue


def thumbnail_hashes(root: Path, cohort: str, filename: str) -> dict[str, str]:
    folder = root / "data" / COHORTS[cohort] / "thumbnails" / Path(filename).stem
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in
            ("wsi.png", "whole_tumor_region.png", "tissue_segmentation.png")}


def compartment_mask(region: Image.Image, tissue: Image.Image, region_name: str,
                     tissue_name: str | None) -> Image.Image:
    width, height = region.size
    region_rgb = REGION_COLORS[region_name]
    tissue_rgb = TISSUE_COLORS[tissue_name] if tissue_name else None
    region_pixels = region.load()
    tissue_pixels = tissue.load()
    mask = Image.new("L", region.size)
    pixels = mask.load()
    # The OpenTME legend is embedded in the lower-left 138 x 64 pixels.
    for y in range(height):
        for x in range(width):
            if x < 138 and y >= height - 64:
                continue
            if region_pixels[x, y] == region_rgb and (tissue_rgb is None or tissue_pixels[x, y] == tissue_rgb):
                pixels[x, y] = 255
    if mask.getbbox() is None:
        raise ValueError("Selected feature has no resolvable pixels in the OpenTME thumbnail")
    return mask


def heat_overlay(wsi: Image.Image, region: Image.Image, mask: Image.Image) -> tuple[Image.Image, float, int]:
    radius = max(6.0, min(wsi.size) * 0.028)
    blurred = mask.filter(ImageFilter.GaussianBlur(radius=radius))
    values = sorted(blurred.tobytes())
    top = max(1, values[int(0.995 * (len(values) - 1))])
    background = wsi.load()
    local = blurred.load()
    region_pixels = region.load()
    width, height = wsi.size
    output = Image.new("RGB", wsi.size)
    painted = output.load()
    for y in range(height):
        for x in range(width):
            strength = min(1.0, local[x, y] / top)
            base = background[x, y]
            annotated_tissue = min(region_pixels[x, y]) < 245
            legend = x < 138 and y >= height - 64
            if strength < 0.035 or not annotated_tissue or legend:
                painted[x, y] = base
                continue
            # Continuous orange-to-yellow map; intensity is within-slide local
            # occupancy, not a model attention weight or a cell count.
            heat = (222 + round(33 * strength), round(46 + 168 * strength), round(38 + 32 * strength))
            alpha = 0.13 + 0.68 * math.sqrt(strength)
            painted[x, y] = tuple(round((1 - alpha) * b + alpha * h) for b, h in zip(base, heat))
    return output, radius, mask.histogram()[255]


def draw_figure(overlay: Image.Image, cohort: str, label: str, slide_id: str,
                display_feature: str, beta: float, rank: int, total_features: int,
                raw_value: float, contribution: float, positive_class: str) -> Image.Image:
    image_width = 1200
    image_height = round(overlay.height * image_width / overlay.width)
    scaled = overlay.resize((image_width, image_height), Image.Resampling.LANCZOS)
    margin, header, footer = 36, 112, 146
    figure = Image.new("RGB", (image_width + 2 * margin, image_height + header + footer), "white")
    figure.paste(scaled, (margin, header))
    draw = ImageDraw.Draw(figure)
    draw.text((margin, 20), f"{cohort.upper()}  /  {label}  /  {slide_id}", fill=(27, 36, 50), font=font(25, True))
    draw.text((margin, 60), f"TME feature: {display_feature}", fill=(73, 79, 91), font=font(23))
    base_y = header + image_height + 22
    bar_x, bar_y, bar_width, bar_height = margin, base_y, 380, 18
    for x in range(bar_width):
        v = x / (bar_width - 1)
        color = (222 + round(33 * v), round(46 + 168 * v), round(38 + 32 * v))
        draw.line((bar_x + x, bar_y, bar_x + x, bar_y + bar_height), fill=color)
    draw.text((margin, base_y + 27), "Low local occupancy", fill=(72, 80, 91), font=font(17))
    draw.text((margin + 248, base_y + 27), "High", fill=(72, 80, 91), font=font(17))
    draw.text((margin + 445, base_y - 4),
              f"TME-LR coefficient: {beta:+.4f} / 1 SD  (rank {rank}/{total_features})",
              fill=(40, 48, 60), font=font(19))
    draw.text((margin + 445, base_y + 27),
              f"Measured area: {raw_value:.4f}  |  {positive_class} logit term: {contribution:+.4f}",
              fill=(72, 80, 91), font=font(18))
    draw.text((margin, base_y + 77),
              "Color localizes the OpenTME region. LR values are slide-level; color is normalized within this slide.",
              fill=(72, 80, 91), font=font(17))
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentme-root", required=True, type=Path,
                        help="Registered OpenTME release directory containing data/<cancer>/thumbnails")
    parser.add_argument("--models-root", required=True, type=Path,
                        help="Saved TME-LR tme_only directory containing */model.json")
    parser.add_argument("--feature-table", action="append", default=[], metavar="COHORT=CSV",
                        help="Override the saved panel path for a cohort; repeat for each relocated panel")
    parser.add_argument("--manifest", type=Path,
                        default=Path("docs/qualitative-heatmaps/source_wsis.tsv"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    table_overrides = {}
    for item in args.feature_table:
        cohort_key, separator, path = item.partition("=")
        if not separator or cohort_key not in COHORTS or not path or cohort_key in table_overrides:
            parser.error("--feature-table must be a unique cohort=CSV pair")
        table_overrides[cohort_key] = Path(path)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        slides = list(csv.DictReader(handle, delimiter="\t"))
    if not slides:
        raise ValueError("Manifest has no slides")
    grouped: dict[str, list[dict[str, str]]] = {}
    for slide in slides:
        grouped.setdefault(slide["cohort"], []).append(slide)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for cohort, cohort_slides in grouped.items():
        model = model_for_cohort(args.models_root, cohort)
        feature, display, region_name, tissue_name = FEATURES[cohort]
        idx = model["feature_names"].index(feature)
        beta = model["coef"][idx]
        scale = model["scaler"]["scale"][idx]
        mean = model["scaler"]["mean"][idx]
        rank = sorted(range(len(model["coef"])), key=lambda j: abs(model["coef"][j]), reverse=True).index(idx) + 1
        task = model["task"]
        test = {row["slide_id"]: row for row in task["memberships"]["test"]}
        positive_class = next(name for name, number in task["label_dict"].items() if number == 1)
        feature_file = table_overrides.get(cohort, Path(task["tme_csv"]))
        if not feature_file.exists():
            raise FileNotFoundError(f"Provide --feature-table {cohort}=CSV for this relocated panel")
        measurements = rows_by_slide(feature_file, {row["slide_id"] for row in cohort_slides})
        for slide in cohort_slides:
            slide_id = slide["slide_id"]
            if slide_id not in test:
                raise ValueError(f"{slide_id} is not in the saved fold-0 test set")
            label = next(name for name, number in task["label_dict"].items() if number == test[slide_id]["label"])
            raw = float(measurements[slide_id][feature])
            if not math.isfinite(raw) or raw <= 0:
                raise ValueError(f"Missing or nonpositive real feature value: {slide_id} {feature}")
            source_table = args.opentme_root / "data" / COHORTS[cohort] / (
                f"whole_tumor_region_tissue_features_{COHORTS[cohort]}_RUO.csv")
            with source_table.open(newline="", encoding="utf-8") as handle:
                source_row = next((row for row in csv.DictReader(handle)
                                   if row["TCGA_FILE_NAME"] == Path(slide["filename"]).stem), None)
            if source_row is None or not math.isclose(raw, float(source_row[feature]), abs_tol=1e-12):
                raise ValueError(f"Panel and original OpenTME measurement disagree: {slide_id}")
            z = (raw - mean) / scale
            contribution = beta * z
            wsi, region, tissue = source_images(args.opentme_root, cohort, slide["filename"])
            mask = compartment_mask(region, tissue, region_name, tissue_name)
            overlay, radius, pixels = heat_overlay(wsi, region, mask)
            figure = draw_figure(overlay, cohort, label, slide_id, display, beta, rank,
                                 len(model["feature_names"]), raw, contribution, positive_class)
            name = f"{cohort}_{label.lower()}_tme_feature.png"
            figure.save(args.output_dir / name, optimize=True)
            row = {"cohort": cohort, "slide_id": slide_id, "label": label, "figure": name,
                   "feature": feature, "model_rank": rank, "model_feature_count": len(model["feature_names"]),
                   "model_id": task["id"], "thumbnail_sha256": thumbnail_hashes(
                       args.opentme_root, cohort, slide["filename"]),
                   "beta": beta, "measured_value": raw, "standardized_value": z,
                   "positive_class": positive_class, "logit_contribution": contribution,
                   "mask_pixels": pixels, "blur_radius_pixels": radius}
            report.append(row)
            print(f"{name}: {feature}, {pixels} mask pixels, beta={beta:+.4f}, beta*z={contribution:+.4f}")
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
