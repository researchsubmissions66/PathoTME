#!/usr/bin/env python3
"""Render feature-localization heatmaps from real OpenTME slide thumbnails.

The spatial color is a smoothed thumbnail segmentation mask. The TME-LR model
supplies slide-level coefficients for four visible area features; it does not
supply pixel scores.
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

# Four WTR area features with corresponding OpenTME thumbnail annotations in
# every cohort. Region and tissue colors are the thumbnail legend swatches.
FEATURES = (
    ("RELATIVE_AREA_CARCINOMA_IN_TUMOR_CORE", "Carcinoma in tumor core", "core", "carcinoma"),
    ("RELATIVE_AREA_INNER_INVASIVE_MARGIN", "Inner invasive margin", "inner", None),
    ("RELATIVE_AREA_STROMA_IN_INNER_INVASIVE_MARGIN", "Stroma in inner margin", "inner", "stroma"),
    ("RELATIVE_AREA_OUTER_INVASIVE_MARGIN", "Outer invasive margin", "outer", None),
)
REGION_COLORS = {
    "core": (5, 255, 188),
    "inner": (0, 151, 111),
    "outer": (214, 0, 110),
}
TISSUE_COLORS = {"carcinoma": (153, 0, 0), "stroma": (255, 204, 102)}


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
    histogram = blurred.histogram()
    target = math.ceil(0.995 * wsi.width * wsi.height)
    cumulative = 0
    top = 1
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            top = max(1, value)
            break
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


def draw_atlas(cohort: str, slide_rows: list[dict], positive_class: str) -> Image.Image:
    tile_width, margin, gap = 440, 34, 20
    width = 2 * margin + len(FEATURES) * tile_width + (len(FEATURES) - 1) * gap
    image_heights = [round(row["wsi_size"][1] * tile_width / row["wsi_size"][0])
                     for row in slide_rows]
    height = 145 + sum(52 + image_height + 68 for image_height in image_heights) + 83
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    draw.text((margin, 17), f"{cohort.upper()}  /  Four measured TME features  /  Held-out fold 0",
              fill=(27, 36, 50), font=font(27, True))
    draw.text((margin, 57), "Each column localizes one OpenTME area feature on the same WSI.",
              fill=(73, 79, 91), font=font(20))
    for column, (_, display, _, _) in enumerate(FEATURES):
        x = margin + column * (tile_width + gap)
        draw.text((x, 109), display, fill=(39, 49, 62), font=font(19, True))
    y = 145
    for row, image_height in zip(slide_rows, image_heights):
        draw.text((margin, y + 6), f"{row['label']}  /  {row['slide_id']}",
                  fill=(27, 36, 50), font=font(22, True))
        image_y = y + 52
        for column, (feature, _, _, _) in enumerate(FEATURES):
            x = margin + column * (tile_width + gap)
            entry = row["features"][feature]
            scaled = entry["overlay"].resize((tile_width, image_height), Image.Resampling.LANCZOS)
            figure.paste(scaled, (x, image_y))
            draw.rectangle((x, image_y, x + tile_width - 1, image_y + image_height - 1),
                           outline=(219, 221, 225), width=1)
            draw.text((x, image_y + image_height + 7),
                      f"Area frac {entry['raw']:.4f}  |  LR beta {entry['beta']:+.4f}  (rank {entry['rank']})",
                      fill=(47, 55, 66), font=font(17))
            draw.text((x, image_y + image_height + 33),
                      f"{positive_class} logit term {entry['contribution']:+.4f}",
                      fill=(79, 86, 96), font=font(16))
        y = image_y + image_height + 68
    bar_width, bar_height = 340, 17
    for offset in range(bar_width):
        strength = offset / (bar_width - 1)
        color = (222 + round(33 * strength), round(46 + 168 * strength),
                 round(38 + 32 * strength))
        draw.line((margin + offset, y + 2, margin + offset, y + 2 + bar_height), fill=color)
    draw.text((margin, y + 26), "Low local occupancy", fill=(72, 80, 91), font=font(16))
    draw.text((margin + 257, y + 26), "High", fill=(72, 80, 91), font=font(16))
    draw.text((margin + 400, y + 1),
              "Color is normalized within each panel. LR beta and logit term are slide-level, not pixel attention.",
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
        ranked = sorted(range(len(model["coef"])), key=lambda j: abs(model["coef"][j]), reverse=True)
        feature_states = {}
        for feature, _, _, _ in FEATURES:
            idx = model["feature_names"].index(feature)
            feature_states[feature] = {
                "beta": model["coef"][idx],
                "scale": model["scaler"]["scale"][idx],
                "mean": model["scaler"]["mean"][idx],
                "rank": ranked.index(idx) + 1,
            }
        task = model["task"]
        test = {row["slide_id"]: row for row in task["memberships"]["test"]}
        positive_class = next(name for name, number in task["label_dict"].items() if number == 1)
        feature_file = table_overrides.get(cohort, Path(task["tme_csv"]))
        if not feature_file.exists():
            raise FileNotFoundError(f"Provide --feature-table {cohort}=CSV for this relocated panel")
        measurements = rows_by_slide(feature_file, {row["slide_id"] for row in cohort_slides})
        atlas_rows = []
        figure_name = f"{cohort}_tme_feature_atlas.png"
        for slide in cohort_slides:
            slide_id = slide["slide_id"]
            if slide_id not in test:
                raise ValueError(f"{slide_id} is not in the saved fold-0 test set")
            label = next(name for name, number in task["label_dict"].items() if number == test[slide_id]["label"])
            source_table = args.opentme_root / "data" / COHORTS[cohort] / (
                f"whole_tumor_region_tissue_features_{COHORTS[cohort]}_RUO.csv")
            with source_table.open(newline="", encoding="utf-8") as handle:
                source_row = next((row for row in csv.DictReader(handle)
                                   if row["TCGA_FILE_NAME"] == Path(slide["filename"]).stem), None)
            if source_row is None:
                raise ValueError(f"Slide is absent from the original OpenTME table: {slide_id}")
            wsi, region, tissue = source_images(args.opentme_root, cohort, slide["filename"])
            atlas_row = {"label": label, "slide_id": slide_id, "wsi_size": wsi.size, "features": {}}
            hashes = thumbnail_hashes(args.opentme_root, cohort, slide["filename"])
            for feature, _, region_name, tissue_name in FEATURES:
                state = feature_states[feature]
                raw = float(measurements[slide_id][feature])
                if not math.isfinite(raw) or raw <= 0 or not math.isclose(
                        raw, float(source_row[feature]), abs_tol=1e-12):
                    raise ValueError(f"Panel and original OpenTME measurement disagree: {slide_id} {feature}")
                z = (raw - state["mean"]) / state["scale"]
                contribution = state["beta"] * z
                mask = compartment_mask(region, tissue, region_name, tissue_name)
                overlay, radius, pixels = heat_overlay(wsi, region, mask)
                atlas_row["features"][feature] = {
                    "overlay": overlay, "raw": raw, "beta": state["beta"],
                    "rank": state["rank"], "contribution": contribution,
                }
                report.append({
                    "cohort": cohort, "slide_id": slide_id, "label": label,
                    "figure": figure_name, "feature": feature,
                    "model_rank": state["rank"], "model_feature_count": len(model["feature_names"]),
                    "model_id": task["id"], "thumbnail_sha256": hashes,
                    "beta": state["beta"], "measured_value": raw, "standardized_value": z,
                    "positive_class": positive_class, "logit_contribution": contribution,
                    "mask_pixels": pixels, "blur_radius_pixels": radius,
                })
                print(f"{cohort} {slide_id}: {feature}, {pixels} mask pixels, beta={state['beta']:+.4f}")
            atlas_rows.append(atlas_row)
        draw_atlas(cohort, atlas_rows, positive_class).save(args.output_dir / figure_name, optimize=True)
    (args.output_dir / "figure_metadata.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
