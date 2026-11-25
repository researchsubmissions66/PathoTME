#!/usr/bin/env python3
"""Place scored real patches on their original whole-slide thumbnail."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path,
                        help="analysis.json from export_tme_patch_examples.py")
    parser.add_argument("--wsi", required=True, type=Path,
                        help="The original WSI with the same exact slide ID and level-0 geometry")
    parser.add_argument("--output", required=True, type=Path, help="New PNG file")
    parser.add_argument("--top", type=int, default=3, help="Number of highest interactions")
    parser.add_argument("--bottom", type=int, default=3, help="Number of lowest interactions")
    parser.add_argument("--max-size", type=int, default=1600)
    args = parser.parse_args(argv)
    if min(args.top, args.bottom, args.max_size) < 1:
        parser.error("--top, --bottom and --max-size must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a new PNG path")

    analysis = json.loads(args.analysis.read_text())
    with (args.analysis.parent / "patch_scores.csv").open() as handle:
        ranked = list(csv.DictReader(handle))
    if not ranked or len(ranked) != analysis["screen"]["union_count"]:
        raise ValueError("Ranked rows disagree with the saved candidate screen")
    if analysis["screen"]["complete_patch_ranking"]:
        scope = "all consumed patches"
    else:
        scope = (f"{len(ranked)} attention-screened of "
                 f"{analysis['screen']['total_consumed_patches']} consumed patches")
    wsi = args.wsi.expanduser().resolve()
    if analysis["slide_id"] not in wsi.name:
        raise ValueError("WSI filename does not contain the analyzed slide ID")

    import openslide
    from PIL import Image, ImageDraw, ImageFont

    with openslide.OpenSlide(str(wsi)) as slide:
        if list(slide.dimensions) != analysis["level0_size"]:
            raise ValueError("WSI dimensions do not match the scored coordinate frame")
        image = slide.get_thumbnail((args.max_size, args.max_size)).convert("RGB")
    header = 86
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small = font
    draw.text((12, 8), f"{analysis['slide_id']}  |  {analysis['class_name']}  |  "
              f"{analysis['feature_name']}", fill="#222222", font=font)
    draw.text((12, 35), f"Patch–TME interaction (pp); {scope}",
              fill="#444444", font=small)
    draw.text((12, 57), "Red: highest    Blue: lowest    Boxes mark actual level-0 H&E crops",
              fill="#444444", font=small)
    sx, sy = image.width / analysis["level0_size"][0], image.height / analysis["level0_size"][1]
    span = analysis["patch_size_level0"]
    selected = [("H", ranked[:args.top], "#cb3030"),
                ("L", list(reversed(ranked[-args.bottom:])), "#2864b8")]
    selected_indices = set()
    for prefix, group, color in selected:
        for position, row in enumerate(group, 1):
            index = int(row["consumed_patch_index"])
            if index in selected_indices:
                continue
            selected_indices.add(index)
            x, y = int(row["x_level0"]), int(row["y_level0"])
            left, top = round(x * sx), round(y * sy) + header
            right, bottom = round((x + span) * sx), round((y + span) * sy) + header
            draw.rectangle((left, top, right, bottom), outline=color, width=3)
            label = f"{prefix}{position} {float(row['interaction_pp']):+.2f}"
            box = draw.textbbox((0, 0), label, font=small)
            label_width = box[2] - box[0] + 6
            label_height = box[3] - box[1] + 6
            label_left = min(left, canvas.width - label_width)
            label_top = max(header, top - label_height)
            draw.rectangle((label_left, label_top, label_left + label_width,
                            label_top + label_height), fill=color)
            draw.text((label_left + 3, label_top + 2), label, fill="white", font=small)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(json.dumps({"output": str(args.output), "marked_patches": len(selected_indices),
                      "screened_patches": len(ranked), "total_consumed_patches":
                      analysis["screen"]["total_consumed_patches"]}))


if __name__ == "__main__":
    main()
