#!/usr/bin/env python3
"""Overlay scored patch–TME interactions on an original WSI thumbnail.

Only attention-screened patches have individual scores. Unscored locations
remain uncolored; this is a sparse evidence map, not a full-slide estimate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--wsi", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-size", type=int, default=1600)
    parser.add_argument("--limit-pp", type=float, default=1.0,
                        help="Symmetric display scale; larger magnitudes are clipped (default: 1 pp)")
    args = parser.parse_args(argv)
    if args.max_size < 1 or args.limit_pp <= 0:
        parser.error("--max-size and --limit-pp must be positive")
    if args.output.exists():
        parser.error("Output already exists")
    analysis = json.loads(args.analysis.read_text())
    with (args.analysis.parent / "patch_scores.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != analysis["screen"]["union_count"]:
        raise ValueError("Saved screen count and patch scores disagree")
    wsi = args.wsi.expanduser().resolve()
    if analysis["slide_id"] not in wsi.name:
        raise ValueError("WSI filename does not match the analyzed slide ID")

    import openslide
    from PIL import Image, ImageDraw, ImageFont
    with openslide.OpenSlide(str(wsi)) as slide:
        if list(slide.dimensions) != analysis["level0_size"]:
            raise ValueError("WSI dimensions do not match the scored level-0 frame")
        base = slide.get_thumbnail((args.max_size, args.max_size)).convert("RGB")
    sx = base.width / analysis["level0_size"][0]
    sy = base.height / analysis["level0_size"][1]
    span = analysis["patch_size_level0"]
    layer = Image.new("RGBA", base.size)
    fill = ImageDraw.Draw(layer)
    outlines = []
    clipped = 0
    for row in rows:
        x, y = int(row["x_level0"]), int(row["y_level0"])
        if x < 0 or y < 0 or x + span > analysis["level0_size"][0] or y + span > analysis["level0_size"][1]:
            raise ValueError("Scored patch falls outside WSI")
        score = float(row["interaction_pp"])
        strength = min(1, abs(score) / args.limit_pp)
        clipped += abs(score) > args.limit_pp
        left, top = round(x * sx), round(y * sy)
        right = max(left, round((x + span) * sx) - 1)
        bottom = max(top, round((y + span) * sy) - 1)
        if score:
            color = (203, 47, 48) if score > 0 else (38, 93, 177)
            fill.rectangle((left, top, right, bottom), fill=(*color, round(210 * strength)))
        outlines.append((left, top, right, bottom))
    overlaid = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    border = ImageDraw.Draw(overlaid)
    for rectangle in outlines:
        border.rectangle(rectangle, outline="#6b6b6b", width=1)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 16)
        note_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        title_font = note_font = ImageFont.load_default()
    header = 92
    figure = Image.new("RGB", (overlaid.width, overlaid.height + header), "white")
    figure.paste(overlaid, (0, header))
    draw = ImageDraw.Draw(figure)
    draw.text((10, 7), f"{analysis['class_name']} patch–TME interaction | {analysis['feature_name']}",
              fill="#222222", font=title_font)
    draw.text((10, 34),
              f"Red + / blue − | ±{args.limit_pp:g} pp display scale | {clipped} scores clipped",
              fill="#333333", font=note_font)
    draw.text((10, 57),
              f"Only {len(rows)} attention-screened of {analysis['screen']['total_consumed_patches']} patches scored; gray borders mark screened patches.",
              fill="#333333", font=note_font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output)
    print(json.dumps({"output": str(args.output), "screened_patches": len(rows),
                      "total_patches": analysis["screen"]["total_consumed_patches"],
                      "display_limit_pp": args.limit_pp, "clipped_scores": clipped}))


if __name__ == "__main__":
    main()
