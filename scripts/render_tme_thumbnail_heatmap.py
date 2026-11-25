#!/usr/bin/env python3
"""Render one full WSI thumbnail with its measured-minus-reference TME interaction."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def _color(value: float, limit: float) -> tuple[int, int, int]:
    """Diverging blue/near-white/red scale, clipped at the labeled limit."""
    fraction = max(-1.0, min(1.0, value / limit))
    center = (241, 242, 239)
    endpoint = (38, 89, 183) if fraction < 0 else (205, 45, 51)
    strength = abs(fraction)
    return tuple(round(a + strength * (b - a)) for a, b in zip(center, endpoint))


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", required=True, type=Path,
                        help="spatial_scores.json from generate_tme_attribution.py")
    parser.add_argument("--class-name", required=True,
                        help="Exact saved class name to display")
    parser.add_argument("--output", required=True, type=Path,
                        help="New single-thumbnail PNG")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists")

    from PIL import Image, ImageDraw
    spatial = json.loads(args.spatial.read_text())
    attribution = json.loads((args.spatial.parent.parent / "attribution.json").read_text())
    if len(spatial["maps"]) != 1 or len(attribution["slides"]) != 1:
        raise ValueError("Expected one spatial input scale and one attributed slide")
    if spatial["classes"] != attribution["classes"] or args.class_name not in spatial["classes"]:
        raise ValueError("Displayed class does not match saved attribution")
    cls = spatial["classes"].index(args.class_name)
    slide = attribution["slides"][0]
    if cls != slide["label"] or not slide["saved_probability_match"] or not slide["restoration_checked"]:
        raise ValueError("Use the verified held-out class and restored checkpoint")
    item, thumb = spatial["maps"][0], spatial["thumbnail"]
    provenance = thumb["provenance"]
    if provenance["source"] == "openslide" and slide["slide_id"] not in provenance["wsi_basename"]:
        raise ValueError("Original WSI does not match the attributed slide")
    coordinates, group_ids = item["coordinates"], item["group_ids"]
    if not coordinates or len(coordinates) != len(group_ids):
        raise ValueError("Patch coordinates and spatial groups disagree")
    scores = [float(group["difference_scores_pp"][cls]) for group in item["groups"]]
    if not scores or not all(math.isfinite(value) for value in scores):
        raise ValueError("Finite saved interaction scores are required")
    if any(type(group_id) is not int or not 0 <= group_id < len(scores) for group_id in group_ids):
        raise ValueError("Invalid patch-to-group assignment")
    counts = Counter(group_ids)
    if any(counts[group_id] != group["patch_count"]
           for group_id, group in enumerate(item["groups"])):
        raise ValueError("Saved joint-group counts disagree with the plotted patches")
    values = [scores[group_id] for group_id in group_ids]
    absolute = sorted(abs(value) for value in values)
    percentile_95 = absolute[math.ceil(0.95 * len(absolute)) - 1]
    limit = percentile_95 or absolute[-1] or 1.0

    source = Image.open(args.spatial.parent / "thumbnail.png").convert("RGB")
    if source.size != (thumb["width"], thumb["height"]):
        raise ValueError("Thumbnail differs from saved level-0 registration")
    sx = source.width / thumb["level0_width"]
    sy = source.height / thumb["level0_height"]
    span = item["patch_size_level0"]
    if span <= 0:
        raise ValueError("A positive level-0 patch span is required")
    overlay = Image.new("RGBA", source.size)
    draw = ImageDraw.Draw(overlay)
    for (x, y), value in zip(coordinates, values):
        if x < 0 or y < 0 or x >= thumb["level0_width"] or y >= thumb["level0_height"]:
            raise ValueError("A feature-bag coordinate is outside the original WSI")
        left, top = round(x * sx), round(y * sy)
        right = max(left, round((x + span) * sx) - 1)
        bottom = max(top, round((y + span) * sy) - 1)
        draw.rectangle((left, top, right, bottom), fill=(*_color(value, limit), 138))
    merged = Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")

    header, footer = 100, 155
    figure = Image.new("RGB", (source.width, source.height + header + footer), "white")
    figure.paste(merged, (0, header))
    ink = ImageDraw.Draw(figure)
    cohort = attribution["cohort"].upper()
    ink.text((16, 10), f"{cohort} · {args.class_name} · whole-slide thumbnail", fill="#192334", font=_font(23))
    ink.text((16, 44), "TME-conditioned visual effect: measured − training-mean reference",
             fill="#27364a", font=_font(17))
    ink.text((16, 70), f"{spatial['reference_name']}  |  {slide['slide_id']}",
             fill="#465363", font=_font(14))

    bar_left, bar_right = 140, source.width - 140
    bar_top = header + source.height + 46
    if bar_right <= bar_left:
        raise ValueError("Thumbnail is too narrow for a legible color scale")
    ink.text((16, header + source.height + 10),
             "Blue: lower region effect with measured TME     Red: higher region effect",
             fill="#27364a", font=_font(15))
    for column in range(bar_left, bar_right):
        value = limit * (2 * (column - bar_left) / max(1, bar_right - bar_left - 1) - 1)
        ink.line((column, bar_top, column, bar_top + 18), fill=_color(value, limit))
    ink.rectangle((bar_left, bar_top, bar_right, bar_top + 18), outline="#5b6570", width=1)
    ink.text((bar_left - 6, bar_top + 23), f"−{limit:.3g} pp", anchor="ra", fill="#27364a", font=_font(14))
    ink.text(((bar_left + bar_right) // 2, bar_top + 23), "0", anchor="ma", fill="#27364a", font=_font(14))
    ink.text((bar_right + 6, bar_top + 23), f"+{limit:.3g} pp", anchor="la", fill="#27364a", font=_font(14))
    ink.text((16, header + source.height + 99),
             f"{len(coordinates):,} real patch tiles; {len(scores)} joint groups; no interpolation. "
             f"Color clips at the 95th percentile; max |effect| = {absolute[-1]:.3g} pp.",
             fill="#465363", font=_font(13))
    ink.text((16, header + source.height + 121),
             "Uncolored tissue had no feature patch. Colors show model sensitivity, not TME abundance or cell locations.",
             fill="#465363", font=_font(13))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.output)
    print(json.dumps({"output": str(args.output), "cohort": cohort, "slide_id": slide["slide_id"],
                      "class_name": args.class_name, "patches": len(coordinates), "groups": len(scores),
                      "color_limit_pp": limit, "maximum_absolute_score_pp": absolute[-1]}))


if __name__ == "__main__":
    main()
