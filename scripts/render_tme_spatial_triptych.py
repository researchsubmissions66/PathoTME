#!/usr/bin/env python3
"""Render measured/reference/interaction region scores on a real WSI thumbnail."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial", required=True, type=Path,
                        help="spatial_scores.json from generate_tme_attribution.py")
    parser.add_argument("--class-name", required=True,
                        help="Exact saved class name for the displayed probability")
    parser.add_argument("--output", required=True, type=Path, help="New PNG path")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists")
    record = json.loads(args.spatial.read_text())
    if args.class_name not in record["classes"] or len(record["maps"]) != 1:
        raise ValueError("Select a saved class from one spatial input scale")
    cls = record["classes"].index(args.class_name)
    item = record["maps"][0]
    thumb = record["thumbnail"]
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(args.spatial.parent / "thumbnail.png").convert("RGB")
    if image.size != (thumb["width"], thumb["height"]):
        raise ValueError("Thumbnail dimensions disagree with saved provenance")
    if len(item["coordinates"]) != len(item["group_ids"]):
        raise ValueError("Patch coordinates and spatial group IDs disagree")
    scores = {
        key: [float(group[key][cls]) for group in item["groups"]]
        for key in ("observed_scores_pp", "reference_scores_pp", "difference_scores_pp")
    }
    # Measured and reference panels share a color scale so their colors are
    # directly comparable. The difference has its own labeled symmetric scale.
    visual_limit = max([abs(value) for key in ("observed_scores_pp", "reference_scores_pp")
                        for value in scores[key]], default=0.0) or 1.0
    interaction_limit = max(map(abs, scores["difference_scores_pp"]), default=0.0) or 1.0
    modes = (
        ("observed_scores_pp", "Measured TME", visual_limit),
        ("reference_scores_pp", "Feature at training mean", visual_limit),
        ("difference_scores_pp", "Measured − reference", interaction_limit),
    )
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        small = font
    header, footer = 84, 36
    result = Image.new("RGB", (3 * image.width, image.height + header + footer), "white")
    sx = image.width / thumb["level0_width"]
    sy = image.height / thumb["level0_height"]
    span = item["patch_size_level0"]
    for panel_index, (key, title, limit) in enumerate(modes):
        layer = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(layer)
        for (x, y), group_id in zip(item["coordinates"], item["group_ids"]):
            if not (0 <= group_id < len(scores[key])):
                raise ValueError("Spatial group index is out of range")
            value = scores[key][group_id]
            if not value:
                continue
            if x < 0 or y < 0 or x >= thumb["level0_width"] or y >= thumb["level0_height"]:
                raise ValueError("Patch coordinate is outside WSI level-0 geometry")
            color = (203, 47, 48) if value > 0 else (38, 93, 177)
            alpha = round(190 * min(1, abs(value) / limit))
            left, top = round(x * sx), round(y * sy)
            right = max(left, round((x + span) * sx) - 1)
            bottom = max(top, round((y + span) * sy) - 1)
            draw.rectangle((left, top, right, bottom), fill=(*color, alpha))
        overlaid = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
        x_offset = panel_index * image.width
        result.paste(overlaid, (x_offset, header))
        label = ImageDraw.Draw(result)
        label.text((x_offset + 10, 8), title, fill="#222222", font=font)
        label.text((x_offset + 10, 31),
                   f"{args.class_name} probability effect | red + / blue − | ±{limit:.3g} pp",
                   fill="#333333", font=small)
        label.text((x_offset + 10, 53), record["reference_name"],
                   fill="#333333", font=small)
    bottom_draw = ImageDraw.Draw(result)
    bottom_draw.text((10, image.height + header + 8),
                     f"{len(item['groups'])} joint spatial groups; scores repeat on member patches. "
                     "Unmeasured tissue is uncolored. Model sensitivity, not patch-level phenotype labels.",
                     fill="#333333", font=small)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(json.dumps({"output": str(args.output), "class_name": args.class_name,
                      "groups": len(item["groups"]), "visual_scale_pp": visual_limit,
                      "interaction_scale_pp": interaction_limit}))


if __name__ == "__main__":
    main()
