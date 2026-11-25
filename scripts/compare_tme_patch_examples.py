#!/usr/bin/env python3
"""Join two scored TME features on the same real slide and patch coordinates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path):
    path = Path(path)
    analysis = json.loads(path.read_text())
    with (path.parent / "patch_scores.csv").open() as handle:
        rows = {int(row["source_patch_index"]): row for row in csv.DictReader(handle)}
    if len(rows) != analysis["screen"]["union_count"]:
        raise ValueError(f"Patch ranking and candidate count disagree: {path}")
    return analysis, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path, help="First analysis.json")
    parser.add_argument("--second", required=True, type=Path, help="Second analysis.json")
    parser.add_argument("--output", required=True, type=Path, help="New result directory")
    args = parser.parse_args(argv)
    first, a = load(args.first)
    second, b = load(args.second)
    if first["feature_name"] == second["feature_name"]:
        raise ValueError("Compare two different TME features")
    identity = ("slide_id", "class_index", "fold", "arm", "method", "encoder",
                "feature_hdf5_sha256", "visual_reference_sha256", "checkpoint_sha256",
                "level0_size", "patch_size_level0")
    if any(first[key] != second[key] for key in identity):
        raise ValueError("Analyses do not use the same slide, class, bag and checkpoint")

    rows = []
    for index in sorted(a.keys() | b.keys()):
        left, right = a.get(index), b.get(index)
        known = left or right
        if left and right and any(left[key] != right[key] for key in
                              ("consumed_patch_index", "x_level0", "y_level0")):
            raise ValueError(f"Patch {index} has inconsistent coordinates")
        rows.append({
            "source_patch_index": index,
            "consumed_patch_index": int(known["consumed_patch_index"]),
            "x_level0": int(known["x_level0"]),
            "y_level0": int(known["y_level0"]),
            "first_scored": left is not None,
            "first_interaction_pp": float(left["interaction_pp"]) if left else "",
            "first_rank": int(left["interaction_rank"]) if left else "",
            "second_scored": right is not None,
            "second_interaction_pp": float(right["interaction_pp"]) if right else "",
            "second_rank": int(right["interaction_rank"]) if right else "",
        })
    rows.sort(key=lambda row: (-max(abs(row["first_interaction_pp"]) if row["first_scored"] else 0,
                                     abs(row["second_interaction_pp"]) if row["second_scored"] else 0),
                               row["source_patch_index"]))
    summary = {
        "slide_id": first["slide_id"], "class_name": first["class_name"],
        "fold": first["fold"], "first_feature": first["feature_name"],
        "second_feature": second["feature_name"],
        "first_screened": len(a), "second_screened": len(b),
        "shared_scored_patches": len(a.keys() & b.keys()),
        "union_scored_patches": len(rows),
        "total_consumed_patches": first["screen"]["total_consumed_patches"],
        "missing_score_policy": "blank means not screened for that feature; it is not zero",
        "interpretation": "Matched reference-dependent model sensitivities; the TME features are slide-level measurements, not patch labels.",
    }
    output = args.output.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents or output.exists():
        parser.error("Choose a new result directory outside the source repository")
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "shared_scored_patches":
                      summary["shared_scored_patches"], "union_scored_patches": len(rows)}))


if __name__ == "__main__":
    main()
