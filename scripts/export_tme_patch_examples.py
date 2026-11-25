#!/usr/bin/env python3
"""Rank real held-out MUSE patches by interaction with one measured TME feature.

Use the exact completed BRCA/CLIP-RN50 run and its coordinate-bearing HDF5
feature bag. A matching WSI is optional for exporting unaltered tissue crops.
No training, fake patches, or new TME values are created.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PGVL = Path(os.environ.get("PGVL_ROOT", ROOT.parent / "PGVL-Gym")).resolve()
sys.path[:0] = [str(ROOT), str(PGVL)]

from pathotme.attribution import panel_layout
from pathotme.attribution_io import digest, load_bound_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion", required=True, type=Path,
                        help="Completed BRCA MUSE CLIP-RN50 fold_complete.json")
    parser.add_argument("--feature-name", required=True,
                        help="Exact BRCA morph64 feature name to replace with its training-fold mean")
    parser.add_argument("--slide-id", help="Exact held-out slide; default is the first lexicographic test slide")
    parser.add_argument("--class-name", help="IDC or ILC; default is the saved true class")
    parser.add_argument("--candidate-per-condition", type=int, default=32,
                        help="Top attended patches screened under each TME condition (default: 32)")
    parser.add_argument("--selection-note", default="explicit_feature_name",
                        help="Record the prior rule or rationale used to choose the feature")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path,
                        help="New private result directory outside the source repositories")
    parser.add_argument("--wsi", type=Path,
                        help="Matching original WSI; export exact level-0 patch crops and a montage")
    parser.add_argument("--crop-count", type=int, default=6,
                        help="Highest and lowest interaction crops to show in the montage")
    args = parser.parse_args(argv)

    if args.threads < 1 or args.candidate_per_condition < 1 or args.crop_count < 1:
        parser.error("--threads, --candidate-per-condition and --crop-count must be positive")
    output = args.output.expanduser().resolve()
    if any(output == root or root in output.parents for root in (ROOT, PGVL)):
        parser.error("Use a private output directory outside the source repositories")
    if output.exists():
        parser.error("Output already exists; choose a new directory")

    bound = load_bound_run(args.completion, "actual")
    cfg = bound["config"]
    plan = bound["completion"]["plan"]
    if (plan["cohort"], plan["method"], cfg["backbone"]) != ("brca", "muse", "clip-rn50"):
        parser.error("This exporter is bound to completed BRCA MUSE/CLIP-RN50 actual checkpoints")
    feature_names, _ = panel_layout("brca")
    if args.feature_name not in feature_names:
        parser.error("Unknown BRCA morph64 feature name")
    slide_id = args.slide_id or min(row["slide_id"] for row in bound["test_rows"])
    test_rows = {row["slide_id"]: row for row in bound["test_rows"]}
    if slide_id not in test_rows:
        parser.error("The selected slide is outside this saved held-out fold")
    row = test_rows[slide_id]
    classes = [name for name, index in sorted(cfg["label_dict"].items(), key=lambda item: item[1])]
    if [cfg["label_dict"][name] for name in classes] != list(range(len(classes))):
        raise ValueError("Class index order does not match the saved checkpoint")
    if args.class_name and args.class_name not in classes:
        parser.error("--class-name must be one of the saved classes: " + ", ".join(classes))
    true_index = int(cfg["label_dict"].get(row["label"], row["label"]))
    class_index = cfg["label_dict"][args.class_name] if args.class_name else true_index
    if args.wsi:
        wsi = args.wsi.expanduser().resolve()
        if not wsi.is_file() or slide_id not in wsi.name:
            parser.error("--wsi must be an existing original slide whose filename contains the exact slide ID")
    else:
        wsi = None

    print(json.dumps({"slide_id": slide_id, "fold": plan["fold"], "feature_name": args.feature_name,
                      "class_name": classes[class_index], "candidate_per_condition": args.candidate_per_condition,
                      "wsi_crops": bool(wsi), "training": False}), flush=True)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from common.configuration import load_dotenv
    load_dotenv(PGVL / ".env")
    import torch
    from common.interpretability import load_coordinate_bag
    from common.spatial_attribution import bind_spatial_inputs
    from pathotme.attribution_runtime import check_selected_features, load_model, one_batch
    from pathotme.qualitative_patches import (
        SCORE_VERSION, export_wsi_crops, score_patches, write_scores,
    )
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    bound = load_bound_run(args.completion, "actual", weights=True)
    feature_files = check_selected_features(bound, row)
    bridge, model, loader = load_model(bound, "actual", args.device)
    batch = one_batch(loader, slide_id)
    binding = bind_spatial_inputs(bridge, batch, row, slide_id)
    if len(binding) != 1:
        raise ValueError("Expected one coordinate-bound MUSE patch bag")
    binding = binding[0]
    geometry = load_coordinate_bag(binding.source)
    if wsi:
        import openslide
        with openslide.OpenSlide(str(wsi)) as slide:
            if tuple(slide.dimensions) != (geometry.level0_width, geometry.level0_height):
                raise ValueError("WSI dimensions disagree with feature HDF5 level-0 geometry")

    with torch.no_grad():
        detail = bridge.eval_step_with_details(batch, model)
        if detail["metadata"]["slide_id"] != slide_id:
            raise ValueError("Loaded batch slide identity mismatch")
        expected = detail["probabilities"].new_tensor([[
            float(bound["predictions"][slide_id][f"probability_{index}"])
            for index in range(len(classes))]])
        torch.testing.assert_close(detail["probabilities"], expected, atol=1e-4, rtol=1e-4)

    summary, rows = score_patches(
        bridge, model, batch, binding, feature_index=feature_names.index(args.feature_name),
        class_index=class_index, candidate_per_condition=args.candidate_per_condition,
        progress=lambda done, total: print(f"Scored {done}/{total} candidates", flush=True)
        if done % 8 == 0 or done == total else None,
    )
    torch.testing.assert_close(
        detail["probabilities"].double().cpu()[0, class_index],
        torch.tensor(summary["probability_observed"], dtype=torch.double), atol=1e-7, rtol=0)
    summary.update({
        "cohort": "brca", "method": "muse", "encoder": "clip-rn50", "arm": "actual",
        "fold": plan["fold"], "shots": cfg.get("shots", plan.get("shots")),
        "case_id": row["case_id"], "true_class": classes[true_index],
        "predicted_class": classes[int(bound["predictions"][slide_id]["prediction"])],
        "class_name": classes[class_index], "feature_name": args.feature_name,
        "feature_selection_note": args.selection_note,
        "slide_selection_rule": "explicit_slide_id" if args.slide_id else "first_lexicographic_held_out_slide",
        "saved_prediction_match": True, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_hdf5_basename": binding.source.name,
        "feature_hdf5_sha256": digest(binding.source),
        "feature_inventory_verified": len(feature_files) == 1,
        "coordinate_frame": "level0", "patch_size_level0": geometry.patch_size_level0,
        "level0_size": [geometry.level0_width, geometry.level0_height],
        "checkpoint_sha256": {Path(path).name: value for path, value in
                              bound["provenance"]["checkpoint_sha256"].items()},
        "completion_sha256": bound["provenance"]["completion_sha256"],
        "launch_identity": bound["provenance"]["launch_identity"],
        "source_sha256": {
            "scripts/export_tme_patch_examples.py": digest(Path(__file__)),
            "pathotme/qualitative_patches.py": digest(ROOT / "pathotme/qualitative_patches.py"),
        },
        "elapsed_seconds": time.monotonic() - started,
    })
    if summary["score_version"] != SCORE_VERSION:
        raise ValueError("Unexpected patch score definition")
    write_scores(output, summary, rows)
    if wsi:
        summary["wsi_crops"] = export_wsi_crops(
            wsi, output / "crops", rows, slide_id=slide_id,
            patch_span=geometry.patch_size_level0,
            level0_size=(geometry.level0_width, geometry.level0_height),
            crop_count=args.crop_count)
        (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", "output": str(output),
                      "screened_patches": len(rows), "total_patches": summary["screen"]["total_consumed_patches"],
                      "highest_interaction_pp": rows[0]["interaction_pp"],
                      "lowest_interaction_pp": rows[-1]["interaction_pp"],
                      "crop_images": bool(wsi)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
