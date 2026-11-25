"""Patch and TME-feature interaction for a saved MUSE adapter.

This is a finite-difference model sensitivity, not a patch-level TME
measurement or a causal attribution. Attention only screens candidates; the
reported scores come from complete model predictions after interventions.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


SCORE_VERSION = "muse_patch_tme_train_mean_interaction_pp_v1"


def score_patches(bridge, model, batch, binding, *, feature_index, class_index,
                  candidate_per_condition=32, progress=None):
    """Score an attention-screened union of real, coordinate-bound patches.

    One visual intervention replaces a consumed patch vector with the fixed
    mean of the original consumed bag. One TME intervention sets a single
    feature to zero after the saved training-fold standardizer. The interaction
    is the difference between the two visual effects, in percentage points.
    """
    import torch

    if bridge.name != "muse" or binding.batch_key != 0 or binding.scale != "patch":
        raise ValueError("This scorer requires the coordinate-bound MUSE patch bag")
    if any(module.training for module in model.modules()):
        raise ValueError("The saved model must be fully in evaluation mode")
    if not bool(model.standardizer.fitted.item()):
        raise ValueError("The saved training-fold standardizer must be fitted")
    width = int(model.standardizer.feature_count)
    if type(feature_index) is not int or not 0 <= feature_index < width:
        raise ValueError("Invalid TME feature index")
    n_classes = int(bridge.cfg["n_classes"])
    if type(class_index) is not int or not 0 <= class_index < n_classes:
        raise ValueError("Invalid target class")
    if type(candidate_per_condition) is not int or candidate_per_condition < 1:
        raise ValueError("candidate_per_condition must be positive")

    original = batch[0]
    if original.ndim not in (2, 3) or (original.ndim == 3 and original.shape[0] != 1):
        raise ValueError("Expected one MUSE feature bag")
    flat = original.reshape(-1, original.shape[-1])
    if (len(flat) != len(binding.coordinates)
            or len(flat) != len(binding.flat_indices)
            or len(flat) != len(binding.source_indices)
            or not torch.equal(binding.flat_indices.cpu(), torch.arange(len(flat)))):
        raise ValueError("Consumed patches do not align one-to-one with coordinates")
    visual_reference = flat.mean(0)
    if not torch.isfinite(visual_reference).all():
        raise ValueError("Nonfinite visual reference")
    reference_sha = hashlib.sha256(
        visual_reference.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def forward(value, *, neutral_tme=False, attention=False):
        seen = []

        def standardizer_hook(_module, inputs, output):
            if output.shape != (1, width) or not torch.isfinite(output).all():
                raise ValueError("Expected one finite standardized TME row")
            seen.append((float(output[0, feature_index]),
                         bool(torch.isnan(inputs[0][0, feature_index]))))
            if not neutral_tme:
                return output
            changed = output.clone()
            changed[0, feature_index] = 0
            return changed

        hook = model.standardizer.register_forward_hook(standardizer_hook)
        try:
            inputs, metadata = bridge.inputs(value, model)
            result = model(*inputs, return_details=attention)
            probability = result["probabilities"] if attention else result[0]
            probability = probability.detach().double().cpu()
            if (len(seen) != 1 or probability.shape != (1, n_classes)
                    or not torch.isfinite(probability).all()
                    or (probability < 0).any() or (probability > 1).any()
                    or not torch.allclose(probability.sum(1), torch.ones(1, dtype=torch.double),
                                          atol=1e-5, rtol=0)):
                raise ValueError("Expected one standardized-TME call and valid probabilities")
            weights = None
            if attention:
                weights = result["patch_attention"].detach().float().cpu()
                if weights.shape != (n_classes, len(flat)) or not torch.isfinite(weights).all():
                    raise ValueError("MUSE attention does not align with the consumed bag")
            return probability[0], weights, seen[0], metadata
        finally:
            hook.remove()

    def shortlist(weights):
        values = weights[class_index].tolist()
        return sorted((i for i, score in enumerate(values) if score > 0),
                      key=lambda i: (-values[i], i))[:candidate_per_condition]

    def remove_patch(index):
        changed = list(batch)
        replacement = original.clone()
        replacement.reshape(-1, original.shape[-1])[index] = visual_reference
        changed[0] = replacement
        return changed

    with torch.no_grad():
        actual, actual_attention, measured, actual_meta = forward(batch, attention=True)
        neutral, neutral_attention, neutral_measured, neutral_meta = forward(
            batch, neutral_tme=True, attention=True)
        if actual_meta["slide_id"] != neutral_meta["slide_id"] or measured != neutral_measured:
            raise ValueError("Slide identity or measured TME changed across conditions")
        if measured[1]:
            raise ValueError("The selected TME feature was imputed for this slide")
        actual_top = shortlist(actual_attention)
        neutral_top = shortlist(neutral_attention)
        if not actual_top and not neutral_top:
            raise ValueError("No MUSE-attended candidate patches")
        candidates = sorted(set(actual_top) | set(neutral_top))
        rows = []
        for done, index in enumerate(candidates, 1):
            changed = remove_patch(index)
            without_actual, _, _, _ = forward(changed)
            without_neutral, _, _, _ = forward(changed, neutral_tme=True)
            actual_effect = 100 * (actual - without_actual)
            neutral_effect = 100 * (neutral - without_neutral)
            xy = binding.coordinates[index].tolist()
            rows.append({
                "consumed_patch_index": index,
                "source_patch_index": int(binding.source_indices[index]),
                "x_level0": int(xy[0]), "y_level0": int(xy[1]),
                "attention_observed": float(actual_attention[class_index, index]),
                "attention_feature_reference": float(neutral_attention[class_index, index]),
                "screened_observed": index in actual_top,
                "screened_feature_reference": index in neutral_top,
                "visual_effect_observed_pp": float(actual_effect[class_index]),
                "visual_effect_feature_reference_pp": float(neutral_effect[class_index]),
                "interaction_pp": float(actual_effect[class_index] - neutral_effect[class_index]),
                "probability_without_patch_observed": float(without_actual[class_index]),
                "probability_without_patch_feature_reference": float(without_neutral[class_index]),
            })
            if progress is not None:
                progress(done, len(candidates))
        repeat_actual, _, _, _ = forward(batch)
        repeat_neutral, _, _, _ = forward(batch, neutral_tme=True)
        torch.testing.assert_close(repeat_actual, actual, atol=1e-7, rtol=0)
        torch.testing.assert_close(repeat_neutral, neutral, atol=1e-7, rtol=0)

    rows.sort(key=lambda row: (-row["interaction_pp"], row["consumed_patch_index"]))
    for rank, row in enumerate(rows, 1):
        row["interaction_rank"] = rank
    summary = {
        "score_version": SCORE_VERSION,
        "slide_id": actual_meta["slide_id"],
        "class_index": class_index,
        "feature_index": feature_index,
        "standardized_feature_value": measured[0],
        "feature_was_imputed": measured[1],
        "probability_observed": float(actual[class_index]),
        "probability_feature_reference": float(neutral[class_index]),
        "feature_effect_pp": float(100 * (actual[class_index] - neutral[class_index])),
        "screen": {
            "rule": "union_of_top_positive_muse_patch_attention_per_tme_condition",
            "candidate_per_condition": candidate_per_condition,
            "observed_count": len(actual_top),
            "feature_reference_count": len(neutral_top),
            "union_count": len(candidates),
            "total_consumed_patches": len(flat),
            "complete_patch_ranking": len(candidates) == len(flat),
        },
        "visual_reference": "fixed_mean_of_all_original_consumed_patch_vectors",
        "visual_reference_sha256": reference_sha,
        "tme_reference": "zero_after_saved_training_fold_standardization",
        "forward_calls": 4 + 2 * len(candidates),
        "baseline_restored": True,
        "score_definition": "100 * ([p_c(X,z)-p_c(X_without_i,z)] - [p_c(X,z_j=0)-p_c(X_without_i,z_j=0)])",
        "interpretation": "Reference-dependent model sensitivity for screened patches; not patch-local TME measurement, cell localization, an additive decomposition or a causal effect.",
    }
    return summary, rows


def write_scores(output, summary, rows):
    """Save the full screened ranking and audit metadata without local paths."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "patch_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_wsi_crops(wsi_path, output, rows, *, slide_id, patch_span, level0_size,
                     crop_count=6):
    """Export unaltered level-0 crops for highest and lowest interactions."""
    import openslide
    from PIL import Image, ImageDraw

    wsi_path = Path(wsi_path).expanduser().resolve()
    if slide_id not in wsi_path.name:
        raise ValueError("WSI filename must contain the exact analyzed slide ID")
    if type(crop_count) is not int or crop_count < 1:
        raise ValueError("crop_count must be positive")
    if type(patch_span) is not int or patch_span < 1:
        raise ValueError("Invalid level-0 patch span")
    chosen = rows[:crop_count] + list(reversed(rows[-crop_count:]))
    unique = list(dict.fromkeys(row["consumed_patch_index"] for row in chosen))
    by_index = {row["consumed_patch_index"]: row for row in rows}
    crops = []
    with openslide.OpenSlide(str(wsi_path)) as slide:
        if tuple(slide.dimensions) != tuple(level0_size):
            raise ValueError("WSI dimensions do not match the feature HDF5 coordinate frame")
        for index in unique:
            row = by_index[index]
            x, y = row["x_level0"], row["y_level0"]
            if min(x, y) < 0 or x + patch_span > slide.dimensions[0] or y + patch_span > slide.dimensions[1]:
                raise ValueError("A selected crop falls outside the matched WSI")
            crops.append((row, slide.read_region((x, y), 0, (patch_span, patch_span)).convert("RGB")))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tile = 256
    label_height = 38
    columns = min(4, len(crops))
    rows_count = (len(crops) + columns - 1) // columns
    montage = Image.new("RGB", (columns * tile, rows_count * (tile + label_height)), "white")
    drawer = ImageDraw.Draw(montage)
    manifest = []
    for position, (row, image) in enumerate(crops):
        name = f"patch_{row['consumed_patch_index']:05d}.png"
        image.save(output / name)
        view = image.copy()
        view.thumbnail((tile, tile))
        col, line = position % columns, position // columns
        left, top = col * tile, line * (tile + label_height)
        montage.paste(view, (left + (tile - view.width) // 2, top + (tile - view.height) // 2))
        drawer.text((left + 5, top + tile + 3),
                    f"patch {row['consumed_patch_index']}  {row['interaction_pp']:+.3f} pp", fill="black")
        drawer.text((left + 5, top + tile + 19), f"({row['x_level0']}, {row['y_level0']})", fill="black")
        manifest.append({"file": name, "consumed_patch_index": row["consumed_patch_index"],
                         "source_patch_index": row["source_patch_index"],
                         "interaction_pp": row["interaction_pp"]})
    montage.save(output / "montage.png")
    return {"wsi_basename": wsi_path.name, "wsi_size_bytes": wsi_path.stat().st_size,
            "crop_count": len(crops), "patch_size_level0": patch_span,
            "crops": manifest, "montage": "montage.png"}
