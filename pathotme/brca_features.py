"""Versioned breast-only panels; selection is fixed without label association.

This module is dependency-light. The shared reference preserves the NSCLC
measurement/transform definitions but never loads lung rows. The primary
panel adds segmented-tissue geometry, not molecular or single-file markers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from pathotme import features as shared

REVISION = "9262bc0cd0cd7774d0f7bcbfe6ae5f4665898b25"
PANELS = ("brca_morph64_v1", "brca_shared62_v1")


def digest(path):
    """Hash file bytes without importing tensor libraries."""
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def panel_spec(name):
    """Return the ordered feature, source, token and transform contract."""
    if name not in PANELS:
        raise ValueError(f"unknown BRCA panel: {name}")
    cells = {f"cell_{cell.lower()}": tuple(
        c for c in shared.CELL_COMPOSITION if f"_{cell}_IN_" in c)
        for cell in shared.CELL_CLASSES}
    if name == "brca_shared62_v1":
        low = dict(zip(("global_tissue", "compartment_extent", "tumor_core_composition",
                        "invasive_margin_composition", "tls"),
                       (shared.GLOBAL_TISSUE, shared.WHOLE_TUMOR_TISSUE[:3],
                        shared.WHOLE_TUMOR_TISSUE[3:6], shared.WHOLE_TUMOR_TISSUE[6:],
                        shared.TLS_DERIVED)))
        spatial_groups = shared.SPATIAL_CELL_GROUPS
        sources = {k.replace("lung_cancer", "breast_cancer"): v
                   for k, v in shared.SOURCE_COLUMNS.items()}
        names = shared.FEATURE_NAMES
        percentages = shared.PERCENT_TO_FRACTION
        logarithms = shared.LOG1P_NONNEGATIVE
    else:
        low = {
            "tissue_composition": tuple(f"RELATIVE_AREA_{t}" for t in
                ("CARCINOMA", "STROMA", "EPITHELIAL_TISSUE", "NECROSIS", "VESSEL")),
            **{f"{t.lower()}_geometry": tuple(f"{m}_{t}" for m in
                ("AVG_ECCENTRICITY", "AVG_SOLIDICITY", "LARGEST_ROUNDNESS"))
                + (f"REGIONS_PER_MM2_VALID_TISSUE_{t}",) for t in ("CARCINOMA", "STROMA")},
            "compartment_extent": shared.WHOLE_TUMOR_TISSUE[:3],
            **{f"{label}_composition": tuple(f"RELATIVE_AREA_{t}_IN_{region}" for t in
                ("CARCINOMA", "STROMA", "EPITHELIAL_TISSUE", "NECROSIS"))
                for label, region in (("tumor_core", "TUMOR_CORE"),
                                      ("invasive_margin", "INNER_INVASIVE_MARGIN"))},
        }
        spatial_groups = ("FIBROBLASTS", "LYMPHOCYTES", "MACROPHAGES")
        morphology = tuple(c for group in list(low.values())[:3] for c in group
                           if not c.startswith("REGIONS_PER_"))
        sources = {
            "tme_features_breast_cancer_RUO.csv": morphology + (
                "REGION_COUNT_CARCINOMA", "REGION_COUNT_STROMA", "ABSOLUTE_AREA_VALID_TISSUE"),
            "whole_tumor_region_tissue_features_breast_cancer_RUO.csv": tuple(
                c for group in list(low.values())[3:] for c in group),
            "whole_tumor_region_cell_features_breast_cancer_RUO.csv": shared.CELL_COMPOSITION,
            "whole_tumor_region_neighborhood_features_breast_cancer_RUO.csv": tuple(
                c for g in spatial_groups for c in shared.SPATIAL_INTERACTIONS
                if f"_OF_{g}_AROUND_" in c),
        }
        names = tuple(c for group in low.values() for c in group) + shared.CELL_COMPOSITION + sources[
            "whole_tumor_region_neighborhood_features_breast_cancer_RUO.csv"]
        # The release's WTR relative areas are fractions, unlike slide-level %.
        percentages = low["tissue_composition"] + tuple(
            c for c in shared.CELL_COMPOSITION if c.startswith("CELL_PERCENTAGE_"))
        logarithms = tuple(c for c in names if c.startswith((
            "CELL_DENSITY_", "AVG_MIN_DISTANCE_", "REGIONS_PER_MM2_")))
    spatial = {f"spatial_{g.lower()}": tuple(c for c in shared.SPATIAL_INTERACTIONS
               if f"_OF_{g}_AROUND_" in c) for g in spatial_groups}
    high = {**cells, **spatial}
    covered = [c for group in (*low.values(), *high.values()) for c in group]
    if len(covered) != len(set(covered)) or set(covered) != set(names):
        raise ValueError("token schema must cover each feature exactly once")
    spec = {"panel": name, "cohort": "TCGA-BRCA", "revision": REVISION,
            "feature_names": names, "sources": sources, "low_tokens": low,
            "high_tokens": high, "percent_to_fraction": percentages,
            "log1p": logarithms,
            "missing_policy": "non_TLS_NaN_train_median; TLS_missing_count_zero_reference_only",
            "fold_fitted": "training_median_then_training_mean_population_std_no_indicators",
            "geometry_derivation": "region_count / valid_tissue_area_um2 * 1e6; invalid denominator => NaN"}
    spec["schema_sha256"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec


def number(value):
    """Parse quantitative values; missing stays missing and infinity is rejected."""
    if value is None or str(value).strip().lower() in ("", "nan", "na", "n/a", "null"):
        return float("nan")
    result = float(value)
    if math.isinf(result):
        raise ValueError("infinite quantitative value")
    return result


def selected_rows(root, panel):
    """Join exact slide barcodes across breast tables, without any labels."""
    root = Path(root)
    if root.name != "breast_cancer":
        root = root / "data/breast_cancer"
    spec = panel_spec(panel)
    tables = []
    for filename, columns in spec["sources"].items():
        with (root / filename).open() as handle:
            reader = csv.DictReader(handle)
            required = {"TCGA_FILE_NAME", "TCGA_PROJECT_ID", *columns}
            if required - set(reader.fieldnames):
                raise ValueError(f"missing columns in {filename}: {required - set(reader.fieldnames)}")
            table = {}
            for row in reader:
                if row["TCGA_PROJECT_ID"].replace("_", "-") != "TCGA-BRCA":
                    raise ValueError("non-BRCA data in breast panel")
                slide = row["TCGA_FILE_NAME"].split(".")[0]
                if slide in table:
                    raise ValueError(f"duplicate slide barcode: {slide}")
                table[slide] = {c: number(row[c]) for c in columns}
            tables.append(table)
    if not tables[0] or any(set(t) != set(tables[0]) for t in tables[1:]):
        raise ValueError("breast source tables have unequal/empty slide universes; no inner-join dropping")
    result = []
    for slide in sorted(tables[0]):
        row = {c: v for table in tables for c, v in table[slide].items()}
        if panel == "brca_shared62_v1":
            for stage in ("IMMATURE", "MATURE"):
                count = row[f"COUNT_TLS_{stage}"]
                # Exact legacy reference convention, not newly inferred semantics.
                count = 0.0 if math.isnan(count) else max(0.0, count)
                row[f"TLS_{stage}_PRESENT"] = float(count > 0)
                row[f"LOG1P_COUNT_TLS_{stage}"] = math.log1p(count)
        else:
            area = row["ABSOLUTE_AREA_VALID_TISSUE"]
            for tissue in ("CARCINOMA", "STROMA"):
                count = row[f"REGION_COUNT_{tissue}"]
                if count < 0 or area < 0:
                    raise ValueError("negative area or region count")
                row[f"REGIONS_PER_MM2_VALID_TISSUE_{tissue}"] = (
                    count * 1e6 / area if area > 0 else float("nan"))
        values = {c: row[c] for c in spec["feature_names"]}
        if all(math.isnan(v) for v in values.values()):
            raise ValueError(f"all-missing slide: {slide}")
        result.append({"slide_id": slide, **values})
    return result


def transform_rows(rows, panel):
    """Apply fixed unit/log transforms only; no cohort-wide fitted statistics."""
    spec = panel_spec(panel)
    output = []
    for row in rows:
        values = []
        for column in spec["feature_names"]:
            value = number(row[column])
            if value < 0:
                raise ValueError(f"negative TME measurement: {column}")
            if column in spec["percent_to_fraction"]:
                value /= 100.0
            if column in spec["log1p"]:
                value = math.log1p(value)
            values.append(value)
        output.append(values)
    return output


def write_panel(root, output, panel):
    """Write a new private feature table and source-byte provenance; never overwrite."""
    output = Path(output)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("panel artifacts already exist; use a versioned output")
    spec = panel_spec(panel)
    rows = selected_rows(root, panel)
    transform_rows(rows, panel)  # validate domain before publishing anything
    source_root = Path(root)
    if source_root.name != "breast_cancer":
        source_root /= "data/breast_cancer"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slide_id", *spec["feature_names"]])
        writer.writeheader(); writer.writerows(rows)
    metadata = {**spec, "slide_count": len(rows), "selected_table_sha256": digest(output),
                "source_sha256": {k: digest(source_root / k) for k in spec["sources"]},
                "builder_sha256": digest(__file__), "redistribution": "prohibited",
                "selection": "fixed biological-role hypothesis; no labels or held-out associations"}
    with metadata_path.open("x") as handle:
        json.dump(metadata, handle, indent=2); handle.write("\n")
    return metadata
