"""Leakage-safe OpenTME feature contract for TCGA-NSCLC.

The panel is fixed from biological roles before looking at subtype association.
Identifiers, project names, indication fields, image resolution, and every
other metadata column are deliberately excluded.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence


GLOBAL_TISSUE = (
    "RELATIVE_AREA_VALID_TISSUE",
    "RELATIVE_AREA_CARCINOMA",
    "RELATIVE_AREA_STROMA",
    "RELATIVE_AREA_NECROSIS",
    "RELATIVE_AREA_EPITHELIAL_TISSUE",
    "RELATIVE_AREA_VESSEL",
)

WHOLE_TUMOR_TISSUE = (
    "RELATIVE_AREA_TUMOR_CORE",
    "RELATIVE_AREA_INNER_INVASIVE_MARGIN",
    "RELATIVE_AREA_OUTER_INVASIVE_MARGIN",
    "RELATIVE_AREA_CARCINOMA_IN_TUMOR_CORE",
    "RELATIVE_AREA_STROMA_IN_TUMOR_CORE",
    "RELATIVE_AREA_NECROSIS_IN_TUMOR_CORE",
    "RELATIVE_AREA_STROMA_IN_INNER_INVASIVE_MARGIN",
    "RELATIVE_AREA_CARCINOMA_IN_INNER_INVASIVE_MARGIN",
)

CELL_CLASSES = (
    "CARCINOMA_CELL",
    "LYMPHOCYTE",
    "MACROPHAGE",
    "GRANULOCYTE",
    "PLASMA_CELL",
    "FIBROBLAST",
    "ENDOTHELIAL_CELL",
)
CELL_REGIONS = ("TUMOR_CORE", "INNER_INVASIVE_MARGIN")
CELL_COMPOSITION = tuple(
    f"{metric}_{cell_class}_IN_{region}"
    for region in CELL_REGIONS
    for cell_class in CELL_CLASSES
    for metric in ("CELL_PERCENTAGE", "CELL_DENSITY")
)

SPATIAL_CELL_GROUPS = (
    "LYMPHOCYTES",
    "MACROPHAGES",
    "FIBROBLASTS",
    "PLASMA_CELLS",
)
SPATIAL_INTERACTIONS = tuple(
    f"{metric}_OF_{cell_group}_AROUND_CARCINOMA_CELL_"
    f"IN_WHOLE_TUMOR_REGION_{radius}"
    for cell_group in SPATIAL_CELL_GROUPS
    for radius in (20, 40)
    for metric in ("RATIO", "AVG_MIN_DISTANCE")
)

RAW_TLS_COUNTS = ("COUNT_TLS_IMMATURE", "COUNT_TLS_MATURE")
TLS_DERIVED = (
    "TLS_IMMATURE_PRESENT",
    "TLS_MATURE_PRESENT",
    "LOG1P_COUNT_TLS_IMMATURE",
    "LOG1P_COUNT_TLS_MATURE",
)

FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = OrderedDict((
    ("tissue_architecture", GLOBAL_TISSUE + WHOLE_TUMOR_TISSUE),
    ("cell_composition", CELL_COMPOSITION),
    ("spatial_interactions", SPATIAL_INTERACTIONS),
    ("tls", TLS_DERIVED),
))
FEATURE_NAMES = tuple(
    feature for group in FEATURE_GROUPS.values() for feature in group)

SOURCE_COLUMNS: Mapping[str, tuple[str, ...]] = OrderedDict((
    ("tme_features_lung_cancer_RUO.csv", GLOBAL_TISSUE),
    ("whole_tumor_region_tissue_features_lung_cancer_RUO.csv",
     WHOLE_TUMOR_TISSUE),
    ("whole_tumor_region_cell_features_lung_cancer_RUO.csv",
     CELL_COMPOSITION),
    ("whole_tumor_region_neighborhood_features_lung_cancer_RUO.csv",
     SPATIAL_INTERACTIONS),
    ("tertiary_lymphoid_structures_tls_features_lung_cancer_RUO.csv",
     RAW_TLS_COUNTS),
))

_schema_payload = json.dumps(
    {"groups": FEATURE_GROUPS, "sources": SOURCE_COLUMNS},
    separators=(",", ":"), sort_keys=False).encode("utf-8")
SCHEMA_SHA256 = hashlib.sha256(_schema_payload).hexdigest()

PERCENT_TO_FRACTION = GLOBAL_TISSUE + tuple(
    name for name in CELL_COMPOSITION if name.startswith("CELL_PERCENTAGE_"))
LOG1P_NONNEGATIVE = tuple(
    name for name in CELL_COMPOSITION if name.startswith("CELL_DENSITY_")) + tuple(
        name for name in SPATIAL_INTERACTIONS
        if name.startswith("AVG_MIN_DISTANCE_"))
TRANSFORM_CONTRACT = {
    "percent_to_fraction": PERCENT_TO_FRACTION,
    "log1p_nonnegative": LOG1P_NONNEGATIVE,
    "unchanged": tuple(
        name for name in FEATURE_NAMES
        if name not in set(PERCENT_TO_FRACTION + LOG1P_NONNEGATIVE)),
    "fold_fitted": ("training_median_imputation_with_indicators", "z_score"),
}
TRANSFORM_SHA256 = hashlib.sha256(json.dumps(
    TRANSFORM_CONTRACT, separators=(",", ":"), sort_keys=False
).encode("utf-8")).hexdigest()


def _slide_id(filename: object) -> str:
    """Return the TCGA slide barcode before the filename UUID/suffix."""
    value = str(filename)
    return value.split(".", 1)[0]


def build_nsclc_feature_table(opentme_root: str | Path):
    """Build the ordered 62-variable NSCLC feature table.

    TLS absence is structural: missing counts are converted to zero before
    deriving presence and log-count variables. Other missing values remain
    missing so every fold can impute them using training patients only.
    """
    import numpy as np
    import pandas as pd

    root = Path(opentme_root)
    if root.name != "lung_cancer" and (root / "data" / "lung_cancer").is_dir():
        root = root / "data" / "lung_cancer"
    if not root.is_dir():
        raise FileNotFoundError(f"OpenTME lung-cancer directory is missing: {root}")

    frames = []
    for filename, columns in SOURCE_COLUMNS.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        header = tuple(pd.read_csv(path, nrows=0).columns)
        missing = sorted(set(columns) - set(header))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame = pd.read_csv(path, usecols=["TCGA_FILE_NAME", *columns])
        frame.insert(0, "slide_id", frame.pop("TCGA_FILE_NAME").map(_slide_id))
        if frame["slide_id"].duplicated().any():
            duplicates = frame.loc[
                frame["slide_id"].duplicated(keep=False), "slide_id"]
            raise ValueError(
                f"{path} contains duplicate slide IDs: "
                f"{', '.join(sorted(set(duplicates))[:5])}")
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)

    table = frames[0]
    for frame in frames[1:]:
        table = table.merge(frame, on="slide_id", how="inner",
                            validate="one_to_one")

    immature = table["COUNT_TLS_IMMATURE"].fillna(0).clip(lower=0)
    mature = table["COUNT_TLS_MATURE"].fillna(0).clip(lower=0)
    table["TLS_IMMATURE_PRESENT"] = immature.gt(0).astype(float)
    table["TLS_MATURE_PRESENT"] = mature.gt(0).astype(float)
    table["LOG1P_COUNT_TLS_IMMATURE"] = np.log1p(immature)
    table["LOG1P_COUNT_TLS_MATURE"] = np.log1p(mature)
    table = table.loc[:, ["slide_id", *FEATURE_NAMES]].sort_values(
        "slide_id", kind="stable").reset_index(drop=True)

    if len(FEATURE_NAMES) != 62 or len(set(FEATURE_NAMES)) != 62:
        raise AssertionError("The OpenTME core panel must contain 62 unique features")
    if table.empty or table["slide_id"].duplicated().any():
        raise ValueError("OpenTME feature join produced an invalid slide table")
    if table[list(FEATURE_NAMES)].isna().all(axis=1).any():
        raise ValueError("At least one slide has no selected OpenTME measurements")
    return table


def write_nsclc_feature_table(
    opentme_root: str | Path,
    output_csv: str | Path,
    *,
    source_revision: str,
) -> dict:
    """Materialize the non-redistributable selected table and provenance."""
    table = build_nsclc_feature_table(opentme_root)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, output)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "source": "Aignostics/OpenTME",
        "source_revision": source_revision,
        "redistribution": "prohibited",
        "cohort": "TCGA-NSCLC",
        "slide_count": int(len(table)),
        "feature_count": len(FEATURE_NAMES),
        "feature_schema_sha256": SCHEMA_SHA256,
        "selected_table_sha256": digest,
        "feature_groups": {
            name: list(columns) for name, columns in FEATURE_GROUPS.items()},
        "source_files": list(SOURCE_COLUMNS),
        "excluded_metadata": [
            "TCGA_FILE_NAME", "TCGA_SLIDE_UUID", "TCGA_CASE_ID",
            "TCGA_PROJECT_ID", "INDICATION", "IMAGE_RESOLUTION"],
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(metadata_tmp, metadata_path)
    return metadata


def validate_feature_table(frame, expected_slides: Sequence[str] | None = None):
    """Validate column order, uniqueness, and optional split coverage."""
    expected_columns = ("slide_id", *FEATURE_NAMES)
    if tuple(frame.columns) != expected_columns:
        raise ValueError("Selected OpenTME table does not match the pinned schema")
    if frame["slide_id"].duplicated().any():
        raise ValueError("Selected OpenTME table contains duplicate slide IDs")
    if expected_slides is not None:
        missing = sorted(set(map(str, expected_slides)) - set(frame["slide_id"]))
        if missing:
            raise ValueError(
                "OpenTME is missing required split slides: " + ", ".join(missing[:5]))
    return frame


def transform_nsclc_features(frame):
    """Apply deterministic unit/skew transforms before fold-fitted scaling.

    The input table is copied. Missing values remain missing for training-fold
    median imputation; no statistic is learned in this function.
    """
    import numpy as np

    transformed = frame.loc[:, list(FEATURE_NAMES)].copy()
    for column in PERCENT_TO_FRACTION:
        transformed[column] = transformed[column] / 100.0
    for column in LOG1P_NONNEGATIVE:
        observed = transformed[column].dropna()
        if (observed < 0).any():
            raise ValueError(
                f"OpenTME nonnegative feature {column} contains negative values")
        transformed[column] = np.log1p(transformed[column])
    return transformed
