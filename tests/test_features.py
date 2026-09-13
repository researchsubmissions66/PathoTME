import pandas as pd
import pytest

from pathotme.features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    transform_nsclc_features,
)


def test_core_panel_is_ordered_unique_and_62_wide():
    assert len(FEATURE_NAMES) == 62
    assert len(set(FEATURE_NAMES)) == 62
    assert [len(group) for group in FEATURE_GROUPS.values()] == [14, 28, 16, 4]


def test_core_panel_excludes_identity_and_label_proxies():
    forbidden = {"TCGA_PROJECT_ID", "INDICATION", "TCGA_CASE_ID", "TCGA_FILE_NAME"}
    assert forbidden.isdisjoint(FEATURE_NAMES)
    assert all(not feature.startswith("TCGA_") for feature in FEATURE_NAMES)


def test_semantic_unit_and_skew_transforms():
    row = {name: 0.0 for name in FEATURE_NAMES}
    row["RELATIVE_AREA_VALID_TISSUE"] = 50.0
    row["CELL_PERCENTAGE_LYMPHOCYTE_IN_TUMOR_CORE"] = 25.0
    row["CELL_DENSITY_LYMPHOCYTE_IN_TUMOR_CORE"] = 99.0
    distance = (
        "AVG_MIN_DISTANCE_OF_LYMPHOCYTES_AROUND_CARCINOMA_CELL_"
        "IN_WHOLE_TUMOR_REGION_20")
    row[distance] = 9.0
    transformed = transform_nsclc_features(pd.DataFrame([row]))
    assert transformed["RELATIVE_AREA_VALID_TISSUE"].iloc[0] == 0.5
    assert transformed[
        "CELL_PERCENTAGE_LYMPHOCYTE_IN_TUMOR_CORE"].iloc[0] == 0.25
    assert transformed[
        "CELL_DENSITY_LYMPHOCYTE_IN_TUMOR_CORE"].iloc[0] == pytest.approx(
            4.605170186)
    assert transformed[distance].iloc[0] == pytest.approx(2.302585093)
