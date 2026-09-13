"""Dependency-light breast panel tests, independent of private assets."""
import csv
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pathotme.brca_features import panel_spec, selected_rows, transform_rows, write_panel


def source(tmp_path, panel="brca_morph64_v1"):
    root = tmp_path / "data/breast_cancer"
    root.mkdir(parents=True)
    for filename, cols in panel_spec(panel)["sources"].items():
        with (root / filename).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["TCGA_FILE_NAME", "TCGA_PROJECT_ID", *cols])
            writer.writeheader()
            writer.writerow({"TCGA_FILE_NAME": "TCGA-X-1-01Z-DX1.uuid.svs",
                             "TCGA_PROJECT_ID": "TCGA_BRCA", **{c: 1 for c in cols},
                             **({"ABSOLUTE_AREA_VALID_TISSUE": 2e6} if "ABSOLUTE_AREA_VALID_TISSUE" in cols else {})})
    return tmp_path


def test_panels_are_distinct_and_every_feature_belongs_to_exactly_one_token():
    for name, width, counts in [("brca_morph64_v1",64,(6,10)),("brca_shared62_v1",62,(5,11))]:
        spec=panel_spec(name)
        assert len(spec["feature_names"]) == width
        assert (len(spec["low_tokens"]),len(spec["high_tokens"])) == counts
        flat=[c for group in (*spec["low_tokens"].values(),*spec["high_tokens"].values()) for c in group]
        assert len(flat)==len(set(flat))==width
        assert set(flat)==set(spec["feature_names"])
        assert all("breast_cancer" in p for p in spec["sources"])
    assert not any("TLS" in c for c in panel_spec("brca_morph64_v1")["feature_names"])


def test_numeric_derivation_units_and_schema(tmp_path):
    rows=selected_rows(source(tmp_path),"brca_morph64_v1")
    assert rows[0]["REGIONS_PER_MM2_VALID_TISSUE_CARCINOMA"]==0.5
    spec=panel_spec("brca_morph64_v1"); values=transform_rows(rows,spec["panel"])[0]
    assert values[spec["feature_names"].index("RELATIVE_AREA_CARCINOMA")]==0.01
    assert values[spec["feature_names"].index("RELATIVE_AREA_CARCINOMA_IN_TUMOR_CORE")]==1
    assert values[spec["feature_names"].index("REGIONS_PER_MM2_VALID_TISSUE_CARCINOMA")]==math.log1p(0.5)


def test_shared_reference_uses_breast_rows_with_legacy_62_order(tmp_path):
    from pathotme.features import FEATURE_NAMES
    rows=selected_rows(source(tmp_path,"brca_shared62_v1"),"brca_shared62_v1")
    assert tuple(rows[0])==("slide_id",*FEATURE_NAMES)
    assert rows[0]["TLS_MATURE_PRESENT"]==1
    assert rows[0]["LOG1P_COUNT_TLS_MATURE"]==math.log(2)


def test_provenance_and_no_overwrite(tmp_path):
    root=source(tmp_path); output=tmp_path/"selected.csv"
    report=write_panel(root,output,"brca_morph64_v1")
    assert report["cohort"]=="TCGA-BRCA" and len(report["source_sha256"])==4
    with pytest.raises(FileExistsError): write_panel(root,output,"brca_morph64_v1")


@pytest.mark.parametrize("fault",["wrong_cohort","duplicate","missing_column","unequal_rows"])
def test_source_contract_fails_closed(tmp_path,fault):
    root=source(tmp_path); p=root/"data/breast_cancer/tme_features_breast_cancer_RUO.csv"
    content=p.read_text()
    if fault=="wrong_cohort":content=content.replace("TCGA_BRCA","TCGA-LUAD")
    elif fault=="duplicate":content+=content.splitlines()[1]+"\n"
    elif fault=="missing_column":content=content.replace("REGION_COUNT_CARCINOMA","WRONG")
    else:content=content.splitlines()[0]+"\n"
    p.write_text(content)
    with pytest.raises(ValueError):selected_rows(root,"brca_morph64_v1")
