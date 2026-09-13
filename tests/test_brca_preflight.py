"""Synthetic strict experiment-contract checks without Torch or private data."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import run_brca_vila as runner
from pathotme.brca_features import panel_spec


def csv_file(path,columns,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as h:
        w=csv.DictWriter(h,fieldnames=columns);w.writeheader();w.writerows(rows)


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    panel=panel_spec("brca_morph64_v1");base=tmp_path/"base";base.mkdir()
    feature=tmp_path/"feature.bin";feature.touch()
    prompt=tmp_path/"prompt.csv";prompt.write_text("BRCA task-owned bank")
    config=tmp_path/"base.yaml";config.write_text("placeholder")
    splits=tmp_path/"splits";all_rows=[];test=[]
    for phase,n in [("train",32),("val",32),("test",2)]:
        rows=[{"slide_id":f"{phase}{i}","case_id":f"patient-{phase}{i}","label_id":i%2,
               "low":str(feature),"high":str(feature)} for i in range(n)]
        csv_file(splits/"fold0"/f"{phase}.csv",list(rows[0]),rows);all_rows+=rows
        if phase=="test":test=rows
    manifest=tmp_path/"manifest.csv";csv_file(manifest,list(all_rows[0]),all_rows)
    table=tmp_path/"tme.csv"
    csv_file(table,["slide_id",*panel["feature_names"]],[{"slide_id":r["slide_id"],
        **{c:1 for c in panel["feature_names"]}} for r in all_rows])
    metadata={"panel":panel["panel"],"cohort":"TCGA-BRCA","revision":panel["revision"],
        "schema_sha256":panel["schema_sha256"],"selected_table_sha256":runner.sha(table),
        "builder_sha256":runner.sha(ROOT/"pathotme/brca_features.py")}
    table.with_suffix(".csv.metadata.json").write_text(json.dumps(metadata))
    for filename in ["metrics.json","config.json"]:(base/filename).write_text("{}")
    (base/"fold0_best.pt").write_text("synthetic checkpoint identity only")
    csv_file(base/"fold0_predictions.csv",["slide_id","case_id","label"],[
        {"slide_id":r["slide_id"],"case_id":r["case_id"],"label":r["label_id"]} for r in test])
    cfg={"method":"vila_mil","task":"brca","backbone":"clip-rn50","shots":16,
        "feature_space_id":"openai/clip-rn50@official","feature_resolutions":{"low":"5x","high":"10x"},
        "label_dict":{"IDC":0,"ILC":1},"classnames":["invasive ductal carcinoma","invasive lobular carcinoma"],
        "vila_prompt_file_classnames":["IDC","ILC"],"results_dir":str(base),"split_dir":str(splits),
        "feature_path_column_s":"low","feature_path_column_l":"high","text_prompt_path":str(prompt),
        "vila_prompt_file_sha256":runner.sha(prompt),"dataset_csv":str(manifest)}
    monkeypatch.setattr(runner,"load_yaml_config",lambda p:cfg)
    monkeypatch.setattr(runner,"validate_resume_state",lambda *a:[{"fold":0,"sample_failures":{}}])
    contract={"panel":panel["panel"],"exact_coverage_folds":[0],"shots":16,"seed":1,
        "model":{"raw_feature_count":64,"tokenization":panel["panel"]},
        "training":{"optimizer":"adam","checkpoint_monitor":"val_error","epochs":200},
        "tme_only":{"model":"logistic_regression","checkpoint_monitor":"val_error","c_grid":[.01,1]},
        "base_config":str(config),"base_checkpoint_dir":str(base),"selected_features_csv":str(table)}
    return contract,cfg


def test_preflight_binds_panel_mode_source_bytes_and_execution(experiment):
    contract,cfg=experiment
    actual=runner.preflight(contract,0,"actual")[3]
    assert actual!=runner.preflight(contract,0,"zero")[3]
    assert actual!=runner.preflight(contract,0,"actual","smoke")[3]
    assert actual!=runner.preflight(contract,0,"actual","tme_only")[3]
    (Path(cfg["results_dir"])/"fold0_best.pt").write_text("changed checkpoint")
    assert actual!=runner.preflight(contract,0,"actual")[3]


@pytest.mark.parametrize("fault",["labels","prompt","table","patient_leak","missing_feature","missing_tme"])
def test_preflight_rejects_semantic_or_asset_changes(experiment,fault):
    contract,cfg=experiment
    if fault=="labels":cfg["label_dict"]={"ILC":0,"IDC":1}
    elif fault=="prompt":Path(cfg["text_prompt_path"]).write_text("wrong bank")
    elif fault=="table":Path(contract["selected_features_csv"]).write_text("wrong table")
    else:
        p=Path(cfg["split_dir"])/"fold0/val.csv";content=p.read_text()
        if fault=="patient_leak":content=content.replace("patient-val0","patient-train0")
        elif fault=="missing_feature":content=content.replace("feature.bin","missing.bin")
        else:content=content.replace("val0,","absent-slide,",1)
        p.write_text(content)
    with pytest.raises(ValueError):runner.preflight(contract,0,"actual")


def test_check_only_imports_are_dependency_light():
    import subprocess
    result=subprocess.run([sys.executable,"-c",f"import sys;sys.path.insert(0,{str(ROOT/'scripts')!r});import run_brca_vila;assert 'torch' not in sys.modules;assert 'pandas' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
