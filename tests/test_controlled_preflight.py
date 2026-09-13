"""Torch-free launch, provenance and external deployment regression checks."""
import copy
import csv
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),"/path/to/PGVL-Gym"]
import run_controlled_vila as runner
import launch_controlled_vila as launcher
from eval_wsi_only import validate_target
from pathotme.controlled_vila import CONDITIONS, SHUFFLE_POLICY
from prepare_cptac_brca import histology, specimen_matches


def test_histology_and_specimen_mapping_are_conservative():
    assert histology({"primary_diagnosis":"Infiltrating duct carcinoma, NOS","morphology":"8500/3"})[0]=="IDC"
    assert histology({"primary_diagnosis":"Invasive lobular carcinoma","morphology":"Not Reported"})[0]=="ILC"
    assert histology({"primary_diagnosis":"Infiltrating duct carcinoma, NOS","morphology":"8500/2"})[0] is None
    assert histology({"primary_diagnosis":"Infiltrating duct and lobular carcinoma","morphology":"8522/3"})[0] is None
    assert specimen_matches("abc_D1_D1","abc")
    assert not specimen_matches("abc-other","abc") and not specimen_matches("xxabc","abc")


def test_external_rejects_label_scale_patient_substitutions():
    target={"status":"ready","cohort":"CPTAC_BRCA","label_dict":{"IDC":0,"ILC":1},
            "backbone":"conch","feature_space_id":"hf:MahmoodLab/conch","feature_dim":512,
            "representation":"shared_embedding","magnifications":{"low":"5x","high":"10x"},
            "patch_sizes":{"low":512,"high":256},"fit_on_target":False}
    rows=[{"slide_id":"a","case_id":"a","label":"IDC"},{"slide_id":"b","case_id":"b","label":"ILC"}]
    cfg={"label_dict":target["label_dict"],"feature_space_id":target["feature_space_id"]}
    validate_target(target,rows,{"tcga"},cfg)
    for key,value in (("fit_on_target",True),("status","not_ready"),("feature_dim",768),
                      ("label_dict",{"ILC":0,"IDC":1}),("patch_sizes",{"low":256,"high":256})):
        with pytest.raises(ValueError):validate_target({**target,key:value},rows,{"tcga"},cfg)
    with pytest.raises(ValueError):validate_target(target,rows,{"a"},cfg)
    with pytest.raises(ValueError):validate_target(target,[rows[0]],set(),cfg)


def test_complete_requires_identity_and_artifacts(tmp_path):
    artifact=tmp_path/"predictions.csv";artifact.write_text("value\n1\n")
    record={"status":"completed","identity":"id","artifact_sha256":{artifact.name:runner.sha(artifact)}}
    (tmp_path/"metrics.json").write_text(json.dumps(record))
    assert runner.verify_completed(tmp_path,"id")
    with pytest.raises(ValueError):runner.verify_completed(tmp_path,"wrong")
    artifact.write_text("changed\n")
    with pytest.raises(ValueError):runner.verify_completed(tmp_path,"id")


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    feature=tmp_path/"feature.pt";feature.touch()
    phases={p:[{"slide_id":f"{p}{i}","case_id":f"{p}{i}","label_id":str(i),
                "low":str(feature),"high":str(feature)} for i in range(2)] for p in ("train","val","test")}
    for phase,rows in phases.items():
        path=tmp_path/"fold0"/f"{phase}.csv";path.parent.mkdir(exist_ok=True)
        with path.open("w") as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    predictions=[{"slide_id":r["slide_id"],"case_id":r["case_id"],"label":r["label_id"]} for r in phases["test"]]
    with (tmp_path/"fold0_predictions.csv").open("w") as h:
        writer=csv.DictWriter(h,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    (tmp_path/"metrics.json").write_text("{}")
    cfg={"method":"vila_mil","task":"brca","backbone":"conch","feature_dim":512,"shots":16,
         "feature_space_id":"hf:MahmoodLab/conch","encoder_extension_strategy":"paired_feature_context_v1",
         "encoder_extension":{"status":"extended"},"feature_resolutions":{"low":"5x","high":"10x"},
         "label_dict":{"IDC":0,"ILC":1},"results_dir":str(tmp_path),"split_dir":str(tmp_path),
         "feature_path_column_s":"low","feature_path_column_l":"high","backbone_weights":str(feature)}
    for k in ("classnames","vila_prompt_file_classnames","text_prompt_path","vila_prompt_file_sha256",
              "vila_prompt_bank_sha256","vila_prompt_format","vila_prompt_layout","vila_scale_recipe"):cfg[k]="fixed"
    contract={"conditions":list(CONDITIONS),"folds":[0],"panel":"brca_morph64_v1","shots":16,"seed":1,
              "shuffle_policy":SHUFFLE_POLICY,"reference_contract":"reference","base_config":"base",
              "base_checkpoint_dir":str(tmp_path),"training":{"optimizer":"adam","checkpoint_monitor":"val_error","epochs":1,"lr":.0001},
              "student":{"temperature":2,"auxiliary_weight":.1,"distillation_weight":1,
                         "teachers":{"kd_visual":"visual","kd_real":"actual","kd_shuffled":"shuffled"}},
              "fusion":{"selection":"validation_error_ties_smallest_alpha"}}
    ref_cfg=copy.deepcopy(cfg)
    monkeypatch.setattr(runner,"load_yaml_config",lambda p:{"results_root":str(tmp_path)} if p=="reference" else cfg)
    monkeypatch.setattr(runner,"reference_preflight",lambda *a:(ref_cfg,phases,{},"reference_id"))
    monkeypatch.setattr(runner,"verify_completed",lambda *a:{"artifact_sha256":{}})
    monkeypatch.setattr(runner,"validate_resume_state",lambda *a:[{"fold":0}])
    monkeypatch.setattr(runner,"sha",lambda p:"unchanged")
    return contract,cfg,feature


def test_preflight_detects_changed_condition_and_feature(experiment):
    contract,cfg,feature=experiment
    result=runner.preflight(contract,0,"aux_real")
    assert result[2]["inference_requires_tme"] is False
    assert runner.preflight(contract,0,"actual")[2]["inference_requires_tme"] is True
    assert runner.preflight(contract,0,"kd_real")[2]["teacher_identity"]==runner.preflight(contract,0,"actual")[3]
    assert result[3]!=runner.preflight(contract,0,"aux_shuffled")[3]
    with pytest.raises(ValueError):runner.preflight(contract,5,"aux_real")
    cfg["label_dict"]={"ILC":0,"IDC":1}
    with pytest.raises(ValueError):runner.preflight(contract,0,"aux_real")
    cfg["label_dict"]={"IDC":0,"ILC":1};feature.unlink()
    with pytest.raises(ValueError):runner.preflight(contract,0,"aux_real")


def test_launcher_smoke_and_teacher_dependencies(tmp_path,monkeypatch):
    contract={"results_root":str(tmp_path),"folds":list(range(5)),"conditions":list(CONDITIONS),
              "student":{"teachers":{"kd_real":"actual","kd_shuffled":"shuffled"}}}
    monkeypatch.setenv("PGVL_CONDA_ENV","/test/env");monkeypatch.setenv("PATHOTME_RESULTS_ROOT",str(tmp_path))
    monkeypatch.setattr(launcher,"load_yaml_config",lambda _:contract)
    monkeypatch.setattr(launcher,"preflight",lambda c,f,m,s:({}, {}, {},f"{f}-{m}-{s}"))
    monkeypatch.setattr(launcher,"verify_completed",lambda *a:None)
    plans=launcher.build_plan(tmp_path/"contract.yaml")
    assert len(plans)==51 and len(plans[0]["components"])==10
    for row in plans[1:]:
        assert "smoke" in row["depends_on"]
        if row["key"].startswith("kd_real"):
            assert f"actual-f{row['components'][0]['fold']}" in row["depends_on"]
        assert "--gres=gpu:1" in row["sbatch"] and "HF_HUB_OFFLINE=1" in row["sbatch"][-1]


def test_external_preflight_needs_no_tme_table(tmp_path):
    import eval_wsi_only as external
    source=tmp_path/"source";source.mkdir()
    native=tmp_path/"native";native.mkdir()
    for folder,name in ((source,"fold0_inference.pt"),(native,"fold0_best.pt")):(folder/name).write_bytes(b"checkpoint")
    prompt=tmp_path/"prompt.csv";prompt.write_text("fixed prompt\n")
    weights=tmp_path/"weights.bin";weights.write_bytes(b"weights")
    source_manifest=tmp_path/"source.csv";source_manifest.write_text("case_id\nTCGA-example\n")
    feature=tmp_path/"bag.h5";feature.write_bytes(b"header validated separately")
    cfg={"dataset_csv":str(source_manifest),"label_dict":{"IDC":0,"ILC":1},"feature_space_id":"hf:MahmoodLab/conch",
         "backbone_weights":str(weights),"text_prompt_path":str(prompt),"feature_path_column_s":"low","feature_path_column_l":"high",
         "tme_feature_csv":str(tmp_path/"DOES_NOT_EXIST_TME.csv")}
    (source/"config.json").write_text(json.dumps({"resolved_config":cfg}))
    record={"status":"completed","identity":"student-id","fold":0,
            "artifact_sha256":{name:external.digest(source/name) for name in ("config.json","fold0_inference.pt")},
            "provenance":{"inference_requires_tme":False,"contract":{"base_checkpoint_dir":str(native)},
                          "source_sha256":{str(native/"fold0_best.pt"):external.digest(native/"fold0_best.pt")}}}
    (source/"metrics.json").write_text(json.dumps(record))
    manifest=tmp_path/"target.csv"
    manifest.write_text(f"slide_id,case_id,label,low,high\na,a,IDC,{feature},{feature}\nb,b,ILC,{feature},{feature}\n")
    label_source=tmp_path/"labels.json";label_source.write_text("{}")
    audit=tmp_path/"audit.json";stat=feature.stat()
    audit.write_text(json.dumps({"status":"passed","manifest_sha256":external.digest(manifest),
                                "files":{str(feature):{"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}}}))
    target={"status":"ready","cohort":"CPTAC_BRCA","label_dict":{"IDC":0,"ILC":1},"backbone":"conch",
            "feature_space_id":"hf:MahmoodLab/conch","feature_dim":512,"representation":"shared_embedding",
            "magnifications":{"low":"5x","high":"10x"},"patch_sizes":{"low":512,"high":256},"fit_on_target":False,
            "manifest":str(manifest),"manifest_sha256":external.digest(manifest),"label_source":str(label_source),
            "label_source_sha256":external.digest(label_source),"runtime_encoder_checkpoint_sha256":external.digest(weights),
            "feature_audit":str(audit),"feature_audit_sha256":external.digest(audit)}
    target_path=tmp_path/"target.yaml";target_path.write_text(json.dumps(target))
    for native_mode in (False,True):
        _,rows,_,payload,_=external.preflight(source,target_path,native_mode)
        assert len(rows)==2 and payload["tme_files_accessed"] is False
