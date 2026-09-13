"""Exercise every runner branch on tiny synthetic bags, without local weights."""
import copy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts"),"/path/to/PGVL-Gym"]
import run_controlled_vila as runner
from common.configuration import load_yaml_config
from methods.vila_mil.adapter import ViLaMILMethod
from methods.vila_mil.model import ViLa_MIL_Model
from pathotme.brca_features import panel_spec, transform_rows
from pathotme.controlled_vila import CONDITIONS, donor_maps


def test_all_conditions_complete_and_students_export_without_tme(tmp_path,monkeypatch):
    import train
    runner.environment()
    cfg=load_yaml_config("/path/to/PGVL-Gym/benchmarks/tcga_brca/configs/vila_mil_conch/brca_16shot.yaml")
    contract=load_yaml_config(ROOT/"configs/vila_conch_brca_controlled_16shot.yaml")
    contract["training"]["epochs"]=1;contract["results_root"]=str(tmp_path/"results")
    contract["base_checkpoint_dir"]=str(tmp_path/"baseline")
    baseline=Path(contract["base_checkpoint_dir"]);baseline.mkdir()
    torch.manual_seed(7)
    base=ViLa_MIL_Model(SimpleNamespace(input_size=16,hidden_size=8,prototype_number=3),
                        num_classes=2,prompt_features=torch.randn(4,16))
    torch.save(base.state_dict(),baseline/"fold0_best.pt")
    monkeypatch.setattr(ViLaMILMethod,"build_model",lambda self:copy.deepcopy(base).to(self.device))
    phases={p:[{"slide_id":f"{p}{i}","case_id":f"{p}{i}","label_id":i} for i in (0,1)] for p in ("train","val","test")}
    loaders=[]
    for rows in phases.values():
        loaders.append([(torch.randn(1,4,16),torch.randn(1,6,16),
                         {"slide_id":[r["slide_id"]],"case_id":[r["case_id"]]},torch.tensor([r["label_id"]])) for r in rows])
    monkeypatch.setattr(train,"build_loaders",lambda *a:loaders)
    names=panel_spec(contract["panel"])["feature_names"]
    raw=[{"slide_id":r["slide_id"],**{n:str(r["label_id"]+1) for n in names}} for rows in phases.values() for r in rows]
    feature_csv=tmp_path/"tme.csv"
    with feature_csv.open("w") as h:
        w=csv.DictWriter(h,fieldnames=["slide_id",*names]);w.writeheader();w.writerows(raw)
    ref={"selected_features_csv":str(feature_csv),"results_root":str(tmp_path/"reference"),"panel":contract["panel"]}
    monkeypatch.setattr(runner,"load_yaml_config",lambda path:ref if str(path)==str(contract["reference_contract"]) else cfg)
    control_dir=Path(ref["results_root"])/"tme_only/fold0";control_dir.mkdir(parents=True)
    values=np.asarray(transform_rows(raw,contract["panel"]))
    classifier=make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression()).fit(values[:2],[0,1])
    joblib.dump({"pipeline":classifier,"identity":"reference_id"},control_dir/"fold0_classifier.joblib")
    for folder in (control_dir,baseline):
        with (folder/"fold0_predictions.csv").open("w") as h:
            w=csv.DictWriter(h,fieldnames=["slide_id","case_id","label","probability_0","probability_1"]);w.writeheader()
            for r in phases["test"]:w.writerow({"slide_id":r["slide_id"],"case_id":r["case_id"],"label":r["label_id"],"probability_0":.4,"probability_1":.6})
    identities={c:f"id_{c}" for c in CONDITIONS}
    for condition in CONDITIONS:
        payload={"donor_maps":donor_maps(phases,1,0),"reference_tme_only_identity":"reference_id"}
        if condition in ("kd_real","kd_shuffled"):
            payload["teacher_identity"]=identities[contract["student"]["teachers"][condition]]
        output=runner.output_dir(contract,0,condition);output.mkdir(parents=True)
        runner.execute(contract,cfg,phases,payload,identities[condition],output,0,condition,"cpu",False)
        report=runner.verify_completed(output,identities[condition])
        assert report["status"]=="completed"
        if condition in runner.STUDENTS:
            export=torch.load(output/"fold0_inference.pt",weights_only=True)
            assert not any("auxiliary" in k or "standardizer" in k or "conditioner" in k for k in export["state_dict"])
            clone=copy.deepcopy(base);clone.load_state_dict(export["state_dict"],strict=True)
            assert json.loads((output/"metrics.json").read_text())["condition"]==condition
