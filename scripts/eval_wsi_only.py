#!/usr/bin/env python3
"""Evaluate an exported student on a verified external manifest, with no TME I/O.

The target contract must explicitly attest the feature extraction and label
sources. No training, target normalization, threshold fitting or selection is
performed. The default is read-only preflight; --execute enables inference.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

PGVL=Path(os.environ.get("PGVL_REPO_ROOT","/path/to/PGVL-Gym"))
sys.path.insert(0,str(PGVL))
from common.configuration import load_dotenv, load_yaml_config


def digest(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):result.update(block)
    return result.hexdigest()


def table(path):
    with Path(path).open() as handle:return list(csv.DictReader(handle))


def validate_target(target, rows, source_cases, cfg):
    """Reject changed class/encoder/scale semantics and source-patient overlap."""
    expected={"cohort":"CPTAC_BRCA","label_dict":{"IDC":0,"ILC":1},"backbone":"conch",
              "feature_space_id":"hf:MahmoodLab/conch","feature_dim":512,
              "representation":"shared_embedding","magnifications":{"low":"5x","high":"10x"},
              "patch_sizes":{"low":512,"high":256},"fit_on_target":False}
    if any(target.get(k)!=v for k,v in expected.items()) or target.get("status")!="ready":
        raise ValueError("external target contract not ready or incompatible")
    if cfg["label_dict"]!=target["label_dict"] or cfg["feature_space_id"]!=target["feature_space_id"]:
        raise ValueError("source/target scientific identity mismatch")
    if not rows or len({r["slide_id"] for r in rows})!=len(rows):raise ValueError("empty/duplicate target slides")
    labels={};seen_cases=set()
    for r in rows:
        if not r["slide_id"] or not r["case_id"] or r["label"] not in target["label_dict"]:
            raise ValueError("missing identities or incompatible target labels")
        case=r["case_id"];seen_cases.add(case)
        if case in labels and labels[case]!=r["label"]:raise ValueError("conflicting patient labels")
        labels[case]=r["label"]
    if seen_cases & set(source_cases):raise ValueError("source-target patient overlap")
    if set(labels.values())!={"IDC","ILC"}:raise ValueError("external subtype evaluation needs both classes")


def preflight(source, target_path, native_baseline=False):
    source=Path(source);target_path=Path(target_path)
    record=json.loads((source/"metrics.json").read_text())
    if record["status"]!="completed" or record["provenance"]["inference_requires_tme"]:
        raise ValueError("requires a completed WSI-only student")
    for name,value in record["artifact_sha256"].items():
        if digest(source/name)!=value:raise ValueError("source artifact changed")
    cfg=json.loads((source/"config.json").read_text())["resolved_config"]
    target=load_yaml_config(target_path)
    if target.get("status")!="ready":raise ValueError("target is not ready; supply verified histology labels and feature manifest")
    rows=table(target["manifest"])
    source_cases={r["case_id"] for r in table(cfg["dataset_csv"])}
    validate_target(target,rows,source_cases,cfg)
    if digest(target["manifest"])!=target["manifest_sha256"]:raise ValueError("target manifest changed")
    if digest(cfg["backbone_weights"])!=target["runtime_encoder_checkpoint_sha256"]:raise ValueError("different runtime encoder checkpoint")
    if not target.get("label_source") or digest(target["label_source"])!=target["label_source_sha256"]:
        raise ValueError("target label provenance missing/changed")
    audit=json.loads(Path(target["feature_audit"]).read_text())
    if digest(target["feature_audit"])!=target["feature_audit_sha256"] or audit["status"]!="passed" or audit["manifest_sha256"]!=target["manifest_sha256"]:
        raise ValueError("missing/changed HDF5 feature audit")
    feature_stats={}
    for r in rows:
        for key in ("feature_path_column_s","feature_path_column_l"):
            path=Path(os.path.expandvars(r[cfg[key]]))
            stat=path.stat();feature_stats[str(path)]={"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}
            checked=audit["files"].get(str(path),{})
            if any(checked.get(k)!=v for k,v in feature_stats[str(path)].items()):raise ValueError("feature changed since header audit")
    checkpoint=source/f"fold{record['fold']}_inference.pt"
    if native_baseline:
        checkpoint=Path(record["provenance"]["contract"]["base_checkpoint_dir"])/f"fold{record['fold']}_best.pt"
        if digest(checkpoint)!=record["provenance"]["source_sha256"][str(checkpoint.resolve())]:raise ValueError("native reference checkpoint changed")
    files=[source/"metrics.json",source/"config.json",checkpoint,target_path,Path(target["manifest"]),
           Path(target["label_source"]),Path(target["feature_audit"]),Path(cfg["text_prompt_path"]),Path(__file__),
           PGVL/"methods/vila_mil/adapter.py",PGVL/"methods/vila_mil/model.py",
           PGVL/"methods/vila_mil/model_utils.py",PGVL/"common/models/paired_encoder_extension.py",
           PGVL/"methods/mscpt/dataset.py",PGVL/"common/datasets/dataset_generic.py"]
    payload={"source_identity":record["identity"],"fold":record["fold"],"target":target,"native_baseline":native_baseline,
             "source_sha256":{str(p):digest(p) for p in files},"feature_stats":feature_stats,
             "tme_files_accessed":False,"fitting_performed":False}
    identity=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return cfg,rows,checkpoint,payload,identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-result",type=Path,required=True)
    parser.add_argument("--target-contract",type=Path,required=True)
    parser.add_argument("--output",type=Path);parser.add_argument("--execute",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--native-baseline",action="store_true")
    parser.add_argument("--expected-source-identity");parser.add_argument("--expected-code-sha256")
    parser.add_argument("--expected-target-sha256")
    args=parser.parse_args();load_dotenv(PGVL/".env")
    os.environ.setdefault("PATHOTME_DATA_ROOT","/path/to/shared/PathoTME-data")
    if args.expected_code_sha256 and digest(__file__)!=args.expected_code_sha256:raise ValueError("queued evaluator changed")
    if args.expected_target_sha256 and digest(args.target_contract)!=args.expected_target_sha256:raise ValueError("queued target contract changed")
    cfg,rows,checkpoint,payload,identity=preflight(args.source_result,args.target_contract,args.native_baseline)
    if args.expected_source_identity and payload["source_identity"]!=args.expected_source_identity:raise ValueError("wrong source experiment")
    if not args.execute:
        print(json.dumps({"status":"ready","identity":identity,"slides":len(rows),"requires_tme":False}));return
    if args.output is None:parser.error("--output required with --execute")
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/".run.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        marker=args.output/"metrics.json"
        if marker.exists():
            completed=json.loads(marker.read_text())
            if completed.get("identity")!=identity or completed.get("status")!="completed" or any(
                    digest(args.output/n)!=h for n,h in completed["artifact_sha256"].items()):raise ValueError("incompatible external completion")
            print("already_completed");return
        if any(p.name!=".run.lock" for p in args.output.iterdir()):raise ValueError("preserve existing external results")
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader
        from methods.mscpt.dataset import MSCPT_Dataset
        from methods.vila_mil.adapter import ViLaMILMethod
        from common.models.paired_encoder_extension import project_paired_features
        from common.statistical_analysis import aggregate_patients, extended_metrics
        model=ViLaMILMethod(cfg,args.device).build_model()
        saved=torch.load(checkpoint,map_location=args.device,weights_only=True)
        if not args.native_baseline and saved["identity"]!=payload["source_identity"]:raise ValueError("export identity mismatch")
        model.load_state_dict(saved if args.native_baseline else saved["state_dict"],strict=True);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        frame=pd.DataFrame(rows)
        for key in ("feature_path_column_s","feature_path_column_l"):
            frame[cfg[key]]=frame[cfg[key]].map(os.path.expandvars)
        ds=MSCPT_Dataset(frame,"",None,cfg["label_dict"],feature_path_column_s=cfg["feature_path_column_s"],
            feature_path_column_l=cfg["feature_path_column_l"],feature_dim=cfg["feature_dim"],include_metadata=True)
        predictions=[]
        with torch.no_grad():
            for low,high,meta,label in DataLoader(ds,batch_size=1,shuffle=False,num_workers=4):
                low,high=(project_paired_features(model,x.squeeze(0).to(args.device)) for x in (low,high))
                if not torch.isfinite(low).all() or not torch.isfinite(high).all():raise ValueError("nonfinite feature payload")
                # Dummy label: target labels never enter the model, only final scoring.
                prob=model(low,torch.zeros(len(low),2,device=args.device),high,
                    torch.zeros(len(high),2,device=args.device),torch.zeros(1,dtype=torch.long,device=args.device))[0][0].cpu().numpy()
                if not np.isfinite(prob).all():raise ValueError("nonfinite external predictions")
                predictions.append({"slide_id":meta["slide_id"][0],"case_id":meta["case_id"][0],"label":int(label.item()),
                                    "probability_0":float(prob[0]),"probability_1":float(prob[1]),"prediction":int(prob.argmax())})
        frame=pd.DataFrame(predictions);path=args.output/"predictions.csv";frame.to_csv(path,index=False)
        report={"status":"completed","identity":identity,"provenance":payload,
                "slide_metrics":extended_metrics(frame),"patient_metrics":extended_metrics(aggregate_patients(frame)),
                "artifact_sha256":{path.name:digest(path)}}
        (args.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
