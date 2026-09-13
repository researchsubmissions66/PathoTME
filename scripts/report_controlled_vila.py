#!/usr/bin/env python3
"""Report only complete matched cohorts; patient bootstrap never counts folds as patients."""
import argparse
import json
from pathlib import Path

from run_controlled_vila import (environment, load_yaml_config, preflight, output_dir,
                                  verify_completed, ROOT, CONDITIONS, STUDENTS, sha)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default=ROOT/"configs/vila_conch_brca_controlled_16shot.yaml")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--bootstrap",type=int,default=1000)
    args=p.parse_args();environment();contract=load_yaml_config(args.config)
    if args.output.exists():raise FileExistsError("preserve previous reports")
    import numpy as np
    import pandas as pd
    from common.statistical_analysis import aggregate_patients, extended_metrics, paired_bootstrap_differences
    frames={};status=[];provenance={};folds=contract["folds"]
    for condition in CONDITIONS:
        pieces=[]
        for fold in folds:
            _,_,_,identity=preflight(contract,fold,condition)
            root=output_dir(contract,fold,condition)
            record=verify_completed(root,identity)
            status.append({"condition":condition,"fold":fold,"completed":bool(record),"identity":identity})
            if record:
                path=root/f"fold{fold}_predictions.csv";piece=pd.read_csv(path);piece["fold"]=fold;pieces.append(piece)
                provenance[str(path)]=sha(path)
        if len(pieces)==len(folds):
            combined=pd.concat(pieces,ignore_index=True)
            if combined["slide_id"].duplicated().any() or (combined.groupby("case_id")["fold"].nunique()>1).any():
                raise ValueError("OOF patient/slide reuse")
            frames[condition]=combined
    # Reuse existing reference predictions; no new classification runs.
    reference=load_yaml_config(contract["reference_contract"])
    for name,root in (("native",Path(contract["base_checkpoint_dir"])),("tme_only",Path(reference["results_root"])/"tme_only")):
        pieces=[]
        for fold in folds:
            preflight(contract,fold,"student_ce")  # validates both references and exact folds
            path=(root/f"fold{fold}" if name=="tme_only" else root)/f"fold{fold}_predictions.csv"
            piece=pd.read_csv(path);piece["fold"]=fold;pieces.append(piece);provenance[str(path)]=sha(path)
        frames[name]=pd.concat(pieces,ignore_index=True)
    summaries=[{"domain":"TCGA_OOF","unit":unit,"condition":c,"folds":len(folds),**extended_metrics(data)}
               for c,f in frames.items() for unit,data in (("slide",f),("patient",aggregate_patients(f)))]
    comparisons=[("zero","actual"),("shuffled","actual"),("tme_only","actual"),("fusion","actual"),
                 ("student_ce","aux_real"),("aux_shuffled","aux_real"),("student_ce","kd_real"),
                 ("kd_visual","kd_real"),("kd_shuffled","kd_real")]
    boot=[]
    def compare(domain,data,pairs):
        for a,b in pairs:
            if a not in data or b not in data:continue
            left=data[a].set_index("slide_id",verify_integrity=True).sort_index()
            right=data[b].set_index("slide_id",verify_integrity=True).sort_index()
            if not left.index.equals(right.index) or not left[["case_id","label"]].equals(right[["case_id","label"]]):
                raise ValueError("paired slide, patient or class mismatch")
            result=paired_bootstrap_differences(left.reset_index(),right.reset_index(),replicates=args.bootstrap,seed=1)
            result["domain"]=domain;result["reference"]=a;result["candidate"]=b;boot.append(result)
    compare("TCGA_OOF",frames,comparisons)
    external={}
    for condition in ("native",*STUDENTS):
        pieces=[]
        for fold in folds:
            root=Path(contract["results_root"])/"external/cptac_brca"/condition/f"fold{fold}"
            marker=root/"metrics.json"
            if not marker.exists():continue
            from eval_wsi_only import preflight as external_preflight
            source=output_dir(contract,fold,"student_ce" if condition=="native" else condition)
            *_,identity=external_preflight(source,contract["external"]["contract"],condition=="native")
            verify_completed(root,identity)
            path=root/"predictions.csv";piece=pd.read_csv(path).set_index("slide_id",verify_integrity=True).sort_index()
            summaries.append({"domain":"CPTAC_source_fold","unit":"patient","condition":condition,"fold":fold,**extended_metrics(aggregate_patients(piece.reset_index()))})
            pieces.append(piece);provenance[str(path)]=sha(path)
        if len(pieces)==len(folds):
            if any(not f.index.equals(pieces[0].index) or not f[["case_id","label"]].equals(pieces[0][["case_id","label"]]) for f in pieces):
                raise ValueError("external folds have different patients or labels")
            combined=pieces[0].copy()
            columns=["probability_0","probability_1"]
            combined[columns]=np.stack([f[columns].to_numpy() for f in pieces]).mean(0)
            combined["prediction"]=combined[columns].to_numpy().argmax(1)
            external[condition]=combined.reset_index()
            summaries.append({"domain":"CPTAC_equal_weight_source_ensemble","unit":"patient","condition":condition,**extended_metrics(aggregate_patients(external[condition]))})
    compare("CPTAC_equal_weight_source_ensemble",external,[*comparisons,("native","aux_real"),("native","kd_real")])
    args.output.mkdir(parents=True)
    pd.DataFrame(status).to_csv(args.output/"completion.csv",index=False)
    pd.DataFrame(summaries).to_csv(args.output/"metrics.csv",index=False)
    if boot:pd.concat(boot,ignore_index=True).to_csv(args.output/"paired_patient_bootstrap.csv",index=False)
    (args.output/"provenance.json").write_text(json.dumps({"source_sha256":provenance,"reporter_sha256":sha(__file__),
        "bootstrap_replicates":args.bootstrap,"note":"Exploratory multiple comparisons, no multiplicity correction. Intervals conditional on fitted models; not retraining uncertainty. External folds share patients; only predeclared equal-weight ensemble is patient-bootstrapped."},indent=2)+"\n")


if __name__=="__main__":main()
