#!/usr/bin/env python3
"""CPU-only TME classifier control with the same BRCA splits and selection unit."""
import argparse
import fcntl
import json
import os
from pathlib import Path

from run_brca_vila import PGVL, load_dotenv, load_yaml_config, preflight, read_csv, sha
from pathotme.brca_features import transform_rows


def fit_classifier(values, labels, val_values, val_labels, grid, seed):
    """Fit all preprocessing on train only; select C by validation error."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    best=None;best_error=float("inf");selected=None
    for c in grid:
        model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
            LogisticRegression(C=float(c),max_iter=2000,solver="lbfgs",random_state=seed))
        model.fit(values,labels)
        error=float((model.predict(val_values)!=val_labels).mean())
        if error<best_error:
            best=model;best_error=error;selected=float(c)
    return best,selected,best_error


def execute(contract,cfg,phases,payload,identity,output,fold):
    import csv
    import joblib
    import numpy as np
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
    _,rows=read_csv(contract["selected_features_csv"])
    lookup=dict(zip((r["slide_id"] for r in rows),transform_rows(rows,contract["panel"])))
    arrays={p:np.asarray([lookup[r["slide_id"]] for r in split]) for p,split in phases.items()}
    labels={p:np.asarray([int(r["label_id"]) for r in split]) for p,split in phases.items()}
    model,c,error=fit_classifier(arrays["train"],labels["train"],arrays["val"],labels["val"],
                                 contract["tme_only"]["c_grid"],contract["seed"]+fold)
    def metrics(y,probs):
        prediction=probs.argmax(-1)
        return {"accuracy":float(accuracy_score(y,prediction)),
                "balanced_accuracy":float(balanced_accuracy_score(y,prediction)),
                "macro_f1":float(f1_score(y,prediction,average="macro")),
                "auroc_ovr":float(roc_auc_score(y,probs[:,1]))}
    probs=model.predict_proba(arrays["test"])
    predictions=output/f"fold{fold}_predictions.csv"
    patient_rows={}
    with predictions.open("x") as h:
        writer=csv.DictWriter(h,fieldnames=["slide_id","case_id","label","prediction","probability_0","probability_1"])
        writer.writeheader()
        for row,y,prob in zip(phases["test"],labels["test"],probs):
            writer.writerow({"slide_id":row["slide_id"],"case_id":row["case_id"],"label":int(y),
                             "prediction":int(prob.argmax()),"probability_0":prob[0],"probability_1":prob[1]})
            patient_rows.setdefault(row["case_id"],[]).append((y,prob))
    py=[];pp=[]
    for records in patient_rows.values():
        if len({int(y) for y,_ in records})!=1:raise ValueError("conflicting patient labels")
        py.append(records[0][0]);pp.append(np.stack([p for _,p in records]).mean(0))
    checkpoint=output/f"fold{fold}_classifier.joblib"
    joblib.dump({"pipeline":model,"identity":identity,"panel_spec":payload["panel_spec"]},checkpoint)
    config=output/"config.json";config.write_text(json.dumps({**payload,"identity":identity},indent=2)+"\n")
    result={"status":"completed","identity":identity,"panel":contract["panel"],"fold":fold,
        "condition":"tme_only","selected_c":c,"validation_error":error,"provenance":payload,
        "slurm_job_id":os.environ.get("SLURM_JOB_ID"),
        "tme_only":{"slide_metrics":metrics(labels["test"],probs),"patient_metrics":metrics(np.asarray(py),np.asarray(pp))},
        "artifact_sha256":{p.name:sha(p) for p in (predictions,checkpoint,config)}}
    temporary=output/"metrics.json.tmp";temporary.write_text(json.dumps(result,indent=2)+"\n")
    os.replace(temporary,output/"metrics.json")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config",required=True)
    parser.add_argument("--fold",type=int,required=True)
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--expected-identity")
    args=parser.parse_args(argv)
    load_dotenv(PGVL/".env")
    os.environ.setdefault("PGVL_REPO_ROOT",str(PGVL))
    os.environ.setdefault("PATHOTME_DATA_ROOT","/path/to/shared/PathoTME-data")
    os.environ.setdefault("PATHOTME_RESULTS_ROOT","/path/to/shared/PathoTME-results")
    contract=load_yaml_config(args.experiment_config)
    cfg,phases,payload,identity=preflight(contract,args.fold,"actual","tme_only")
    if args.expected_identity and identity!=args.expected_identity:raise ValueError("launch identity changed")
    output=Path(contract["results_root"])/"tme_only"/f"fold{args.fold}"
    if args.check_only:
        print(json.dumps({"status":"ready","condition":"tme_only","identity":identity,"results_dir":str(output)}));return 0
    output.mkdir(parents=True,exist_ok=True)
    with (output/".run.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        metrics=output/"metrics.json"
        if metrics.exists():
            saved=json.loads(metrics.read_text())
            if saved.get("identity")!=identity or saved.get("status")!="completed" or any(
                    sha(output/k)!=v for k,v in saved["artifact_sha256"].items()):
                raise ValueError("incompatible or changed result state")
            print("already_completed");return 0
        if any(p.name!=".run.lock" for p in output.iterdir()):raise ValueError("partial output preserved")
        execute(contract,cfg,phases,payload,identity,output,args.fold)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
