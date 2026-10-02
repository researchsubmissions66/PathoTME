#!/usr/bin/env python3
"""Dry-run by default; submit exact single-fold experiments behind smoke gates."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from run_controlled_vila import (ROOT, PGVL, CONDITIONS, STUDENTS, environment,
                                  load_yaml_config, preflight, output_dir, verify_completed, sha)


def build_plan(config, with_external=False):
    contract=load_yaml_config(config)
    python=str(Path(os.environ["PGVL_CONDA_ENV"])/"bin/python")
    runner=str(ROOT/"scripts/run_controlled_vila.py")
    env=["env","HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1","HF_HOME=/path/to/shared/.cache_huggingface",
         "PYTHONNOUSERSITE=1","TOKENIZERS_PARALLELISM=false","OMP_NUM_THREADS=6","MKL_NUM_THREADS=6",
         "LD_LIBRARY_PATH="+str(Path(python).parent.parent/"lib")+
         (":"+os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")]
    def component(fold,condition,smoke=False):
        _,_,payload,identity=preflight(contract,fold,condition,smoke)
        output=output_dir(contract,fold,condition,smoke)
        done=bool(verify_completed(output,identity,smoke))
        if not done and output.exists() and any(p.name!=".run.lock" for p in output.iterdir()):
            raise ValueError(f"partial condition preserved; explicit recovery required: {output}")
        return {"fold":fold,"condition":condition,"identity":identity,"output":str(output),"completed":done,"inputs":payload}
    smoke=[component(0,c,True) for c in CONDITIONS]
    suite_id=hashlib.sha256(json.dumps([c["identity"] for c in smoke]).encode()).hexdigest()
    plans=[{"key":"smoke","components":smoke,"depends_on":[],"time":"01:00:00",
            "command":[python,"-u",runner,"--config",str(config),"--fold","0","--smoke-suite","--expected-suite-identity",suite_id]}]
    for fold in contract["folds"]:
        for condition in contract["conditions"]:
            c=component(fold,condition)
            deps=["smoke"]
            if condition in ("kd_real","kd_shuffled"):
                deps.append(f"{contract['student']['teachers'][condition]}-f{fold}")
            plans.append({"key":f"{condition}-f{fold}","components":[c],"depends_on":deps,
                          "time":"01:00:00" if condition in STUDENTS else "00:30:00" if condition=="fusion" else "00:45:00",
                          "command":[python,"-u",runner,"--config",str(config),"--fold",str(fold),
                                     "--condition",condition,"--expected-identity",c["identity"]]})
    logs=Path(os.environ["PATHOTME_RESULTS_ROOT"])/"logs"
    target_path=Path(contract["external"]["contract"]) if with_external else None
    if with_external:
        target=load_yaml_config(target_path)
        if target["status"]!="ready" or sha(target["manifest"])!=target["manifest_sha256"] or sha(target["feature_audit"])!=target["feature_audit_sha256"]:
            raise ValueError("external cohort is not ready or its frozen assets changed")
        audit=json.loads(Path(target["feature_audit"]).read_text())
        for path,stat in audit["files"].items():
            actual=Path(path).stat()
            if actual.st_size!=stat["size"] or actual.st_mtime_ns!=stat["mtime_ns"]:raise ValueError("external features changed since audit")
    for p in plans:
        name="ptme-cbr-v1-"+p["key"]
        p.update(job_name=name,skip_completed=all(c["completed"] for c in p["components"]))
        p["external_components"]=[]
        if with_external and p["key"]!="smoke" and p["components"][0]["condition"] in STUDENTS:
            c=p["components"][0]
            for native in ([False,True] if c["condition"]=="student_ce" else [False]):
                output=Path(contract["results_root"])/"external/cptac_brca"/("native" if native else c["condition"])/f"fold{c['fold']}"
                command=[python,"-u",str(ROOT/"scripts/eval_wsi_only.py"),"--source-result",c["output"],
                    "--target-contract",str(target_path),"--output",str(output),"--execute",
                    "--expected-source-identity",c["identity"],"--expected-code-sha256",sha(ROOT/"scripts/eval_wsi_only.py"),
                    "--expected-target-sha256",sha(target_path)]
                if native:command.append("--native-baseline")
                p["external_components"].append({"condition":"native" if native else c["condition"],
                    "fold":c["fold"],"output":str(output),"command":command,"target_contract":target,
                    "source_identity":c["identity"]})
            # Do not silently skip requested external work when source is complete.
            if p["skip_completed"]:
                from eval_wsi_only import preflight as external_preflight
                for e in p["external_components"]:
                    *_,external_identity=external_preflight(c["output"],target_path,e["condition"]=="native")
                    if not verify_completed(e["output"],external_identity):
                        raise ValueError("only external inference remains; use an explicit evaluation-only launch")
        wrap="set -eu\n"+"\n".join(shlex.join(env+command) for command in [p["command"],*[e["command"] for e in p["external_components"]]])
        p["sbatch"]=["sbatch","--parsable","--account=shared-delta-gpu","--partition=gpuA100x4",
            "--nodes=1","--ntasks=1","--gres=gpu:1","--cpus-per-task=6","--mem=24G",f"--time={p['time']}",
            f"--job-name={name}",f"--chdir={PGVL}",f"--output={logs}/{name}-%j.out",
            "--comment=pathotme-conch-brca-controlled-v1","--wrap",wrap]
    return plans


def persist(path,ledger):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(ledger,indent=2)+"\n");os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=ROOT/"configs/vila_conch_brca_controlled_16shot.yaml")
    p.add_argument("--report",type=Path,required=True);p.add_argument("--submit",action="store_true")
    p.add_argument("--with-external",action="store_true")
    args=p.parse_args();environment()
    if args.report.exists(): raise FileExistsError("preserve the existing ledger")
    with (ROOT/".controlled_launch.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plans=build_plan(args.config,args.with_external)
        if args.submit:
            names=subprocess.check_output(["squeue","-u",os.environ["USER"],"-h","-o","%j"],text=True).splitlines()
            if any(p["job_name"] in names for p in plans): raise ValueError("campaign jobs already active; do not duplicate")
        ledger={"schema":"pathotme.controlled_vila.v1","created_at":datetime.datetime.now().astimezone().isoformat(),
                "dry_run":not args.submit,"launcher_sha256":sha(__file__),"plans":plans,
                "external_evaluation":"35 CPTAC evaluations follow WSI-only source folds in their existing allocations" if args.with_external else "not requested"}
        with args.report.open("x") as handle: json.dump(ledger,handle,indent=2)
        submitted={}; completed=set()
        for row in plans:
            if row["skip_completed"]:
                row["status"]="skipped_completed";completed.add(row["key"])
            elif args.submit:
                dependencies=[]
                for key in row["depends_on"]:
                    if key in submitted: dependencies.append(submitted[key])
                    elif key not in completed: raise ValueError(f"dependency not resolved: {key}")
                command=row["sbatch"].copy()
                if dependencies: command.insert(1,"--dependency=afterok:"+":".join(dependencies))
                row["submitted_command"]=command;row["status"]="submitting";persist(args.report,ledger)
                result=subprocess.run(command,capture_output=True,text=True)
                row.update(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
                if result.returncode:
                    row["status"]="submission_failed";persist(args.report,ledger);raise RuntimeError(result.stderr)
                job=result.stdout.strip().split(";")[0]
                if not job.isdigit():
                    persist(args.report,ledger);raise RuntimeError("ambiguous submission; inspect queue and ledger")
                row.update(job_id=job,status="submitted");submitted[row["key"]]=job
            else: row["status"]="dry_run_ready"
            persist(args.report,ledger)
            print(json.dumps({k:row[k] for k in ("key","status","time","depends_on","job_id") if k in row}),flush=True)


if __name__=="__main__": main()
