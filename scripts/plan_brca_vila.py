#!/usr/bin/env python3
"""Read-only BRCA campaign planner; never submits jobs or overwrites a report."""
import argparse
import json
import os
from pathlib import Path

from run_brca_vila import ROOT, PGVL, load_dotenv, load_yaml_config, preflight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    load_dotenv(PGVL/".env")
    os.environ.setdefault("PGVL_REPO_ROOT",str(PGVL))
    os.environ.setdefault("PATHOTME_DATA_ROOT","/path/to/shared/PathoTME-data")
    os.environ.setdefault("PATHOTME_RESULTS_ROOT","/path/to/shared/PathoTME-results")
    records=[]
    for config in sorted((ROOT/"configs").glob("vila_mil_brca_*_16shot.yaml")):
        contract=load_yaml_config(config)
        for fold in contract["exact_coverage_folds"]:
            for condition in ("actual","zero","tme_only"):
                execution="tme_only" if condition=="tme_only" else "train"
                mode="zero" if condition=="zero" else "actual"
                try:
                    _,phases,payload,identity=preflight(contract,fold,mode,execution)
                    runner="run_brca_tme_only.py" if execution=="tme_only" else "run_brca_vila.py"
                    command=[str(ROOT/"scripts"/runner),"--experiment-config",str(config),"--fold",str(fold)]
                    if execution!="tme_only":command += ["--tme-mode",mode]
                    command += ["--expected-identity",identity]
                    record={"status":"ready","panel":contract["panel"],"condition":condition,"fold":fold,
                            "identity":identity,"config":str(config),"command_arguments":command,
                            "split_counts":{p:len(r) for p,r in phases.items()},
                            "results_dir":str(Path(contract["results_root"])/condition/f"fold{fold}"),
                            "recommended_resources":{"gpus":0 if execution=="tme_only" else 1,
                                "cpus":2 if execution=="tme_only" else 6,
                                "memory":"8G" if execution=="tme_only" else "24G",
                                "time":"00:20:00" if execution=="tme_only" else "00:45:00"}}
                except Exception as error:
                    record={"status":"blocked","config":str(config),"fold":fold,"condition":condition,"error":str(error)}
                records.append(record)
                print(json.dumps(record),flush=True)
    result={"schema":"pathotme.brca_plan.v1","dry_run":True,"submitted_jobs":[],"runs":records,
            "gpu_smoke_required_before_campaign":True}
    if args.report:
        with args.report.open("x") as h:json.dump(result,h,indent=2);h.write("\n")
    return int(any(r["status"]!="ready" for r in records))


if __name__=="__main__":
    raise SystemExit(main())
