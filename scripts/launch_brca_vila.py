#!/usr/bin/env python3
"""Plan or submit BRCA smokes and matched folds, preserving per-condition identities.

CPU-only comparators share their matching actual-adapter allocation because
this cluster user has no CPU account. They do not request another GPU.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess

from run_brca_vila import ROOT, PGVL, load_dotenv, load_yaml_config, preflight, sha


def complete(output, identity, smoke=False):
    """Skip only identity- and artifact-valid completed results."""
    output=Path(output)
    marker=output/("smoke_report.json" if smoke else "metrics.json")
    if marker.exists():
        data=json.loads(marker.read_text())
        if data.get("identity")!=identity or data.get("status")!=("smoke_passed" if smoke else "completed"):
            raise ValueError(f"incompatible completed result: {marker}")
        if any(sha(output/p)!=h for p,h in data.get("artifact_sha256",{}).items()):
            raise ValueError(f"changed result artifacts: {output}")
        return True
    if output.exists() and any(p.name!=".run.lock" for p in output.iterdir()):
        raise ValueError(f"partial result requires explicit recovery: {output}")
    return False


def build_plan():
    """Refresh every native/panel/fold contract before any submission."""
    python=str(Path(os.environ["PGVL_CONDA_ENV"])/"bin/python")
    logs=Path(os.environ["PATHOTME_RESULTS_ROOT"])/"logs"
    environment=["env","HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1",
        "HF_HOME=/path/to/huggingface-cache","PYTHONNOUSERSITE=1",
        "TOKENIZERS_PARALLELISM=false","OMP_NUM_THREADS=6","MKL_NUM_THREADS=6",
        "LD_LIBRARY_PATH="+str(Path(python).parent.parent/"lib")+
        (":"+os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")]
    plans=[]
    for panel,short in [("brca_morph64_v1","m64"),("brca_shared62_v1","s62")]:
        config=ROOT/f"configs/vila_mil_{panel}_16shot.yaml"
        contract=load_yaml_config(config)
        selections=[(0,"actual","smoke")]+[(f,m,"train") for f in contract["exact_coverage_folds"] for m in ("actual","zero")]
        for fold,mode,execution in selections:
            _,_,payload,identity=preflight(contract,fold,mode,execution)
            root=Path(contract["results_root"])
            output=(root/"smoke" if execution=="smoke" else root)/mode/f"fold{fold}"
            command=[python,"-u",str(ROOT/"scripts/run_brca_vila.py"),"--experiment-config",str(config),
                     "--fold",str(fold),"--tme-mode",mode,"--expected-identity",identity]
            if execution=="smoke":command.append("--smoke-only")
            components=[{"condition":execution if execution=="smoke" else mode,"identity":identity,
                         "output":str(output),"inputs":payload,"command":command,
                         "completed":complete(output,identity,execution=="smoke")}]
            if execution=="train" and mode=="actual":
                _,_,cpu_payload,cpu_id=preflight(contract,fold,"actual","tme_only")
                cpu_output=root/"tme_only"/f"fold{fold}"
                cpu_command=[python,"-u",str(ROOT/"scripts/run_brca_tme_only.py"),"--experiment-config",str(config),
                             "--fold",str(fold),"--expected-identity",cpu_id]
                components.append({"condition":"tme_only","identity":cpu_id,"output":str(cpu_output),
                                   "inputs":cpu_payload,"command":cpu_command,"completed":complete(cpu_output,cpu_id)})
                if components[0]["completed"] and not components[1]["completed"]:
                    raise RuntimeError("only a CPU comparator remains; do not reserve a new GPU just for it")
            name=f"ptme-brca-{short}-smoke" if execution=="smoke" else f"ptme-brca-{short}-{mode[0]}-f{fold}"
            wrap="set -eu\n"+"\n".join(shlex.join(environment+c["command"]) for c in components if not c["completed"])
            submit=["sbatch","--parsable","--account=YOUR_SLURM_ACCOUNT","--partition=gpuA100x4",
                    "--nodes=1","--ntasks=1","--gres=gpu:1","--cpus-per-task=6","--mem=24G",
                    "--time=00:45:00",f"--job-name={name}",f"--chdir={PGVL}",
                    f"--output={logs}/{name}-%j.out","--comment=pathotme-brca-v1-20260907","--wrap",wrap]
            plans.append({"panel":panel,"fold":fold,"mode":mode,"execution":execution,
                          "job_name":name,"components":components,"command_template":submit,
                          "dependency_panel":None if execution=="smoke" else panel,
                          "skip_completed":all(c["completed"] for c in components)})
    return plans


def persist(path, ledger):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(ledger,indent=2)+"\n");os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit",action="store_true")
    parser.add_argument("--report",required=True,type=Path)
    args=parser.parse_args()
    load_dotenv(PGVL/".env")
    os.environ.setdefault("PGVL_REPO_ROOT",str(PGVL))
    os.environ.setdefault("PATHOTME_DATA_ROOT","/path/to/shared/PathoTME-data")
    os.environ.setdefault("PATHOTME_RESULTS_ROOT","/path/to/shared/PathoTME-results")
    if args.report.exists():raise FileExistsError("preserve existing launch ledger")
    with (ROOT/".brca_launch.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plans=build_plan()
        if args.submit:
            queued=subprocess.check_output(["squeue","-u",os.environ["USER"],"-h","-o","%j"],text=True).splitlines()
            if any(p["job_name"] in queued for p in plans):
                raise RuntimeError("BRCA jobs already queued; inspect their ledger rather than duplicating")
        ledger={"schema":"pathotme.brca_launch.v1","created_at":datetime.datetime.now().astimezone().isoformat(),
                "dry_run":not args.submit,"launcher_sha256":sha(__file__),"plans":plans,
                "cpu_policy":"TME-only conditions follow matching real-TME adapter in same allocation; no CPU account available"}
        with args.report.open("x") as handle:json.dump(ledger,handle,indent=2)
        smoke_jobs={}
        for p in plans:
            if p["skip_completed"]:
                p["status"]="skipped_completed"
            elif args.submit:
                command=list(p["command_template"])
                if p["dependency_panel"] and p["dependency_panel"] in smoke_jobs:
                    command.insert(1,"--dependency=afterok:"+smoke_jobs[p["dependency_panel"]])
                p["submitted_command"]=command
                persist(args.report,ledger)  # record exact request before external action
                result=subprocess.run(command,capture_output=True,text=True)
                p.update(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
                if result.returncode==0:
                    job=result.stdout.strip().split(";")[0]
                    if not job.isdigit():raise RuntimeError("unrecognized sbatch response; inspect ledger")
                    p.update(job_id=job,status="submitted")
                    if p["execution"]=="smoke":smoke_jobs[p["panel"]]=job
                else:p["status"]="submission_failed"
                persist(args.report,ledger)
                if result.returncode:raise RuntimeError("submission failed; preserve partial ledger")
            else:p["status"]="dry_run_ready"
            print(json.dumps({k:v for k,v in p.items() if k not in ("components","command_template","submitted_command")}),flush=True)
        persist(args.report,ledger)


if __name__=="__main__":
    main()
