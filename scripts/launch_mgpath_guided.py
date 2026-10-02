#!/usr/bin/env python3
"""Dry-run or submit isolated MGPATH/TME folds with immutable launch identities."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from run_mgpath_guided import ROOT, PGVL, preflight
from common.configuration import load_dotenv, load_yaml_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dependency", help="successful smoke job ID")
    parser.add_argument("--partition", default="gpuA100x4")
    parser.add_argument("--time", default="00:45:00")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    load_dotenv(PGVL / ".env")
    os.environ.setdefault("PGVL_REPO_ROOT", str(PGVL))
    os.environ.setdefault("PATHOTME_DATA_ROOT", "/path/to/shared/PathoTME-data")
    os.environ.setdefault("PATHOTME_RESULTS_ROOT", "/path/to/shared/PathoTME-results")
    if args.dependency and not args.dependency.isdigit():
        parser.error("dependency must be a numeric job ID")
    if args.submit and args.report is None:
        parser.error("submission requires a new --report path")
    if args.report and args.report.exists():
        parser.error("report already exists; preserve the existing launch ledger")
    config = ROOT / "configs/mgpath_tme_guided_nsclc_16shot.yaml"
    contract = load_yaml_config(config)
    python = str(Path(os.environ["PGVL_CONDA_ENV"]) / "bin/python")
    selection = [(0, "actual")] if args.smoke_only else [
        (f, m) for f in contract["exact_coverage_folds"] for m in ("actual", "zero")]
    logs = Path(os.environ["PATHOTME_RESULTS_ROOT"]) / "logs"
    plans = []
    for fold, mode in selection:
        _, phases, payload, identity = preflight(
            contract, fold, mode, "smoke" if args.smoke_only else "train")
        command = [python, "-u", str(ROOT / "scripts/run_mgpath_guided.py"),
                   "--experiment-config", str(config), "--fold", str(fold),
                   "--tme-mode", mode, "--expected-identity", identity]
        if args.smoke_only: command.append("--smoke-only")
        env = ["env", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
               "HF_HOME=/path/to/shared/.cache_huggingface", "PYTHONNOUSERSITE=1",
               "OMP_NUM_THREADS=6", "MKL_NUM_THREADS=6",
               "LD_LIBRARY_PATH=" + str(Path(python).parent.parent / "lib")
               + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")]
        tag = "smoke" if args.smoke_only else mode
        submit = ["sbatch", "--parsable", "--account=shared-delta-gpu",
                  f"--partition={args.partition}", "--gres=gpu:1", "--cpus-per-task=6",
                  "--mem=24G", f"--time={args.time}", f"--job-name=ptme-mgp-{tag}-f{fold}",
                  f"--chdir={PGVL}", f"--output={logs}/pathotme-mgpath-{tag}-f{fold}-%j.out"]
        if args.dependency: submit.append(f"--dependency=afterok:{args.dependency}")
        submit += ["--wrap", shlex.join(env + command)]
        plans.append({"fold": fold, "tme_mode": mode, "identity": identity,
                      "inputs": payload, "command": submit,
                      "split_counts": {k: len(v) for k, v in phases.items()}})
    if not args.submit:
        for plan in plans:
            print(json.dumps({k: v for k, v in plan.items() if k != "inputs"}))
        return
    logs.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Create the plan before the first external action. Persist every returned
    # ID immediately, preserving even a partially submitted campaign.
    ledger = {"created_at": datetime.now(timezone.utc).isoformat(), "plans": plans}
    with args.report.open("x") as handle: json.dump(ledger, handle, indent=2)
    for plan in plans:
        result = subprocess.run(plan["command"], capture_output=True, text=True)
        plan["returncode"] = result.returncode
        plan["stdout"] = result.stdout.strip(); plan["stderr"] = result.stderr.strip()
        if result.returncode == 0:
            plan["job_id"] = result.stdout.strip().split(";")[0]
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2) + "\n")
        os.replace(temporary, args.report)
        print(json.dumps({k: v for k, v in plan.items() if k != "inputs"}), flush=True)
        if result.returncode: raise RuntimeError("submission failed; inspect the preserved ledger")


if __name__ == "__main__":
    main()
