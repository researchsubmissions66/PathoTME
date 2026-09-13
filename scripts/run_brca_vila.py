#!/usr/bin/env python3
"""Fingerprint and run one exact BRCA fold; check-only never imports Torch.

New BRCA outputs cannot resume or overwrite NSCLC or native ViLa results.
Model construction checks and smoke use training bags only.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PGVL = Path(os.environ.get("PGVL_REPO_ROOT", "/path/to/PGVL-Gym"))
sys.path[:0] = [str(ROOT), str(PGVL)]
from common.configuration import load_dotenv, load_yaml_config
from common.run_state import validate_resume_state
from pathotme.brca_features import digest, panel_spec, transform_rows

_HASH_CACHE = {}


def sha(path):
    """Cache unchanged file hashes within this read-only planning process."""
    path = Path(path).resolve(); stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = digest(path)
    return _HASH_CACHE[key]


def read_csv(path):
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def preflight(contract, fold, mode, execution="train"):
    """Validate the exact BRCA baseline, panel, prompts, patients and run identity."""
    if fold not in contract["exact_coverage_folds"] or mode not in ("actual", "zero"):
        raise ValueError("unregistered fold or mode")
    if execution not in ("train", "smoke", "tme_only") or (execution == "tme_only" and mode != "actual"):
        raise ValueError("invalid execution condition")
    panel = panel_spec(contract["panel"]); width = len(panel["feature_names"])
    arch, training = contract["model"], contract["training"]
    if (arch["raw_feature_count"] != width or arch["tokenization"] != contract["panel"]
            or training["optimizer"] != "adam" or training["checkpoint_monitor"] != "val_error"
            or training["epochs"] < 1 or contract["shots"] != 16):
        raise ValueError("unsupported BRCA experiment contract")
    if (contract["tme_only"]["model"] != "logistic_regression"
            or contract["tme_only"]["checkpoint_monitor"] != "val_error"
            or not contract["tme_only"]["c_grid"]
            or any(not math.isfinite(float(c)) or float(c) <= 0 for c in contract["tme_only"]["c_grid"])):
        raise ValueError("invalid TME-only comparator contract")
    cfg = load_yaml_config(contract["base_config"])
    if (cfg["method"] != "vila_mil" or cfg["task"] != "brca" or cfg["backbone"] != "clip-rn50"
            or cfg["feature_space_id"] != "openai/clip-rn50@official"
            or cfg["feature_resolutions"] != {"low": "5x", "high": "10x"}
            or cfg["label_dict"] != {"IDC": 0, "ILC": 1}
            or cfg["classnames"] != ["invasive ductal carcinoma", "invasive lobular carcinoma"]
            or cfg["vila_prompt_file_classnames"] != ["IDC", "ILC"]
            or cfg["shots"] != 16 or cfg.get("encoder_extension")):
        raise ValueError("requires registered native-encoder BRCA IDC/ILC ViLa 16-shot")
    base = Path(contract["base_checkpoint_dir"])
    if base.resolve() != Path(cfg["results_dir"]).resolve():
        raise ValueError("baseline directory mismatch")
    state = json.loads((base / "metrics.json").read_text())
    valid = validate_resume_state(state, base / "config.json", "vila_mil", cfg)
    record = next((r for r in valid if r["fold"] == fold), None)
    if record is None or any(record.get("sample_failures", {}).values()):
        raise ValueError("baseline fold missing or has sample failures")
    features = Path(contract["selected_features_csv"])
    metadata_path = features.with_suffix(features.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    if (metadata["panel"] != contract["panel"] or metadata["cohort"] != "TCGA-BRCA"
            or metadata["revision"] != panel["revision"]
            or metadata["schema_sha256"] != panel["schema_sha256"]
            or metadata["selected_table_sha256"] != sha(features)
            or metadata["builder_sha256"] != sha(ROOT / "pathotme/brca_features.py")):
        raise ValueError("feature table provenance changed")
    header, rows = read_csv(features)
    if header != ["slide_id", *panel["feature_names"]]:
        raise ValueError("feature order changed")
    values = transform_rows(rows, contract["panel"])
    ids = [r["slide_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate feature rows")
    lookup = dict(zip(ids, values))
    phases = {}; paths = []; patients = {}
    for phase in ("train", "val", "test"):
        path = Path(cfg["split_dir"]) / f"fold{fold}" / f"{phase}.csv"
        _, split = read_csv(path); phases[phase] = split; paths.append(path)
        slides = [r["slide_id"] for r in split]
        if len(slides) != len(set(slides)) or set(slides) - set(ids):
            raise ValueError(f"{phase}: duplicate slides or missing OpenTME rows")
        if set(int(r["label_id"]) for r in split) != {0, 1}:
            raise ValueError(f"{phase}: invalid classes")
        if phase != "test" and any(sum(int(r["label_id"]) == c for r in split) != 16 for c in (0, 1)):
            raise ValueError("train/validation must have 16 slides per class")
        patients[phase] = {r["case_id"] for r in split}
        if any(not Path(os.path.expandvars(r[cfg[k]])).is_file() for r in split
               for k in ("feature_path_column_s", "feature_path_column_l")):
            raise ValueError(f"{phase}: missing native visual feature")
    if any(patients[a] & patients[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("patient leakage")
    all_missing = [panel["feature_names"][i] for i in range(width)
                   if all(math.isnan(lookup[r["slide_id"]][i]) for r in phases["train"])]
    if all_missing:
        raise ValueError(f"training-only imputation impossible: {all_missing}")
    predictions = base / f"fold{fold}_predictions.csv"
    _, saved = read_csv(predictions)
    expected = {r["slide_id"]: (r["case_id"], int(r["label_id"])) for r in phases["test"]}
    observed = {r["slide_id"]: (r["case_id"], int(r["label"])) for r in saved}
    if len(observed) != len(saved) or observed != expected:
        raise ValueError("baseline predictions differ from exact holdout")
    if sha(cfg["text_prompt_path"]) != cfg["vila_prompt_file_sha256"]:
        raise ValueError("BRCA prompt bank changed")
    files = [Path(contract["base_config"]), base / "config.json", base / "metrics.json",
             base / f"fold{fold}_best.pt", predictions, features, metadata_path,
             Path(cfg["text_prompt_path"]), Path(cfg["dataset_csv"]), *paths,
             Path(__file__), *(ROOT / p for p in (
                 "pathotme/brca_features.py", "pathotme/brca_vila.py", "pathotme/brca_vila_adapter.py",
                 "pathotme/guided_vila.py", "pathotme/guided_vila_adapter.py", "pathotme/features.py",
                 "scripts/run_vila_guided.py", "scripts/run_brca_tme_only.py")), *(PGVL / p for p in (
                 "methods/vila_mil/model.py", "methods/vila_mil/model_utils.py",
                 "methods/vila_mil/adapter.py", "methods/base.py", "train.py",
                 "common/configuration.py", "common/run_state.py"))]
    payload = {"contract": contract, "panel_spec": panel, "fold": fold, "tme_mode": mode,
               "execution_mode": execution, "source_sha256": {str(p.resolve()): sha(p) for p in files}}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cfg, phases, payload, identity


def evaluate(loader, method, model, metric_fn):
    """Export panel-owned token attention with the correct dynamic role names."""
    import numpy as np
    import torch
    from run_vila_guided import _metric_bundle
    spec = panel_spec(method.panel)
    probabilities=[]; labels=[]; metadata=[]; attention=[]
    model.eval()
    with torch.no_grad():
        for batch in loader:
            details=method.eval_step_with_details(batch, model)
            probabilities.append(details["probabilities"][0].cpu().numpy())
            labels.append(int(batch[-1].reshape(-1)[0])); metadata.append(details["metadata"])
            row={}
            for scale in ("low", "high"):
                scores=details[f"{scale}_tme_group_attention"].mean(dim=(0,1)).cpu().tolist()
                names=spec[f"{scale}_tokens"]
                if len(scores)!=len(names): raise ValueError("attention role mismatch")
                row.update({f"tme_attention_{scale}_{name}":v for name,v in zip(names,scores)})
            attention.append(row)
    probabilities=np.asarray(probabilities);labels=np.asarray(labels)
    return probabilities, labels, metadata, attention, _metric_bundle(probabilities,labels,metadata,metric_fn)


def execute(args, cfg, phases, contract, payload, identity, output):
    """Train only the panel adapter and select checkpoints on validation error."""
    import numpy as np
    import pandas as pd
    import torch
    from train import build_loaders, classification_metrics, set_seed
    from run_vila_guided import _atomic_json, _atomic_csv, _atomic_torch, _run_epoch, _metric_bundle
    from pathotme.brca_vila_adapter import BreastGuidedViLaMethod
    training=contract["training"]; arch=contract["model"]
    cfg={**cfg,"_fold_index":args.fold,"results_dir":str(output),
         "base_checkpoint_dir":contract["base_checkpoint_dir"],"tme_feature_csv":contract["selected_features_csv"],
         "tme_panel":contract["panel"],"tme_mode":args.tme_mode,
         "tme_hidden_dim":arch["hidden_dim"],"tme_attention_heads":arch["attention_heads"],
         "tme_dropout":arch["dropout"],"tme_initial_gate":arch["initial_gate"],
         "lr":training["lr"],"weight_decay":training["weight_decay"]}
    _atomic_json(output/"config.json",{**payload,"identity":identity,"resolved_config":cfg,
                                      "slurm_job_id":os.environ.get("SLURM_JOB_ID")})
    seed=contract["seed"]+args.fold;set_seed(seed)
    loaders=build_loaders("vila_mil",cfg,args.fold)
    method=BreastGuidedViLaMethod(cfg,device=args.device)
    model=method.build_model();method.prepare_fold(args.fold,model,loaders[0])
    trainable=[(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith("conditioner.") for n,_ in trainable):
        raise RuntimeError("unexpected trainable parameters")
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    batch=next(iter(loaders[0]));inputs,_=method._inputs(batch,model);model.eval()
    with torch.no_grad():
        native=model.base(*inputs[:5])[0]
        torch.testing.assert_close(model(*inputs,bypass_conditioner=True)[0],native)
        projection=model.conditioner.output_projection.weight.clone()
        model.conditioner.output_projection.weight.zero_()
        torch.testing.assert_close(model(*inputs)[0],native,rtol=1e-5,atol=1e-5)
        model.conditioner.output_projection.weight.copy_(projection)
    torch.set_rng_state(rng)
    if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    optimizer=method.build_optimizer(model)
    if args.smoke_only:
        model.train();step=method.train_step(batch,model,optimizer,None)
        gradients=[p.grad for p in model.conditioner.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum()>0 for g in gradients):
            raise RuntimeError("invalid adapter gradients")
        if any(p.grad is not None for p in model.base.parameters()):raise RuntimeError("native gradient leak")
        _atomic_json(output/"smoke_report.json",{"status":"smoke_passed","identity":identity,
            "native_bypass_equivalent":True,"zero_residual_equivalent":True,
            "adapter_only_gradients":True,"trainable_parameters":sum(n for _,n in trainable),"loss":step["loss"]})
        return
    best=float("inf");best_epoch=None;stale=0;history=[]
    best_path=output/f"fold{args.fold}_best_adapter.pt";final_path=output/f"fold{args.fold}_final_adapter.pt"
    def checkpoint(epoch):
        return {"format":"pathotme_brca_vila_v1","identity":identity,"panel":contract["panel"],
                "epoch":epoch,"state_dict":model.adapter_state_dict()}
    for epoch in range(training["epochs"]):
        tr=_run_epoch(loaders[0],method,model,classification_metrics,optimizer)
        with torch.no_grad():val=_run_epoch(loaders[1],method,model,classification_metrics)
        monitor=1-float(val["metrics"]["accuracy"]);improved=monitor<best
        if improved:
            best=monitor;best_epoch=epoch;stale=0;_atomic_torch(best_path,checkpoint(epoch))
        else:stale+=1
        history.append({"epoch":epoch,"train_loss":tr["loss"],"val_loss":val["loss"],
                        "val_error":monitor,"checkpoint_improved":improved})
        _atomic_csv(output/"training_history.csv",pd.DataFrame(history))
        print(f"epoch={epoch} train_loss={tr['loss']:.4f} val_error={monitor:.4f}",flush=True)
        if stale>=training["early_stopping_patience"] and epoch>training["early_stopping_min_epoch"]:break
    if best_epoch is None:raise RuntimeError("no finite validation checkpoint")
    _atomic_torch(final_path,checkpoint(epoch))
    model.load_adapter_state_dict(torch.load(best_path,map_location=args.device,weights_only=True)["state_dict"])
    probs,labels,metadata,attention,guided=evaluate(loaders[2],method,model,classification_metrics)
    frame=pd.DataFrame(metadata);frame["label"]=labels;frame["prediction"]=probs.argmax(-1)
    frame[["probability_0","probability_1"]]=probs
    frame=pd.concat([frame,pd.DataFrame(attention)],axis=1)
    native=pd.read_csv(Path(contract["base_checkpoint_dir"])/f"fold{args.fold}_predictions.csv")
    native=native.set_index("slide_id",verify_integrity=True).loc[frame["slide_id"]]
    if not np.array_equal(native["label"].to_numpy(),labels) or not np.array_equal(native["case_id"].to_numpy(),frame["case_id"].to_numpy()):
        raise ValueError("native/guided identities differ")
    native_probs=native[["probability_0","probability_1"]].to_numpy()
    frame[["native_vila_probability_0","native_vila_probability_1"]]=native_probs
    native_metrics=_metric_bundle(native_probs,labels,metadata,classification_metrics)
    prediction_path=output/f"fold{args.fold}_predictions.csv";_atomic_csv(prediction_path,frame)
    files=[best_path,final_path,prediction_path,output/"training_history.csv",output/"config.json"]
    report={"status":"completed","identity":identity,"panel":contract["panel"],"fold":args.fold,
        "tme_mode":args.tme_mode,"seed":seed,"best_epoch":best_epoch,
        "best_validation_monitor":{"name":"val_error","value":best},"provenance":payload,
        "slurm_job_id":os.environ.get("SLURM_JOB_ID"),
        "architecture":{**arch,"trainable_parameter_count":sum(n for _,n in trainable)},
        "split_counts":{p:len(r) for p,r in phases.items()},"native_vila":native_metrics,"tme_guided_vila":guided,
        "artifact_sha256":{p.name:sha(p) for p in files},
        "evaluation_note":"Existing benchmark holdouts; baseline results previously inspected. Not a newly untouched external test."}
    _atomic_json(output/"metrics.json",report)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config",required=True)
    parser.add_argument("--fold",type=int,required=True)
    parser.add_argument("--tme-mode",choices=["actual","zero"],required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--expected-identity")
    args=parser.parse_args(argv)
    load_dotenv(PGVL/".env")
    os.environ.setdefault("PGVL_REPO_ROOT",str(PGVL))
    os.environ.setdefault("PATHOTME_DATA_ROOT","/path/to/shared/PathoTME-data")
    os.environ.setdefault("PATHOTME_RESULTS_ROOT","/path/to/shared/PathoTME-results")
    contract=load_yaml_config(args.experiment_config)
    cfg,phases,payload,identity=preflight(contract,args.fold,args.tme_mode,"smoke" if args.smoke_only else "train")
    if args.expected_identity and identity!=args.expected_identity:raise ValueError("launch identity changed")
    output=Path(contract["results_root"])
    if args.smoke_only:output/= "smoke"
    output=output/args.tme_mode/f"fold{args.fold}"
    ready={"status":"ready","identity":identity,"fold":args.fold,"tme_mode":args.tme_mode,
           "panel":contract["panel"],"results_dir":str(output),"split_counts":{p:len(r) for p,r in phases.items()}}
    if args.check_only:print(json.dumps(ready,indent=2));return 0
    output.mkdir(parents=True,exist_ok=True)
    with (output/".run.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        metric=output/("smoke_report.json" if args.smoke_only else "metrics.json")
        if metric.exists():
            saved=json.loads(metric.read_text())
            if saved.get("identity")!=identity or saved.get("status") not in ("completed","smoke_passed"):
                raise ValueError("incompatible result state")
            if any(sha(output/name)!=value for name,value in saved.get("artifact_sha256",{}).items()):
                raise ValueError("completed artifacts changed")
            print(json.dumps({**ready,"status":"already_completed"}));return 0
        if any(p.name!=".run.lock" for p in output.iterdir()):raise ValueError("partial results preserved; choose a new explicit version")
        execute(args,cfg,phases,contract,payload,identity,output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
