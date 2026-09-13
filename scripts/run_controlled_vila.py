#!/usr/bin/env python3
"""Strict, isolated CONCH ViLa controls and WSI-only privileged supervision.

Planning is Torch-free. Existing native and TME-only outputs are read-only.
Each training condition is one fold; source hashes bind queued work to code.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PGVL = Path(os.environ.get("PGVL_REPO_ROOT", "/path/to/PGVL-Gym"))
sys.path[:0] = [str(ROOT), str(PGVL)]
from run_brca_vila import preflight as reference_preflight, read_csv, sha
from common.configuration import load_dotenv, load_yaml_config
from common.run_state import validate_resume_state
from pathotme.controlled_vila import CONDITIONS, STUDENTS, SHUFFLE_POLICY, donor_maps, select_fusion


def environment():
    load_dotenv(PGVL/".env")
    for k, v in {"PGVL_REPO_ROOT": str(PGVL), "PATHOTME_ROOT": str(ROOT),
                 "PATHOTME_DATA_ROOT": "/path/to/shared/PathoTME-data",
                 "PATHOTME_RESULTS_ROOT": "/path/to/shared/PathoTME-results"}.items():
        os.environ.setdefault(k, v)


def output_dir(contract, fold, condition, smoke=False):
    root = Path(contract["results_root"])
    return (root/"smoke" if smoke else root)/condition/f"fold{fold}"


def verify_completed(output, identity, smoke=False):
    """Require exact identity and every declared artifact hash; never overwrite."""
    marker = Path(output)/("smoke_report.json" if smoke else "metrics.json")
    if not marker.exists():
        return None
    report = json.loads(marker.read_text())
    if report.get("identity") != identity or report.get("status") != ("smoke_passed" if smoke else "completed"):
        raise ValueError(f"incompatible completion marker: {marker}")
    if not report.get("artifact_sha256") or any(sha(Path(output)/p) != h for p, h in report["artifact_sha256"].items()):
        raise ValueError(f"missing/changed artifacts: {marker}")
    return report


def preflight(contract, fold, condition, smoke=False):
    """Check native CONCH, original panel control and source-patient identities."""
    if (condition not in CONDITIONS or condition not in contract["conditions"]
            or fold not in contract["folds"] or contract["panel"] != "brca_morph64_v1"
            or contract["shots"] != 16 or contract["shuffle_policy"] != SHUFFLE_POLICY):
        raise ValueError("unregistered experiment condition")
    t = contract["training"]; s = contract["student"]
    if (t["optimizer"] != "adam" or t["checkpoint_monitor"] != "val_error" or t["epochs"] < 1
            or t["lr"] <= 0 or s["temperature"] <= 0 or s["auxiliary_weight"] < 0
            or s["distillation_weight"] < 0 or contract["fusion"]["selection"] != "validation_error_ties_smallest_alpha"
            or s["teachers"] != {"kd_visual": "visual", "kd_real": "actual", "kd_shuffled": "shuffled"}):
        raise ValueError("unsupported training/selection policy")
    reference = load_yaml_config(contract["reference_contract"])
    ref_cfg, phases, ref_payload, ref_id = reference_preflight(reference, fold, "actual", "tme_only")
    control_dir = Path(reference["results_root"])/"tme_only"/f"fold{fold}"
    control = verify_completed(control_dir, ref_id)
    if control is None:
        raise ValueError("registered TME-only comparator is not completed")
    cfg = load_yaml_config(contract["base_config"])
    required = {"method": "vila_mil", "task": "brca", "backbone": "conch", "feature_dim": 512,
                "feature_space_id": "hf:MahmoodLab/conch", "shots": 16,
                "encoder_extension_strategy": "paired_feature_context_v1",
                "feature_resolutions": {"low": "5x", "high": "10x"}, "label_dict": {"IDC": 0, "ILC": 1}}
    if any(cfg.get(k) != v for k, v in required.items()) or not cfg.get("encoder_extension"):
        raise ValueError("requires explicit CONCH paired-feature ViLa BRCA condition")
    for k in ("classnames", "vila_prompt_file_classnames", "text_prompt_path", "vila_prompt_file_sha256",
              "vila_prompt_bank_sha256", "vila_prompt_format", "vila_prompt_layout", "vila_scale_recipe"):
        if cfg[k] != ref_cfg[k]:
            raise ValueError(f"native comparison semantics differ: {k}")
    base = Path(contract["base_checkpoint_dir"])
    if base.resolve() != Path(cfg["results_dir"]).resolve():
        raise ValueError("CONCH baseline directory mismatch")
    valid = validate_resume_state(json.loads((base/"metrics.json").read_text()), base/"config.json", "vila_mil", cfg)
    record = next((r for r in valid if r["fold"] == fold), None)
    if record is None or any(record.get("sample_failures", {}).values()):
        raise ValueError("CONCH baseline incomplete or failed samples")
    for phase, rows in phases.items():
        _, other = read_csv(Path(cfg["split_dir"])/f"fold{fold}"/f"{phase}.csv")
        if other != rows:
            raise ValueError("native and reference exact split tables differ")
        for row in rows:
            for key in ("feature_path_column_s", "feature_path_column_l"):
                if not Path(os.path.expandvars(row[cfg[key]])).is_file():
                    raise ValueError(f"missing declared CONCH feature for {row['slide_id']}")
    _, predictions = read_csv(base/f"fold{fold}_predictions.csv")
    expected = {r["slide_id"]: (r["case_id"], int(r["label_id"])) for r in phases["test"]}
    observed = {r["slide_id"]: (r["case_id"], int(r["label"])) for r in predictions}
    if observed != expected or len(observed) != len(predictions):
        raise ValueError("CONCH prediction contract mismatch")
    files = [Path(contract["base_config"]), Path(contract["reference_contract"]), base/"config.json",
             base/"metrics.json", base/f"fold{fold}_best.pt", base/f"fold{fold}_predictions.csv",
             control_dir/"metrics.json", *[control_dir/n for n in control["artifact_sha256"]],
             Path(__file__), ROOT/"pathotme/controlled_vila.py", ROOT/"pathotme/privileged_vila.py",
             PGVL/"common/models/paired_encoder_extension.py", Path(cfg["backbone_weights"])]
    payload = {"contract": contract, "fold": fold, "condition": condition, "smoke": smoke,
               "reference_tme_only_identity": ref_id, "reference_inputs": ref_payload,
               "source_sha256": {str(p.resolve()): sha(p) for p in files},
               "donor_maps": donor_maps(phases, contract["seed"], fold),
               "inference_requires_tme": condition not in STUDENTS,
               "test_use": "final evaluation only; previously inspected TCGA benchmark, not untouched external data"}
    if condition in ("kd_real", "kd_shuffled"):
        teacher = s["teachers"][condition]
        payload["teacher_identity"] = preflight(contract, fold, teacher, False)[3]
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cfg, phases, payload, identity


def execute(contract, cfg, phases, payload, identity, output, fold, condition, device, smoke):
    """Execute one registered condition with a strictly source-selected checkpoint."""
    import copy
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from train import build_loaders, classification_metrics, set_seed
    from methods.vila_mil.adapter import ViLaMILMethod
    from pathotme.brca_features import panel_spec, transform_rows
    from pathotme.brca_vila import BreastGuidedViLaMIL
    from pathotme.guided_vila import FoldStandardizer
    from pathotme.guided_vila_adapter import _one_metadata_value, _load_torch_state
    from pathotme.privileged_vila import (ControlledGuidedMethod, PrivilegedStudent,
                                         visual_inputs, auxiliary_loss, distillation_loss)
    from run_vila_guided import _atomic_json, _atomic_csv, _atomic_torch, _metric_bundle

    reference = load_yaml_config(contract["reference_contract"])
    arch, training, student_cfg = contract["model"], contract["training"], contract["student"]
    cfg = {**cfg, "_fold_index": fold, "results_dir": str(output),
           "base_checkpoint_dir": contract["base_checkpoint_dir"],
           "tme_feature_csv": reference["selected_features_csv"], "tme_panel": contract["panel"],
           "tme_mode": "zero" if condition == "zero" else "actual",
           "tme_hidden_dim": arch["hidden_dim"], "tme_attention_heads": arch["attention_heads"],
           "tme_dropout": arch["dropout"], "tme_initial_gate": arch["initial_gate"]}
    _atomic_json(output/"config.json", {"identity": identity, "provenance": payload, "resolved_config": cfg})
    seed = contract["seed"]+fold
    set_seed(seed)
    loaders = build_loaders("vila_mil", cfg, fold)
    names = panel_spec(contract["panel"])["feature_names"]
    _, raw_rows = read_csv(reference["selected_features_csv"])
    table = dict(zip((r["slide_id"] for r in raw_rows), transform_rows(raw_rows, contract["panel"])))
    flat_donors = {a: b for m in payload["donor_maps"].values() for a, b in m.items()}
    scaler = FoldStandardizer(len(names)).to(device)
    scaler.fit(np.asarray([table[r["slide_id"]] for r in phases["train"]]))
    native_method = ViLaMILMethod(cfg, device)
    base = native_method.build_model()
    base.load_state_dict(_load_torch_state(Path(contract["base_checkpoint_dir"])/f"fold{fold}_best.pt", device), strict=True)
    base.eval()
    measured = condition in ("actual", "zero", "shuffled")
    if measured:
        model = BreastGuidedViLaMIL(base, panel=contract["panel"], tme_mode=cfg["tme_mode"],
                    hidden_dim=arch["hidden_dim"], num_heads=arch["attention_heads"],
                    dropout=arch["dropout"], initial_gate=arch["initial_gate"]).to(device)
        model.standardizer.load_state_dict(scaler.state_dict())
        method = ControlledGuidedMethod(cfg, device, flat_donors if condition == "shuffled" else None)
    elif condition in STUDENTS:
        model = PrivilegedStudent(base, len(names), student_cfg["auxiliary_hidden_dim"]).to(device)
    else:
        model = base

    def metadata(batch):
        return {k: _one_metadata_value(batch[2], k) for k in ("slide_id", "case_id")}

    def raw_target(batch, shuffled=False):
        slide = metadata(batch)["slide_id"]
        # This function is called for students on the training split ONLY.
        slide = flat_donors[slide] if shuffled else slide
        return torch.tensor(table[slide], dtype=torch.float32, device=device).unsqueeze(0)

    def eval_visual(loader, current, is_student=False):
        current.eval(); probabilities=[]; labels=[]; metas=[]
        with torch.no_grad():
            for batch in loader:
                b = current.base if is_student else current
                x = visual_inputs(batch, b, device)
                details = current(*x) if is_student else current(*x, return_details=True)
                probabilities.append(details["probabilities"][0].cpu().numpy())
                labels.append(int(x[-1].item())); metas.append(metadata(batch))
        return np.asarray(probabilities), np.asarray(labels), metas

    def prediction_frame(probs, labels, metas):
        if probs.shape != (len(labels), 2) or not np.isfinite(probs).all() or not np.allclose(probs.sum(1), 1):
            raise ValueError("invalid output probabilities")
        frame=pd.DataFrame(metas); frame["label"]=labels; frame["prediction"]=probs.argmax(1)
        frame[["probability_0", "probability_1"]]=probs
        return frame

    artifacts = [output/"config.json"]
    teacher_artifacts = {}
    if condition == "fusion":
        control_dir=Path(reference["results_root"])/"tme_only"/f"fold{fold}"
        saved=joblib.load(control_dir/f"fold{fold}_classifier.joblib")
        if saved["identity"] != payload["reference_tme_only_identity"]:
            raise ValueError("TME classifier identity mismatch")
        classifier=saved["pipeline"]
        if list(classifier.classes_) != [0, 1]: raise ValueError("TME classifier class order mismatch")
        v,y,meta=eval_visual(loaders[1],base)
        t=classifier.predict_proba(np.asarray([table[m["slide_id"]] for m in meta]))
        alpha,scores=select_fusion(v,t,y,contract["fusion"]["alpha_grid"])
        val_frame=prediction_frame(v,y,meta)
        val_frame[["tme_probability_0", "tme_probability_1"]]=t
        path=output/"validation_predictions.csv"; _atomic_csv(path,val_frame); artifacts.append(path)
        report={"selected_alpha":alpha, "validation_candidates":scores, "retrained_classifiers":False}
        if not smoke:
            visual=pd.read_csv(Path(contract["base_checkpoint_dir"])/f"fold{fold}_predictions.csv").set_index("slide_id",verify_integrity=True)
            tme=pd.read_csv(control_dir/f"fold{fold}_predictions.csv").set_index("slide_id",verify_integrity=True)
            ids=[r["slide_id"] for r in phases["test"]]; y=np.array([int(r["label_id"]) for r in phases["test"]])
            meta=[{k:r[k] for k in ("slide_id","case_id")} for r in phases["test"]]
            for frame in (visual,tme):
                if set(frame.index)!=set(ids) or not np.array_equal(frame.loc[ids,"label"],y) or list(frame.loc[ids,"case_id"]) != [m["case_id"] for m in meta]:
                    raise ValueError("fusion prediction identities differ")
            v=visual.loc[ids,["probability_0","probability_1"]].to_numpy()
            t=tme.loc[ids,["probability_0","probability_1"]].to_numpy()
            # Validate numeric inputs without consulting test labels for selection.
            prediction_frame(v,y,meta); prediction_frame(t,y,meta)
            probs=(1-alpha)*v+alpha*t
            report["metrics"]=_metric_bundle(probs,y,meta,classification_metrics)
            report["fixed_half_fusion"]=_metric_bundle(.5*v+.5*t,y,meta,classification_metrics)
            frame=prediction_frame(probs,y,meta)
            frame[["visual_probability_0","visual_probability_1"]]=v
            frame[["tme_probability_0","tme_probability_1"]]=t
            path=output/f"fold{fold}_predictions.csv"; _atomic_csv(path,frame); artifacts.append(path)
    else:
        teacher_cache={}
        if condition.startswith("kd_"):
            teacher=copy.deepcopy(base).eval()
            if condition != "kd_visual":
                teacher=BreastGuidedViLaMIL(teacher,panel=contract["panel"],tme_mode="actual",
                         hidden_dim=arch["hidden_dim"],num_heads=arch["attention_heads"],dropout=arch["dropout"],
                         initial_gate=arch["initial_gate"]).to(device)
                if smoke:
                    teacher.standardizer.load_state_dict(scaler.state_dict())
                else:
                    teacher_dir=output_dir(contract,fold,student_cfg["teachers"][condition])
                    completed=verify_completed(teacher_dir,payload["teacher_identity"])
                    if completed is None: raise ValueError("teacher is not complete")
                    path=teacher_dir/f"fold{fold}_best.pt"
                    checkpoint=_load_torch_state(path,device)
                    if checkpoint["identity"]!=payload["teacher_identity"]: raise ValueError("teacher checkpoint identity mismatch")
                    teacher.load_state_dict(checkpoint["state_dict"],strict=True)
                    teacher_artifacts={str(teacher_dir/"metrics.json"):sha(teacher_dir/"metrics.json"),str(path):sha(path)}
            teacher.eval()
            for p in teacher.parameters(): p.requires_grad_(False)
            with torch.no_grad():
                for batch in loaders[0]:
                    x=visual_inputs(batch,teacher if condition=="kd_visual" else teacher.base,device)
                    d=(teacher(*x,return_details=True) if condition=="kd_visual" else
                       teacher(*x,raw_target(batch,condition=="kd_shuffled"),return_details=True))
                    teacher_cache[metadata(batch)["slide_id"]]=d["logits"].detach().cpu()
            if set(teacher_cache)!={r["slide_id"] for r in phases["train"]}: raise ValueError("teacher training coverage mismatch")
            del teacher
            cache_path=output/"training_teacher_logits.pt"; _atomic_torch(cache_path,teacher_cache); artifacts.append(cache_path)

        # Reset after diagnostic/data/cache construction to match training orders.
        set_seed(seed)
        trainable=[(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
        if measured and any(not n.startswith("conditioner.") for n,_ in trainable): raise RuntimeError("native gradient leak")
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=training["lr"],weight_decay=training["weight_decay"])

        def step(batch, train=False):
            if measured:
                x,_=method._inputs(batch,model)
                d=model(*x,return_details=True)
                loss=d["loss"]
            else:
                x=visual_inputs(batch,model.base,device)
                d=model(*x,auxiliary=train and condition.startswith("aux_"))
                loss=d["loss"]
                if train and condition.startswith("aux_"):
                    loss=loss+student_cfg["auxiliary_weight"]*auxiliary_loss(d["tme_prediction"],raw_target(batch,condition=="aux_shuffled"),scaler)
                if train and condition.startswith("kd_"):
                    target=teacher_cache[metadata(batch)["slide_id"]].to(device)
                    loss=loss+student_cfg["distillation_weight"]*distillation_loss(d["logits"],target,student_cfg["temperature"])
            if not torch.isfinite(loss) or not torch.isfinite(d["logits"]).all(): raise RuntimeError("nonfinite loss/logits")
            if train:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None): raise RuntimeError("nonfinite gradients")
                optimizer.step()
            return d, float(loss.detach())

        def evaluate(loader):
            model.eval(); ps=[];ys=[];ms=[]
            with torch.no_grad():
                for batch in loader:
                    d,_=step(batch,False)
                    ps.append(d["probabilities"][0].cpu().numpy()); ys.append(int(batch[-1].item())); ms.append(metadata(batch))
            return np.asarray(ps),np.asarray(ys),ms

        if smoke:
            batch=next(iter(loaders[0])); model.eval()
            with torch.no_grad():
                if measured:
                    x,_=method._inputs(batch,model); native=model.base(*x[:5])[0]
                    torch.testing.assert_close(model(*x,bypass_conditioner=True)[0],native)
                    projection=model.conditioner.output_projection.weight.clone();model.conditioner.output_projection.weight.zero_()
                    torch.testing.assert_close(model(*x)[0],native,rtol=1e-5,atol=1e-5)
                    model.conditioner.output_projection.weight.copy_(projection)
                else:
                    x=visual_inputs(batch,model.base,device)
                    torch.testing.assert_close(model(*x)["probabilities"],model.base(*x)[0])
            model.train();_,loss=step(batch,True)
            if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()): raise RuntimeError("no gradients")
            report={"loss":loss,"native_equations_verified":True,"training_only_smoke":True,
                    "teacher_is_untrained_smoke_fixture":condition in ("kd_real","kd_shuffled")}
        else:
            best=float("inf");best_epoch=None;stale=0;history=[]
            best_path=output/f"fold{fold}_best.pt"
            for epoch in range(training["epochs"]):
                model.train();losses=[]
                for batch in loaders[0]: losses.append(step(batch,True)[1])
                vp,vy,vm=evaluate(loaders[1]);error=float((vp.argmax(1)!=vy).mean())
                if error<best:
                    best=error;best_epoch=epoch;stale=0
                    _atomic_torch(best_path,{"identity":identity,"epoch":epoch,"state_dict":model.state_dict()})
                    _atomic_csv(output/"validation_predictions.csv",prediction_frame(vp,vy,vm))
                else: stale+=1
                history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"val_error":error})
                _atomic_csv(output/"training_history.csv",pd.DataFrame(history))
                print(f"{condition} fold={fold} epoch={epoch} loss={np.mean(losses):.4f} val_error={error:.4f}",flush=True)
                if stale>=training["early_stopping_patience"] and epoch>training["early_stopping_min_epoch"]: break
            if best_epoch is None: raise RuntimeError("no validation checkpoint")
            model.load_state_dict(_load_torch_state(best_path,device)["state_dict"],strict=True)
            probs,y,meta=evaluate(loaders[2]); prediction_path=output/f"fold{fold}_predictions.csv"
            _atomic_csv(prediction_path,prediction_frame(probs,y,meta))
            artifacts.extend([best_path,prediction_path,output/"validation_predictions.csv",output/"training_history.csv"])
            report={"best_epoch":best_epoch,"validation_error":best,"metrics":_metric_bundle(probs,y,meta,classification_metrics)}
            if condition in STUDENTS:
                path=output/f"fold{fold}_inference.pt"
                _atomic_torch(path,{"identity":identity,"state_dict":model.base.state_dict(),
                                   "input_contract":"native_CONCH_ViLa_WSI_only_no_TME", "config":load_yaml_config(contract["base_config"])})
                artifacts.append(path)
        report["trainable_parameters"]=sum(n for _,n in trainable)
        report["trainable_parameter_names"]=[n for n,_ in trainable]
    report.update(status="smoke_passed" if smoke else "completed",identity=identity,fold=fold,condition=condition,
                  provenance=payload,seed=seed,slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                  teacher_artifact_sha256=teacher_artifacts,artifact_sha256={p.name:sha(p) for p in artifacts})
    _atomic_json(output/("smoke_report.json" if smoke else "metrics.json"),report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--condition",choices=CONDITIONS)
    p.add_argument("--fold",type=int,default=0);p.add_argument("--device",default="cuda:0")
    p.add_argument("--check-only",action="store_true");p.add_argument("--smoke-only",action="store_true")
    p.add_argument("--smoke-suite",action="store_true");p.add_argument("--expected-identity")
    p.add_argument("--expected-suite-identity")
    args=p.parse_args();environment();contract=load_yaml_config(args.config)
    if not args.smoke_suite and args.condition is None: p.error("--condition is required")
    if args.expected_suite_identity:
        identities=[preflight(contract,args.fold,c,True)[3] for c in CONDITIONS]
        actual=hashlib.sha256(json.dumps(identities).encode()).hexdigest()
        if actual!=args.expected_suite_identity: raise ValueError("queued smoke suite identity changed")
    for condition in CONDITIONS if args.smoke_suite else [args.condition]:
        smoke=args.smoke_only or args.smoke_suite
        cfg,phases,payload,identity=preflight(contract,args.fold,condition,smoke)
        if args.expected_identity and identity!=args.expected_identity: raise ValueError("queued identity changed")
        output=output_dir(contract,args.fold,condition,smoke)
        if args.check_only:
            print(json.dumps({"condition":condition,"fold":args.fold,"identity":identity,"output":str(output)}));continue
        output.mkdir(parents=True,exist_ok=True)
        with (output/".run.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if verify_completed(output,identity,smoke): print(f"already_completed {output}",flush=True);continue
            if any(p.name!=".run.lock" for p in output.iterdir()): raise ValueError(f"partial output preserved: {output}")
            execute(contract,cfg,phases,payload,identity,output,args.fold,condition,args.device,smoke)


if __name__=="__main__":
    main()
