#!/usr/bin/env python3
"""Run one fingerprinted, exact-fold PathoTME MGPATH condition.

--check-only is dependency-light and read-only. Completed matching folds are
skipped; changed or partial output directories are never overwritten.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PGVL = Path(os.environ.get("PGVL_REPO_ROOT", "/path/to/PGVL-Gym"))
sys.path.insert(0, str(PGVL))

from common.configuration import load_dotenv, load_yaml_config
from common.run_state import validate_resume_state


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def preflight(contract, fold, mode, execution_mode="train"):
    """Verify baseline ownership, complete TME coverage and patient splits."""
    if fold not in contract["exact_coverage_folds"] or mode not in ("actual", "zero"):
        raise ValueError("unregistered fold or TME mode")
    if (contract["training"]["optimizer"] != "adam"
            or contract["training"]["checkpoint_monitor"] != "val_macro_f1"
            or contract["model"]["raw_feature_count"] != 62
            or contract["model"]["tokenization"] != "semantic_5_low_11_high_v1"):
        raise ValueError("unsupported adapter contract")
    cfg = load_yaml_config(contract["base_config"])
    if (cfg["method"] != "mgpath" or cfg["backbone"] != "plip"
            or cfg["feature_space_id"] != "hf:vinid/plip#vision-preprojection"
            or cfg["feature_resolutions"] != {"low": "5x", "high": "10x"}
            or cfg["label_dict"] != {"LUAD": 0, "LUSC": 1}
            or cfg["shots"] != 16 or cfg.get("encoder_extension")):
        raise ValueError("requires the registered native PLIP NSCLC 16-shot condition")
    base = Path(contract["base_checkpoint_dir"])
    if base.resolve() != Path(cfg["results_dir"]).resolve():
        raise ValueError("baseline checkpoint directory does not match config")
    state = json.loads((base / "metrics.json").read_text())
    valid = validate_resume_state(state, base / "config.json", "mgpath", cfg)
    record = next((r for r in valid if r["fold"] == fold), None)
    if record is None or any(record.get("sample_failures", {}).values()):
        raise ValueError("baseline fold is missing or has sample failures")
    checkpoint = base / f"fold{fold}_best.pt"
    predictions = base / f"fold{fold}_predictions.csv"
    feature_file = Path(contract["selected_features_csv"])
    features = read_csv(feature_file)
    feature_ids = [r["slide_id"] for r in features]
    if len(set(feature_ids)) != len(feature_ids):
        raise ValueError("duplicate OpenTME slide IDs")
    phases = {}; paths = {}; cases = {}
    for phase in ("train", "val", "test"):
        path = Path(cfg["split_dir"]) / f"fold{fold}" / f"{phase}.csv"
        rows = read_csv(path); phases[phase] = rows; paths[phase] = path
        ids = [r["slide_id"] for r in rows]
        if len(set(ids)) != len(ids) or set(ids) - set(feature_ids):
            raise ValueError(f"{phase}: duplicate IDs or missing TME rows")
        if set(int(r["label_id"]) for r in rows) != {0, 1}:
            raise ValueError(f"{phase}: invalid class coverage")
        if phase != "test" and any(sum(int(r["label_id"]) == c for r in rows) != 16
                                   for c in (0, 1)):
            raise ValueError("train and validation must each contain 16 slides per class")
        cases[phase] = {r["case_id"] for r in rows}
        if any(not Path(os.path.expandvars(r[cfg[key]])).is_file()
               for r in rows for key in ("feature_path_column_l", "feature_path_column_s")):
            raise ValueError(f"{phase}: missing PLIP feature file")
    if any(cases[a] & cases[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("patient leakage between partitions")
    expected = {r["slide_id"]: (r["case_id"], int(r["label_id"])) for r in phases["test"]}
    saved = read_csv(predictions)
    observed = {r["slide_id"]: (r["case_id"], int(r["label"])) for r in saved}
    if len(observed) != len(saved) or observed != expected:
        raise ValueError("baseline predictions do not match the exact holdout")
    if sha(cfg["text_prompt_path"]) != cfg["text_prompt_file_sha256"]:
        raise ValueError("native prompt bank changed")
    files = [Path(contract["base_config"]), base / "config.json", base / "metrics.json",
             checkpoint, predictions, feature_file, Path(cfg["text_prompt_path"]), *paths.values(),
             Path(cfg["dataset_csv"]), Path(__file__), ROOT / "pathotme/guided_mgpath.py",
             ROOT / "pathotme/guided_mgpath_adapter.py", ROOT / "pathotme/guided_vila.py",
             ROOT / "pathotme/features.py", ROOT / "scripts/run_vila_guided.py",
             PGVL / "methods/mgpath/model.py", PGVL / "methods/mgpath/adapter.py",
             PGVL / "methods/mgpath/dataset.py", PGVL / "train.py"]
    payload = {"contract": contract, "fold": fold, "tme_mode": mode,
               "execution_mode": execution_mode,
               "source_sha256": {str(p.resolve()): sha(p) for p in files}}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return cfg, phases, payload, identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--tme-mode", choices=["actual", "zero"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--expected-identity")
    args = parser.parse_args(argv)
    load_dotenv(PGVL / ".env")
    os.environ.setdefault("PGVL_REPO_ROOT", str(PGVL))
    os.environ.setdefault("PATHOTME_DATA_ROOT", "/path/to/shared/PathoTME-data")
    os.environ.setdefault("PATHOTME_RESULTS_ROOT", "/path/to/shared/PathoTME-results")
    contract = load_yaml_config(args.experiment_config)
    cfg, phases, payload, identity = preflight(
        contract, args.fold, args.tme_mode, "smoke" if args.smoke_only else "train")
    if args.expected_identity and identity != args.expected_identity:
        raise ValueError("inputs or implementation changed since launch planning")
    output = Path(contract["results_root"]) / args.tme_mode / f"fold{args.fold}"
    if args.smoke_only:
        output = Path(contract["results_root"]) / "smoke" / args.tme_mode / f"fold{args.fold}"
    ready = {"status": "ready", "identity": identity, "fold": args.fold,
             "tme_mode": args.tme_mode, "split_counts": {k: len(v) for k, v in phases.items()},
             "results_dir": str(output), "baseline_checkpoint_verified": True}
    if args.check_only:
        print(json.dumps(ready, indent=2)); return 0
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metrics_file = output / "metrics.json"
        if metrics_file.exists():
            saved = json.loads(metrics_file.read_text())
            if saved.get("identity") != identity or saved.get("status") != "completed":
                raise ValueError("refusing incompatible result state")
            if not all(sha(output / p) == h for p, h in saved["artifact_sha256"].items()):
                raise ValueError("completed artifacts were modified")
            print(json.dumps({**ready, "status": "already_completed"})); return 0
        if any(p.name != ".run.lock" for p in output.iterdir()):
            raise ValueError("partial output exists; preserve it and use a new explicit result root")
        execute(args, cfg, phases, contract, payload, identity, output)
    return 0


def execute(args, cfg, phases, contract, payload, identity, output):
    """Train the adapter with validation-only selection, then evaluate once."""
    import numpy as np
    import pandas as pd
    import torch
    from train import build_loaders, classification_metrics, set_seed
    from run_vila_guided import (_atomic_json, _atomic_csv, _atomic_torch,
                                 _run_epoch, _evaluate_with_attention, _metric_bundle)
    from pathotme.guided_mgpath_adapter import TMEGuidedMGPathMethod

    training = contract["training"]; architecture = contract["model"]
    cfg = {**cfg, "_fold_index": args.fold, "results_dir": str(output),
           "base_checkpoint_dir": contract["base_checkpoint_dir"],
           "tme_feature_csv": contract["selected_features_csv"], "tme_mode": args.tme_mode,
           "tme_hidden_dim": architecture["hidden_dim"],
           "tme_attention_heads": architecture["attention_heads"],
           "tme_dropout": architecture["dropout"], "tme_initial_gate": architecture["initial_gate"],
           "lr": training["lr"], "weight_decay": training["weight_decay"]}
    _atomic_json(output / "config.json", {**payload, "identity": identity,
                 "resolved_config": cfg, "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    seed = int(contract["seed"]) + args.fold; set_seed(seed)
    loaders = build_loaders("mgpath", cfg, args.fold)
    method = TMEGuidedMGPathMethod(cfg, device=args.device)
    model = method.build_model(); method.prepare_fold(args.fold, model, loaders[0])
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith("conditioner.") for n, _ in trainable):
        raise RuntimeError("unexpected trainable parameters")
    # Structural bypass must reproduce the exact native model on a training
    # bag before optimization. No held-out performance is used by this gate.
    rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    batch = next(iter(loaders[0])); inputs, _, _ = method.inputs(batch)
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(*inputs, bypass_conditioner=True),
                                   model.base(*inputs[:4]), rtol=1e-5, atol=1e-5)
    torch.set_rng_state(rng)
    if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
    print("Native bypass equivalence passed; training adapter only", flush=True)
    optimizer = method.build_optimizer(model)
    if args.smoke_only:
        model.train()
        step = method.train_step(batch, model, optimizer)
        gradients = [p.grad for p in model.conditioner.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
            raise RuntimeError("TME adapter has missing/non-finite/zero gradients")
        report = {"status": "smoke_passed", "identity": identity,
                  "native_bypass_equivalent": True, "adapter_only_gradients": True,
                  "trainable_parameters": sum(p for _, p in trainable),
                  "loss": step["loss"], "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        _atomic_json(output / "smoke_report.json", report)
        print(json.dumps(report), flush=True)
        return
    best = float("inf"); best_epoch = None; stale = 0; history = []
    best_path = output / f"fold{args.fold}_best_adapter.pt"
    final_path = output / f"fold{args.fold}_final_adapter.pt"
    def checkpoint(epoch):
        return {"format": "pathotme_mgpath_guided_v1", "identity": identity,
                "epoch": epoch, "state_dict": model.adapter_state_dict()}
    for epoch in range(training["epochs"]):
        tr = _run_epoch(loaders[0], method, model, classification_metrics, optimizer)
        with torch.no_grad():
            val = _run_epoch(loaders[1], method, model, classification_metrics)
        monitor = -float(val["metrics"]["macro_f1"])
        improved = monitor <= best
        if improved:
            best = monitor; best_epoch = epoch; stale = 0
            _atomic_torch(best_path, checkpoint(epoch))
        else: stale += 1
        history.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": val["loss"],
                        "val_macro_f1": -monitor, "checkpoint_improved": improved})
        _atomic_csv(output / "training_history.csv", pd.DataFrame(history))
        print(f"epoch={epoch} train_loss={tr['loss']:.4f} val_macro_f1={-monitor:.4f}", flush=True)
        if stale >= training["early_stopping_patience"] and epoch > training["early_stopping_min_epoch"]:
            break
    _atomic_torch(final_path, checkpoint(epoch))
    model.load_adapter_state_dict(torch.load(best_path, map_location=args.device, weights_only=True)["state_dict"])
    probabilities, labels, metadata, attention, guided = _evaluate_with_attention(
        loaders[2], method, model, classification_metrics)
    frame = pd.DataFrame(metadata); frame["label"] = labels
    frame["prediction"] = probabilities.argmax(-1)
    frame[["probability_0", "probability_1"]] = probabilities
    frame = pd.concat([frame, pd.DataFrame(attention)], axis=1)
    native = pd.read_csv(Path(contract["base_checkpoint_dir"]) / f"fold{args.fold}_predictions.csv")
    native = native.set_index("slide_id", verify_integrity=True).loc[frame["slide_id"]]
    if not np.array_equal(native["label"].to_numpy(), labels) or not np.array_equal(native["case_id"].to_numpy(), frame["case_id"].to_numpy()):
        raise ValueError("native/guided evaluation identities differ")
    native_probs = native[["probability_0", "probability_1"]].to_numpy()
    native_metrics = _metric_bundle(native_probs, labels, metadata, classification_metrics)
    frame[["native_mgpath_probability_0", "native_mgpath_probability_1"]] = native_probs
    prediction_path = output / f"fold{args.fold}_predictions.csv"
    _atomic_csv(prediction_path, frame)
    files = [best_path, final_path, prediction_path, output / "training_history.csv", output / "config.json"]
    result = {"status": "completed", "identity": identity, "fold": args.fold,
              "tme_mode": args.tme_mode, "seed": seed, "best_epoch": best_epoch,
              "best_validation_monitor": {"name": "val_macro_f1", "value": best},
              "provenance": payload, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "architecture": {**architecture, "trainable_parameter_count": sum(p for _, p in trainable),
                               "image_centers": 64, "frozen_native_mgpath": True},
              "split_counts": {k: len(v) for k, v in phases.items()},
              "native_mgpath": native_metrics, "tme_guided_mgpath": guided,
              "delta_vs_native_slide": {k: guided["slide_metrics"][k] - native_metrics["slide_metrics"][k]
                                        for k in ("balanced_accuracy", "macro_f1", "auroc_ovr")},
              "artifact_sha256": {p.name: sha(p) for p in files},
              "evaluation_note": "Previously inspected diagnostic holdouts; not an untouched final five-fold result"}
    _atomic_json(output / "metrics.json", result)
    print(json.dumps({"status": "completed", "results_dir": str(output),
                      "delta_vs_native_slide": result["delta_vs_native_slide"]}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
