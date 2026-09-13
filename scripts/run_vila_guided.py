#!/usr/bin/env python3
"""Train and evaluate the TME-guided ViLa-MIL prototype adapter."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

# Make direct ``python scripts/run_vila_guided.py`` invocation equivalent to
# the Slurm wrapper's PYTHONPATH without relying on the caller's directory.
PATHOTME_ROOT = Path(__file__).resolve().parents[1]
if str(PATHOTME_ROOT) not in sys.path:
    sys.path.insert(0, str(PATHOTME_ROOT))

from pathotme.features import (
    FEATURE_NAMES,
    SCHEMA_SHA256,
    validate_feature_table,
)
from pathotme.guided_vila import (
    GUIDED_PREPROCESSING_SHA256,
    SemanticTMETokenizer,
)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment contract must be a YAML mapping")
    payload = _expand(payload)
    unresolved = sorted({value for value in _strings(payload) if "${" in value})
    if unresolved:
        raise ValueError(
            "undefined variables remain in experiment contract: "
            + ", ".join(unresolved))
    return payload


def _bootstrap_pgvl(root: Path) -> Path:
    root = root.resolve()
    if not (root / "train.py").is_file():
        raise FileNotFoundError(f"PGVL-Gym train.py is missing under {root}")
    sys.path.insert(0, str(root))
    from common.configuration import load_dotenv
    load_dotenv(root / ".env")
    return root


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_torch(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        return torch.load(path, map_location=device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_bundle(probabilities: np.ndarray, labels: np.ndarray,
                   metadata: list[dict[str, str]], classification_metrics):
    slide_metrics = classification_metrics(probabilities, labels, 2)
    frame = pd.DataFrame(metadata)
    frame["label"] = labels
    frame["probability_0"] = probabilities[:, 0]
    frame["probability_1"] = probabilities[:, 1]
    conflicts = frame.groupby("case_id")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("patient-level aggregation found conflicting labels")
    patient = frame.groupby("case_id", sort=True).agg({
        "label": "first", "probability_0": "mean", "probability_1": "mean",
    })
    patient_metrics = classification_metrics(
        patient[["probability_0", "probability_1"]].to_numpy(),
        patient["label"].to_numpy(dtype=int), 2)
    return {"slide_metrics": slide_metrics, "patient_metrics": patient_metrics}


def _run_epoch(loader, method, model, classification_metrics, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    logits_all = []
    labels_all = []
    metadata_all: list[dict[str, str]] = []
    for batch in loader:
        output = (method.train_step(batch, model, optimizer, None)
                  if training else method.eval_step(batch, model))
        logits = output["logits"].detach().float().cpu()
        labels = output["label"].detach().long().cpu().reshape(-1)
        if logits.shape != (len(labels), 2) or not torch.isfinite(logits).all():
            raise ValueError("adapter returned invalid binary logits")
        loss_total += float(output["loss"]) * len(labels)
        logits_all.append(logits)
        labels_all.append(labels)
        metadata = batch[-2]
        slide_ids = metadata["slide_id"]
        case_ids = metadata["case_id"]
        if isinstance(slide_ids, str):
            slide_ids, case_ids = [slide_ids], [case_ids]
        metadata_all.extend({
            "slide_id": str(slide_id), "case_id": str(case_id),
        } for slide_id, case_id in zip(slide_ids, case_ids))
    if not labels_all:
        raise RuntimeError("loader produced no samples")
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all).numpy()
    probabilities = torch.softmax(logits, dim=1).numpy()
    return {
        "loss": loss_total / len(labels),
        "probabilities": probabilities,
        "labels": labels,
        "metadata": metadata_all,
        "metrics": classification_metrics(probabilities, labels, 2),
    }


@torch.no_grad()
def _evaluate_with_attention(loader, method, model, classification_metrics):
    model.eval()
    probability_rows = []
    labels = []
    metadata = []
    attention_rows = []
    low_names = SemanticTMETokenizer.LOW_TOKEN_NAMES
    high_names = SemanticTMETokenizer.HIGH_TOKEN_NAMES
    for batch in loader:
        details = method.eval_step_with_details(batch, model)
        probability = details["probabilities"].detach().float().cpu().numpy()
        label = int(batch[-1].reshape(-1)[0])
        low = details["low_tme_group_attention"].detach().float().cpu()
        high = details["high_tme_group_attention"].detach().float().cpu()
        # [batch=1, prototypes, tokens] -> mean prototype attention.
        low = low.mean(dim=(0, 1)).numpy()
        high = high.mean(dim=(0, 1)).numpy()
        if len(low) != len(low_names) or len(high) != len(high_names):
            raise ValueError("TME attention width does not match token schema")
        probability_rows.append(probability[0])
        labels.append(label)
        metadata.append(details["metadata"])
        attention_rows.append({
            **{f"tme_attention_low_{name}": float(value)
               for name, value in zip(low_names, low)},
            **{f"tme_attention_high_{name}": float(value)
               for name, value in zip(high_names, high)},
        })
    probabilities = np.asarray(probability_rows)
    labels_array = np.asarray(labels, dtype=int)
    bundle = _metric_bundle(
        probabilities, labels_array, metadata, classification_metrics)
    return probabilities, labels_array, metadata, attention_rows, bundle


def _validate_assets(contract, base_config, fold: int):
    feature_csv = Path(contract["selected_features_csv"])
    features = validate_feature_table(pd.read_csv(feature_csv))
    split_root = Path(base_config["split_dir"]) / f"fold{fold}"
    splits = {
        phase: pd.read_csv(split_root / f"{phase}.csv")
        for phase in ("train", "val", "test")
    }
    for phase, split in splits.items():
        validate_feature_table(features, split["slide_id"])
        if set(split["label_id"].astype(int)) != {0, 1}:
            raise ValueError(f"{phase} split does not contain both classes")
    checkpoint = Path(contract["base_checkpoint_dir"]) / f"fold{fold}_best.pt"
    predictions = (
        Path(contract["base_checkpoint_dir"]) / f"fold{fold}_predictions.csv")
    for path in (feature_csv, checkpoint, predictions):
        if not path.is_file():
            raise FileNotFoundError(path)
    return feature_csv, checkpoint, predictions, splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--tme-mode", choices=("actual", "zero"), default="actual")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--epochs", type=int, help="diagnostic override")
    args = parser.parse_args(argv)

    pgvl_hint = Path(os.environ.get("PGVL_REPO_ROOT", "/path/to/PGVL-Gym"))
    pgvl_root = _bootstrap_pgvl(pgvl_hint)
    os.environ.setdefault("PGVL_REPO_ROOT", str(pgvl_root))
    os.environ.setdefault(
        "PATHOTME_DATA_ROOT", "/path/to/shared/PathoTME-data")
    os.environ.setdefault(
        "PATHOTME_RESULTS_ROOT", "/path/to/shared/PathoTME-results")
    contract = _load_contract(args.experiment_config)
    if Path(contract["pgvl_root"]).resolve() != pgvl_root:
        pgvl_root = _bootstrap_pgvl(Path(contract["pgvl_root"]))
    allowed_folds = tuple(map(int, contract["exact_coverage_folds"]))
    if args.fold not in allowed_folds:
        raise ValueError(
            f"fold {args.fold} is not in exact_coverage_folds={allowed_folds}")

    from common.configuration import load_yaml_config
    from train import build_loaders, classification_metrics, set_seed
    from pathotme.guided_vila_adapter import TMEGuidedViLaMethod

    base_config_path = Path(contract["base_config"])
    base_config = load_yaml_config(base_config_path)
    feature_csv, checkpoint, base_prediction_path, splits = _validate_assets(
        contract, base_config, args.fold)
    output_dir = (Path(contract["results_root"]) / args.tme_mode
                  / f"fold{args.fold}")
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not args.rerun:
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            print(json.dumps({
                "status": "already_completed", "results_dir": str(output_dir),
            }, indent=2))
            return 0
        raise RuntimeError(
            f"incomplete result state exists at {metrics_path}; use --rerun")

    cfg = dict(base_config)
    cfg.update({
        "_fold_index": args.fold,
        "results_dir": str(output_dir),
        "tme_feature_csv": str(feature_csv),
        "base_checkpoint_dir": str(Path(contract["base_checkpoint_dir"])),
        "tme_mode": args.tme_mode,
        "tme_hidden_dim": int(contract["model"]["hidden_dim"]),
        "tme_attention_heads": int(contract["model"]["attention_heads"]),
        "tme_dropout": float(contract["model"]["dropout"]),
        "tme_initial_gate": float(contract["model"]["initial_gate"]),
        "lr": float(contract["training"]["lr"]),
        "weight_decay": float(contract["training"]["weight_decay"]),
        "epochs": int(args.epochs or contract["training"]["epochs"]),
        "es_patience": int(contract["training"]["early_stopping_patience"]),
        "es_stop_epoch": int(contract["training"]["early_stopping_min_epoch"]),
    })
    ready = {
        "status": "ready",
        "experiment": contract["experiment"],
        "fold": args.fold,
        "tme_mode": args.tme_mode,
        "split_counts": {phase: len(split) for phase, split in splits.items()},
        "raw_tme_width": len(FEATURE_NAMES),
        "low_tme_tokens": list(SemanticTMETokenizer.LOW_TOKEN_NAMES),
        "high_tme_tokens": list(SemanticTMETokenizer.HIGH_TOKEN_NAMES),
        "base_checkpoint": str(checkpoint),
        "results_dir": str(output_dir),
    }
    if args.check_only:
        print(json.dumps(ready, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_dir / ".run.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError(f"result directory is busy: {output_dir}") from error

    try:
        _atomic_json(output_dir / "config.json", {
            **contract,
            "resolved_base_config": cfg,
            "selected_fold": args.fold,
            "selected_tme_mode": args.tme_mode,
        })
        seed = int(contract["seed"]) + args.fold
        set_seed(seed)
        train_loader, val_loader, test_loader = build_loaders(
            "vila_mil", cfg, args.fold)
        method = TMEGuidedViLaMethod(cfg, device=args.device)
        model = method.build_model()
        method.prepare_fold(args.fold, model, train_loader)
        trainable = [(name, parameter.numel()) for name, parameter
                     in model.named_parameters() if parameter.requires_grad]
        if not trainable or any(name.startswith("base.") for name, _ in trainable):
            raise RuntimeError("only the PathoTME adapter may be trainable")
        optimizer = method.build_optimizer(model)

        epochs = int(cfg["epochs"])
        patience = int(cfg["es_patience"])
        stop_epoch = int(cfg["es_stop_epoch"])
        best_monitor = float("inf")
        best_epoch = None
        counter = 0
        history = []
        best_path = output_dir / f"fold{args.fold}_best_adapter.pt"
        final_path = output_dir / f"fold{args.fold}_final_adapter.pt"
        for epoch in range(epochs):
            train_result = _run_epoch(
                train_loader, method, model, classification_metrics,
                optimizer=optimizer)
            with torch.no_grad():
                val_result = _run_epoch(
                    val_loader, method, model, classification_metrics)
            monitor = 1.0 - float(val_result["metrics"]["accuracy"])
            improved = monitor < best_monitor
            if improved:
                best_monitor = monitor
                best_epoch = epoch
                counter = 0
                _atomic_torch(best_path, {
                    "format": "pathotme_vila_guided_adapter_v1",
                    "fold": args.fold,
                    "tme_mode": args.tme_mode,
                    "epoch": epoch,
                    "base_checkpoint": str(checkpoint),
                    "feature_schema_sha256": SCHEMA_SHA256,
                    "state_dict": model.adapter_state_dict(),
                })
            else:
                counter += 1
            history.append({
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_accuracy": train_result["metrics"]["accuracy"],
                "val_loss": val_result["loss"],
                "val_accuracy": val_result["metrics"]["accuracy"],
                "val_balanced_accuracy": val_result["metrics"]["balanced_accuracy"],
                "val_macro_f1": val_result["metrics"]["macro_f1"],
                "val_auroc_ovr": val_result["metrics"]["auroc_ovr"],
                "checkpoint_improved": improved,
            })
            _atomic_csv(output_dir / "training_history.csv", pd.DataFrame(history))
            print(
                f"epoch={epoch:03d} train_loss={train_result['loss']:.4f} "
                f"val_loss={val_result['loss']:.4f} val_error={monitor:.4f}",
                flush=True)
            if counter >= patience and epoch > stop_epoch:
                print(f"early stopping at epoch {epoch}", flush=True)
                break
        if best_epoch is None or not best_path.is_file():
            raise RuntimeError("training did not create a best adapter checkpoint")
        _atomic_torch(final_path, {
            "format": "pathotme_vila_guided_adapter_v1",
            "fold": args.fold,
            "tme_mode": args.tme_mode,
            "epoch": history[-1]["epoch"],
            "base_checkpoint": str(checkpoint),
            "feature_schema_sha256": SCHEMA_SHA256,
            "state_dict": model.adapter_state_dict(),
        })
        best_payload = _load_torch(best_path, args.device)
        model.load_adapter_state_dict(best_payload["state_dict"])
        probabilities, labels, metadata, attention, metrics = \
            _evaluate_with_attention(
                test_loader, method, model, classification_metrics)

        predictions = pd.DataFrame(metadata)
        predictions["label"] = labels
        predictions["prediction"] = probabilities.argmax(axis=1)
        predictions["probability_0"] = probabilities[:, 0]
        predictions["probability_1"] = probabilities[:, 1]
        predictions = pd.concat(
            [predictions, pd.DataFrame(attention)], axis=1)

        native = pd.read_csv(base_prediction_path)
        required = (
            "slide_id", "case_id", "label", "probability_0", "probability_1")
        if not set(required).issubset(native.columns):
            raise ValueError("native ViLa prediction file has an invalid schema")
        native = native.loc[:, list(required)].copy()
        native["slide_id"] = native["slide_id"].astype(str)
        expected_ids = set(predictions["slide_id"])
        if set(native["slide_id"]) != expected_ids:
            raise ValueError("native and guided holdout slide sets do not match")
        native = native.set_index("slide_id").loc[predictions["slide_id"]]
        if not np.array_equal(native["label"].to_numpy(dtype=int), labels):
            raise ValueError("native and guided holdout labels do not match")
        if not np.array_equal(
                native["case_id"].astype(str).to_numpy(),
                predictions["case_id"].astype(str).to_numpy()):
            raise ValueError("native and guided holdout case identities do not match")
        native_probabilities = native[["probability_0", "probability_1"]].to_numpy()
        native_metrics = _metric_bundle(
            native_probabilities, labels, metadata, classification_metrics)
        predictions["native_vila_probability_0"] = native_probabilities[:, 0]
        predictions["native_vila_probability_1"] = native_probabilities[:, 1]
        _atomic_csv(output_dir / f"fold{args.fold}_predictions.csv", predictions)

        gate_values = torch.sigmoid(model.conditioner.gate_logits).detach().cpu()
        report = {
            "status": "completed",
            "experiment": contract["experiment"],
            "method": "PathoTME TME-guided ViLa-MIL extension",
            "fold": args.fold,
            "tme_mode": args.tme_mode,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_monitor": {
                "name": "val_error", "value": best_monitor,
            },
            "provenance": {
                "base_method": "vila_mil",
                "base_config": str(base_config_path),
                "base_checkpoint": str(checkpoint),
                "base_checkpoint_sha256": _sha256(checkpoint),
                "base_predictions": str(base_prediction_path),
                "opentme_feature_table": str(feature_csv),
                "opentme_feature_schema_sha256": SCHEMA_SHA256,
                "opentme_transform_sha256": GUIDED_PREPROCESSING_SHA256,
                "implementation_status": "pathotme_extension_not_upstream_vila_mil",
            },
            "architecture": {
                "raw_tme_width": len(FEATURE_NAMES),
                "normalized_tme_width": len(FEATURE_NAMES),
                "low_tme_tokens": list(SemanticTMETokenizer.LOW_TOKEN_NAMES),
                "high_tme_tokens": list(SemanticTMETokenizer.HIGH_TOKEN_NAMES),
                "hidden_dim": cfg["tme_hidden_dim"],
                "attention_heads": cfg["tme_attention_heads"],
                "frozen_native_vila": True,
                "trainable_parameter_count": sum(count for _, count in trainable),
                "final_conditioner_gates": {
                    "low": float(gate_values[0]), "high": float(gate_values[1]),
                },
            },
            "split_counts": {phase: len(split) for phase, split in splits.items()},
            "native_vila": native_metrics,
            "tme_guided_vila": metrics,
            "delta_vs_native_slide": {
                metric: (
                    metrics["slide_metrics"][metric]
                    - native_metrics["slide_metrics"][metric])
                for metric in ("balanced_accuracy", "macro_f1", "auroc_ovr")
            },
            "artifacts": {
                "best_adapter": str(best_path),
                "final_adapter": str(final_path),
                "predictions": str(
                    output_dir / f"fold{args.fold}_predictions.csv"),
                "history": str(output_dir / "training_history.csv"),
            },
            "evaluation_note": (
                "These exact-fold holdouts were inspected during PathoTME "
                "development and are diagnostic, not untouched final tests."),
        }
        _atomic_json(metrics_path, report)
        print(json.dumps({
            "status": "completed", "fold": args.fold,
            "tme_mode": args.tme_mode,
            "native_slide_metrics": native_metrics["slide_metrics"],
            "guided_slide_metrics": metrics["slide_metrics"],
            "results_dir": str(output_dir),
        }, indent=2))
        return 0
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
