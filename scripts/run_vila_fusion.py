#!/usr/bin/env python3
"""Run the leakage-safe ViLa-MIL + OpenTME late-fusion canary.

ViLa-MIL is not retrained. The script loads its validation-selected checkpoint,
exports validation probabilities, fits the TME branch on training slides only,
selects regularization and fusion weight on validation slides, and evaluates
the fold-0 diagnostic partition after selection is complete. Fold 0 has already
been inspected during development and must not be represented as a final,
untouched test result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from pathotme.features import (
    FEATURE_NAMES,
    SCHEMA_SHA256,
    TRANSFORM_SHA256,
    transform_nsclc_features,
    validate_feature_table,
)
from pathotme.fusion import (
    binary_metrics,
    blend_probabilities,
    candidate_grid,
    make_tme_model,
    selection_key,
)


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _load_contract(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment contract must be a YAML mapping")
    expanded = _expand(payload)
    unresolved = [
        value for value in _walk_strings(expanded) if "${" in value]
    if unresolved:
        raise ValueError(
            "Undefined variables remain in experiment contract: "
            + ", ".join(sorted(set(unresolved))))
    return expanded


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


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


def _bootstrap_pgvl(pgvl_root: Path):
    root = pgvl_root.resolve()
    if not (root / "train.py").is_file():
        raise FileNotFoundError(f"PGVL-Gym train.py is missing under {root}")
    sys.path.insert(0, str(root))
    from common.configuration import load_dotenv
    load_dotenv(root / ".env")
    return root


def _load_checkpoint(path: Path, device: str):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _first_metadata_value(metadata: dict, key: str) -> str:
    value = metadata[key]
    if isinstance(value, str):
        return value
    if hasattr(value, "__len__") and len(value) == 1:
        return str(value[0])
    raise ValueError(f"Expected one {key} per ViLa batch, got {value!r}")


def _export_validation_predictions(
    pgvl_root: Path,
    config_path: Path,
    checkpoint_dir: Path,
    fold: int,
    device: str,
) -> pd.DataFrame:
    import torch
    from common.configuration import load_yaml_config
    from methods.vila_mil.adapter import ViLaMILMethod
    from train import _batch_metadata, build_loaders

    config = load_yaml_config(config_path)
    if config.get("method") != "vila_mil":
        raise ValueError("The base config must declare method=vila_mil")
    if config.get("backbone") != "clip-rn50":
        raise ValueError("The initial PathoTME canary requires native CLIP-RN50")
    if config.get("feature_resolutions") != {"low": "5x", "high": "10x"}:
        raise ValueError("The initial PathoTME canary requires native 5x/10x bags")

    method = ViLaMILMethod(config, device=device)
    _, validation_loader, _ = build_loaders("vila_mil", config, fold)
    model = method.build_model()
    checkpoint = checkpoint_dir / f"fold{fold}_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(_load_checkpoint(checkpoint, device), strict=True)
    method.on_checkpoint_loaded(model, "best", fold)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in validation_loader:
            output = method.eval_step(batch, model)
            logits = output["logits"].detach().float().cpu()
            labels = output["label"].detach().long().cpu()
            if logits.shape != (1, 2) or labels.numel() != 1:
                raise ValueError("ViLa canary requires batch_size=1 and two classes")
            metadata = _batch_metadata(batch)
            if metadata is None:
                raise ValueError("ViLa base config must set include_metadata=true")
            probability = torch.softmax(logits, dim=1)[0].numpy()
            rows.append({
                "slide_id": _first_metadata_value(metadata, "slide_id"),
                "case_id": _first_metadata_value(metadata, "case_id"),
                "label": int(labels.item()),
                "probability_0": float(probability[0]),
                "probability_1": float(probability[1]),
            })
    frame = pd.DataFrame(rows).sort_values("slide_id", kind="stable")
    if frame.empty or frame["slide_id"].duplicated().any():
        raise ValueError("ViLa validation export is empty or duplicated")
    return frame.reset_index(drop=True)


def _load_base_test_predictions(
    checkpoint_dir: Path, fold: int, split_table: pd.DataFrame,
) -> pd.DataFrame:
    path = checkpoint_dir / f"fold{fold}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"slide_id", "case_id", "label", "probability_0", "probability_1"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks required prediction columns")
    expected = split_table[["slide_id", "case_id", "label_id"]].rename(
        columns={"label_id": "expected_label"})
    checked = frame.merge(expected, on=["slide_id", "case_id"], how="outer",
                          validate="one_to_one", indicator=True)
    if set(checked["_merge"]) != {"both"}:
        raise ValueError("Saved ViLa predictions do not match the declared test split")
    if not np.array_equal(checked["label"].to_numpy(),
                          checked["expected_label"].to_numpy()):
        raise ValueError("Saved ViLa labels do not match the declared test split")
    return frame.loc[:, sorted(required)].sort_values(
        "slide_id", kind="stable").reset_index(drop=True)


def _attach_features(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(features, on="slide_id", how="left",
                               validate="one_to_one")
    if merged[list(FEATURE_NAMES)].isna().all(axis=1).any():
        missing = merged.loc[
            merged[list(FEATURE_NAMES)].isna().all(axis=1), "slide_id"]
        raise ValueError("No OpenTME row for: " + ", ".join(missing[:5]))
    return merged


def _patient_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    patient = frame[["case_id", "label"]].copy()
    patient["probability"] = probability
    inconsistent = patient.groupby("case_id")["label"].nunique()
    if (inconsistent > 1).any():
        raise ValueError("A patient has conflicting subtype labels")
    patient = patient.groupby("case_id", sort=True).agg(
        {"label": "first", "probability": "mean"})
    return binary_metrics(patient["label"].to_numpy(),
                          patient["probability"].to_numpy())


def _metric_bundle(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    return {
        "slide_metrics": binary_metrics(frame["label"].to_numpy(), probability),
        "patient_metrics": _patient_metrics(frame, probability),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fold", type=int,
        help="override the contract fold while retaining every other setting")
    parser.add_argument("--refresh-vila-validation", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true",
        help="validate paths, split coverage, schema, and saved test predictions")
    args = parser.parse_args(argv)

    contract = _load_contract(args.experiment_config)
    pgvl_root = _bootstrap_pgvl(Path(contract["pgvl_root"]))
    base_config_path = Path(contract["base_config"])
    checkpoint_dir = Path(contract["base_checkpoint_dir"])
    feature_csv = Path(contract["selected_features_csv"])
    fold = int(contract["fold"] if args.fold is None else args.fold)
    if fold < 0 or fold >= 5:
        raise ValueError("fold must be in [0, 5)")
    if "results_root" in contract:
        output_dir = Path(contract["results_root"]) / f"fold{fold}"
    elif args.fold is None:
        output_dir = Path(contract["results_dir"])
    else:
        raise ValueError("A fold override requires results_root in the contract")
    seed = int(contract["seed"])
    output_dir.mkdir(parents=True, exist_ok=True)

    from common.configuration import load_yaml_config
    base_config = load_yaml_config(base_config_path)
    split_root = Path(base_config["split_dir"]) / f"fold{fold}"
    splits = {
        phase: pd.read_csv(split_root / f"{phase}.csv")
        for phase in ("train", "val", "test")}
    features = validate_feature_table(pd.read_csv(feature_csv))
    for phase, split in splits.items():
        validate_feature_table(features, split["slide_id"])
        if set(split["label_id"].astype(int)) != {0, 1}:
            raise ValueError(f"{phase} split does not contain both NSCLC classes")

    test_predictions = _load_base_test_predictions(
        checkpoint_dir, fold, splits["test"])
    if args.check_only:
        checkpoint = checkpoint_dir / f"fold{fold}_best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        print(json.dumps({
            "status": "ready",
            "experiment": contract["experiment"],
            "feature_schema_sha256": SCHEMA_SHA256,
            "feature_table_slides": int(len(features)),
            "split_counts": {
                phase: int(len(split)) for phase, split in splits.items()},
            "checkpoint": str(checkpoint),
            "results_dir": str(output_dir),
        }, indent=2))
        return 0

    validation_cache = output_dir / f"fold{fold}_vila_validation_predictions.csv"
    if args.refresh_vila_validation or not validation_cache.is_file():
        validation_predictions = _export_validation_predictions(
            pgvl_root, base_config_path, checkpoint_dir, fold, args.device)
        _atomic_csv(validation_cache, validation_predictions)
    else:
        validation_predictions = pd.read_csv(validation_cache)
    # TME models see only the few-shot training rows. Validation chooses C and
    # the late-fusion alpha. Test probabilities are not inspected until after
    # the winning pair is fixed.
    train = splits["train"][["slide_id", "case_id", "label_id"]].rename(
        columns={"label_id": "label"}).merge(
            features, on="slide_id", how="left", validate="one_to_one")
    validation = _attach_features(validation_predictions, features)
    test = _attach_features(test_predictions, features)
    x_train = transform_nsclc_features(train)
    y_train = train["label"].to_numpy(dtype=int)
    y_validation = validation["label"].to_numpy(dtype=int)
    vila_validation = validation["probability_1"].to_numpy(dtype=float)
    vila_test = test["probability_1"].to_numpy(dtype=float)

    c_values, alphas = candidate_grid(
        contract.get("c_grid"), contract.get("alpha_grid"))
    candidates = []
    fitted = {}
    for c_value in c_values:
        model = make_tme_model(float(c_value), seed)
        model.fit(x_train, y_train)
        fitted[float(c_value)] = model
        tme_validation = model.predict_proba(
            transform_nsclc_features(validation))[:, 1]
        for alpha in alphas:
            probability = blend_probabilities(
                vila_validation, tme_validation, float(alpha))
            metrics = binary_metrics(y_validation, probability)
            candidates.append({
                "c": float(c_value),
                "alpha": float(alpha),
                "selection_key": selection_key(
                    y_validation, probability, alpha=float(alpha),
                    c_value=float(c_value)),
                "validation_metrics": metrics,
            })
    selected = max(candidates, key=lambda item: item["selection_key"])
    selected_c = selected["c"]
    selected_alpha = selected["alpha"]
    selected_model = fitted[selected_c]
    tme_validation = selected_model.predict_proba(
        transform_nsclc_features(validation))[:, 1]
    tme_test = selected_model.predict_proba(
        transform_nsclc_features(test))[:, 1]
    fusion_validation = blend_probabilities(
        vila_validation, tme_validation, selected_alpha)
    fusion_test = blend_probabilities(vila_test, tme_test, selected_alpha)
    fixed_test = blend_probabilities(vila_test, tme_test, 0.5)

    report = {
        "experiment": contract["experiment"],
        "status": "completed",
        "provenance": {
            "method": "PathoTME late-fusion extension of PGVL-Gym ViLa-MIL",
            "not_upstream_vila_mil": True,
            "base_config": str(base_config_path),
            "base_checkpoint": str(checkpoint_dir / f"fold{fold}_best.pt"),
            "base_predictions": str(
                checkpoint_dir / f"fold{fold}_predictions.csv"),
            "opentme_feature_table": str(feature_csv),
            "opentme_feature_schema_sha256": SCHEMA_SHA256,
            "opentme_transform_sha256": TRANSFORM_SHA256,
            "feature_count": len(FEATURE_NAMES),
            "selection": (
                "C and alpha selected on validation balanced accuracy; fold-0 "
                "holdout is an already-inspected diagnostic partition"),
        },
        "fold": fold,
        "shots": int(base_config["shots"]),
        "sample_counts": {
            "train": int(len(train)), "validation": int(len(validation)),
            "test": int(len(test))},
        "selected_hyperparameters": {
            "logistic_regression_c": selected_c,
            "tme_log_odds_weight_alpha": selected_alpha,
        },
        "validation": {
            "native_vila_mil": _metric_bundle(validation, vila_validation),
            "tme_only": _metric_bundle(validation, tme_validation),
            "vila_plus_tme": _metric_bundle(validation, fusion_validation),
        },
        "test": {
            "native_vila_mil": _metric_bundle(test, vila_test),
            "tme_only": _metric_bundle(test, tme_test),
            "vila_plus_tme": _metric_bundle(test, fusion_test),
            "vila_plus_tme_fixed_equal_weight": _metric_bundle(test, fixed_test),
        },
    }
    report["test_improvement_over_native"] = {
        metric: (
            report["test"]["vila_plus_tme"]["slide_metrics"][metric]
            - report["test"]["native_vila_mil"]["slide_metrics"][metric])
        for metric in ("balanced_accuracy", "macro_f1", "auroc_ovr")
    }

    model_path = output_dir / f"fold{fold}_tme_model.joblib"
    joblib.dump(selected_model, model_path)
    output_predictions = test[["slide_id", "case_id", "label"]].copy()
    output_predictions["native_vila_probability_1"] = vila_test
    output_predictions["tme_probability_1"] = tme_test
    output_predictions["vila_plus_tme_probability_1"] = fusion_test
    output_predictions["vila_plus_tme_prediction"] = (
        fusion_test >= 0.5).astype(int)
    _atomic_csv(output_dir / f"fold{fold}_predictions.csv", output_predictions)
    _atomic_json(output_dir / "metrics.json", report)

    base = report["test"]["native_vila_mil"]["slide_metrics"]
    fused = report["test"]["vila_plus_tme"]["slide_metrics"]
    print(json.dumps({
        "selected_c": selected_c,
        "selected_alpha": selected_alpha,
        "native_vila_test": base,
        "vila_plus_tme_test": fused,
        "improvement": report["test_improvement_over_native"],
        "metrics_path": str(output_dir / "metrics.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
