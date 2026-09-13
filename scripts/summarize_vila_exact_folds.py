#!/usr/bin/env python3
"""Summarize paired ViLa-MIL/TME evidence over exact-coverage folds."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pathotme.fusion import binary_metrics


METRICS = ("balanced_accuracy", "macro_f1", "auroc_ovr")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _patient_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "label": "first",
        "native_vila_probability_1": "mean",
        "tme_probability_1": "mean",
        "vila_plus_tme_probability_1": "mean",
    }
    inconsistent = frame.groupby("case_id")["label"].nunique()
    if (inconsistent > 1).any():
        raise ValueError("A patient has conflicting labels")
    return frame.groupby("case_id", sort=True).agg(columns).reset_index()


def _stratified_bootstrap(
    patients: pd.DataFrame, *, samples: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    by_class = {
        label: patients.index[patients["label"] == label].to_numpy()
        for label in (0, 1)}
    if any(len(indices) == 0 for indices in by_class.values()):
        raise ValueError("Both classes are required for stratified bootstrap")
    deltas = {metric: [] for metric in METRICS}
    for _ in range(samples):
        indices = np.concatenate([
            rng.choice(group, size=len(group), replace=True)
            for group in by_class.values()])
        sample = patients.loc[indices]
        labels = sample["label"].to_numpy(dtype=int)
        native = binary_metrics(
            labels, sample["native_vila_probability_1"].to_numpy())
        fused = binary_metrics(
            labels, sample["vila_plus_tme_probability_1"].to_numpy())
        for metric in METRICS:
            deltas[metric].append(fused[metric] - native[metric])
    return {
        metric: {
            "mean_delta": float(np.mean(values)),
            "ci95": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5))],
            "probability_delta_gt_zero": float(np.mean(np.asarray(values) > 0)),
        }
        for metric, values in deltas.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 2, 3])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args(argv)

    fold_reports = []
    frames = []
    for fold in args.folds:
        fold_root = args.results_root / f"fold{fold}"
        metrics_path = fold_root / "metrics.json"
        predictions_path = fold_root / f"fold{fold}_predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(
                f"Fold {fold} is incomplete under {fold_root}")
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        native = report["test"]["native_vila_mil"]["slide_metrics"]
        tme = report["test"]["tme_only"]["slide_metrics"]
        fused = report["test"]["vila_plus_tme"]["slide_metrics"]
        fold_reports.append({
            "fold": fold,
            "selected_hyperparameters": report["selected_hyperparameters"],
            "native_vila_mil": native,
            "tme_only": tme,
            "vila_plus_tme": fused,
            "delta_vila_plus_tme_minus_native": {
                metric: fused[metric] - native[metric] for metric in METRICS},
        })
        frame = pd.read_csv(predictions_path)
        frame.insert(0, "fold", fold)
        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)
    fold_membership = predictions.groupby("case_id")["fold"].nunique()
    if (fold_membership > 1).any():
        raise ValueError(
            "Exact-fold holdout patients overlap; pooled evidence is invalid")
    patients = _patient_frame(predictions)
    native = binary_metrics(
        patients["label"].to_numpy(),
        patients["native_vila_probability_1"].to_numpy())
    tme = binary_metrics(
        patients["label"].to_numpy(), patients["tme_probability_1"].to_numpy())
    fused = binary_metrics(
        patients["label"].to_numpy(),
        patients["vila_plus_tme_probability_1"].to_numpy())
    summary = {
        "status": "completed",
        "scope": "diagnostic exact-coverage folds; not a final five-fold result",
        "folds": fold_reports,
        "mean_fold_metrics": {
            condition: {
                metric: float(np.mean([
                    fold[condition][metric] for fold in fold_reports]))
                for metric in METRICS}
            for condition in ("native_vila_mil", "tme_only", "vila_plus_tme")
        },
        "pooled_patient_metrics": {
            "native_vila_mil": native,
            "tme_only": tme,
            "vila_plus_tme": fused,
        },
        "paired_patient_bootstrap": _stratified_bootstrap(
            patients, samples=args.bootstrap_samples, seed=args.seed),
        "patient_count": int(len(patients)),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    _atomic_json(args.results_root / "exact_folds_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
