"""Validation-selected late fusion for the ViLa-MIL signal canary."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def probability_to_log_odds(probability: np.ndarray) -> np.ndarray:
    """Convert positive-class probabilities to finite log odds."""
    probability = np.asarray(probability, dtype=float)
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(probability) - np.log1p(-probability)


def log_odds_to_probability(log_odds: np.ndarray) -> np.ndarray:
    """Convert log odds to numerically stable probabilities."""
    values = np.asarray(log_odds, dtype=float)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1 + exponential)
    return result


def blend_probabilities(
    vila_probability: np.ndarray,
    tme_probability: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend calibrated evidence in log-odds space.

    ``alpha=0`` is native ViLa-MIL and ``alpha=1`` is the TME-only control.
    Intermediate values constitute the late-fusion extension.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1]")
    combined = (
        (1 - alpha) * probability_to_log_odds(vila_probability)
        + alpha * probability_to_log_odds(tme_probability))
    return log_odds_to_probability(combined)


def binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    """Compute the slide-level metrics used by the canary."""
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    confidence = np.column_stack([1 - probability, probability])
    metrics = {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "nll": float(log_loss(labels, confidence, labels=[0, 1])),
        "auroc_ovr": float(roc_auc_score(labels, probability)),
        "per_class_recall": {},
    }
    for label in (0, 1):
        mask = labels == label
        metrics["per_class_recall"][str(label)] = float(
            (prediction[mask] == label).mean()) if mask.any() else None
    predicted_confidence = np.maximum(probability, 1 - probability)
    bins = np.minimum((predicted_confidence * 10).astype(int), 9)
    correctness = prediction == labels
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if mask.any():
            ece += mask.mean() * abs(
                correctness[mask].mean() - predicted_confidence[mask].mean())
    metrics["ece"] = float(ece)
    return metrics


def make_tme_model(c_value: float, seed: int) -> Pipeline:
    """Construct the fold-local imputation, scaling, and linear classifier."""
    if not math.isfinite(c_value) or c_value <= 0:
        raise ValueError("Logistic-regression C must be positive and finite")
    return Pipeline((
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c_value, max_iter=5000, class_weight="balanced",
            random_state=seed)),
    ))


def selection_key(
    labels: np.ndarray,
    probability: np.ndarray,
    *,
    alpha: float,
    c_value: float,
) -> tuple[float, float, float, float, float]:
    """Rank validation candidates without consulting the test partition."""
    metrics = binary_metrics(labels, probability)
    # Prefer primary BAcc, then AUROC and macro-F1. If all metrics tie, prefer
    # a fusion over a modality endpoint and stronger regularization.
    endpoint_distance = abs(alpha - 0.5)
    return (
        metrics["balanced_accuracy"],
        metrics["auroc_ovr"],
        metrics["macro_f1"],
        -endpoint_distance,
        -c_value,
    )


def candidate_grid(
    c_values: Iterable[float] | None = None,
    alphas: Iterable[float] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the predeclared regularization and fusion grids."""
    cs = tuple(c_values or (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0))
    weights = tuple(alphas or np.linspace(0, 1, 21).tolist())
    if not cs or not weights:
        raise ValueError("Candidate grids may not be empty")
    return cs, weights
