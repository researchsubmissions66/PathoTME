"""Real BRCA split and quantitative TME inputs for the MUSE adapter."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from .core import FoldPreprocessor, Panel


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"missing or duplicate CSV columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def load_tme(path: str | Path, panel: Panel) -> dict[str, torch.Tensor]:
    """Load measured values in the panel's exact column order."""
    with Path(path).expanduser().open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["slide_id", *panel.feature_names]:
            raise ValueError("TME CSV columns differ from the selected panel")
        values: dict[str, torch.Tensor] = {}
        for line, row in enumerate(reader, start=2):
            slide = row["slide_id"].strip()
            if not slide or slide in values:
                raise ValueError(f"missing or duplicate slide_id on TME line {line}")
            try:
                numbers = [float(row[name]) if row[name].strip() else float("nan")
                           for name in panel.feature_names]
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid TME number on line {line}") from error
            values[slide] = torch.tensor(numbers, dtype=torch.float64)
    if not values:
        raise ValueError(f"empty TME CSV: {path}")
    return values


def fit_fold_preprocessor(rows_by_phase: dict[str, list[dict[str, str]]],
                          tme: dict[str, torch.Tensor], panel: Panel) -> FoldPreprocessor:
    """Fit only on training slides after checking patient-disjoint assignments."""
    if set(rows_by_phase) != {"train", "val", "test"}:
        raise ValueError("train, val, and test split files are required")
    seen_slides: set[str] = set()
    patients: dict[str, tuple[str, str]] = {}
    for phase, rows in rows_by_phase.items():
        for row in rows:
            slide = row.get("slide_id", "").strip()
            patient = row.get("case_id", "").strip()
            label = row.get("label", "").strip()
            if not slide or not patient or not label:
                raise ValueError(f"{phase} row needs slide_id, case_id, and label")
            if slide in seen_slides:
                raise ValueError(f"duplicate slide in fold: {slide}")
            seen_slides.add(slide)
            if patient in patients and patients[patient] != (phase, label):
                raise ValueError(f"patient crosses phases or labels: {patient}")
            patients[patient] = (phase, label)
            if slide not in tme:
                raise ValueError(f"TME row missing for slide {slide}")
    training = torch.stack([tme[row["slide_id"]] for row in rows_by_phase["train"]])
    return FoldPreprocessor(panel).fit(training)
