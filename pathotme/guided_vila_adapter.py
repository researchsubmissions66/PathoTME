"""PGVL-Gym bridge for the TME-guided ViLa-MIL experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pathotme.features import (
    FEATURE_NAMES,
    transform_nsclc_features,
    validate_feature_table,
)
from pathotme.guided_vila import TMEGuidedViLaMIL

from methods.base import probabilities_to_logits
from methods.vila_mil.adapter import ViLaMILMethod


def _load_torch_state(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch compatibility
        return torch.load(path, map_location=device)


def _one_metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata[key]
    if isinstance(value, str):
        return value
    if hasattr(value, "__len__") and len(value) == 1:
        return str(value[0])
    raise ValueError(f"expected one {key} per batch, got {value!r}")


class TMEGuidedViLaMethod(ViLaMILMethod):
    """Train only a TME cross-attention adapter over frozen native ViLa-MIL."""

    # Keep the loader dispatch at the native method name. Public provenance is
    # carried separately by ``extension_name`` and the experiment contract.
    name = "vila_mil"
    extension_name = "pathotme_vila_guided"

    def __init__(self, cfg, device: str = "cuda") -> None:
        super().__init__(cfg, device)
        feature_path = Path(self.cfg["tme_feature_csv"])
        feature_table = validate_feature_table(pd.read_csv(feature_path))
        transformed = transform_nsclc_features(feature_table)
        transformed.insert(0, "slide_id", feature_table["slide_id"].astype(str))
        self._tme = transformed.set_index("slide_id", verify_integrity=True)
        if tuple(self._tme.columns) != FEATURE_NAMES:
            raise ValueError("transformed TME table lost the pinned feature order")
        self.fold = int(self.cfg["_fold_index"])

    def build_model(self) -> TMEGuidedViLaMIL:
        base = super().build_model()
        checkpoint = Path(self.cfg["base_checkpoint_dir"]) / (
            f"fold{self.fold}_best.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"native ViLa-MIL fold checkpoint is missing: {checkpoint}")
        base.load_state_dict(_load_torch_state(checkpoint, self.device), strict=True)
        model = TMEGuidedViLaMIL(
            base,
            hidden_dim=int(self.cfg.get("tme_hidden_dim", 128)),
            num_heads=int(self.cfg.get("tme_attention_heads", 4)),
            dropout=float(self.cfg.get("tme_dropout", 0.1)),
            initial_gate=float(self.cfg.get("tme_initial_gate", 0.1)),
            tme_mode=str(self.cfg.get("tme_mode", "actual")),
        )
        return model.to(self.device)

    def prepare_fold(self, fold: int, model: TMEGuidedViLaMIL,
                     train_loader: Any) -> None:
        if fold != self.fold:
            raise ValueError(f"configured fold {self.fold} received fold {fold}")
        table = getattr(train_loader.dataset, "df", None)
        if table is None or "slide_id" not in table:
            raise TypeError("ViLa training dataset does not expose slide identities")
        slide_ids = table["slide_id"].astype(str).tolist()
        missing = sorted(set(slide_ids) - set(self._tme.index))
        if missing:
            raise ValueError(
                "OpenTME is missing training slides: " + ", ".join(missing[:5]))
        values = self._tme.loc[slide_ids, list(FEATURE_NAMES)].to_numpy(
            dtype=np.float32)
        model.standardizer.fit(values)

    def _tme_values(self, metadata: dict[str, Any]) -> torch.Tensor:
        slide_id = _one_metadata_value(metadata, "slide_id")
        if slide_id not in self._tme.index:
            raise KeyError(f"OpenTME has no quantitative row for {slide_id}")
        values = self._tme.loc[slide_id, list(FEATURE_NAMES)].to_numpy(
            dtype=np.float32)
        return torch.as_tensor(values, device=self.device).unsqueeze(0)

    def _inputs(self, batch, model: TMEGuidedViLaMIL):
        if len(batch) != 4 or not isinstance(batch[-2], dict):
            raise ValueError(
                "TME-guided ViLa requires [low, high, metadata, label] batches")
        x_s, x_l, metadata, label = batch
        x_s = self._slide_bag(x_s.to(self.device))
        x_l = self._slide_bag(x_l.to(self.device))
        if self.is_encoder_extension:
            raise ValueError(
                "The first TME-guided experiment is restricted to native "
                "CLIP-RN50 ViLa-MIL")
        label = label.to(self.device)
        coord_s = torch.zeros(x_s.shape[0], 2, device=self.device)
        coord_l = torch.zeros(x_l.shape[0], 2, device=self.device)
        return (x_s, coord_s, x_l, coord_l, label,
                self._tme_values(metadata)), metadata

    def train_step(self, batch, model, optimizer, loss_fn):
        del loss_fn
        inputs, _metadata = self._inputs(batch, model)
        optimizer.zero_grad(set_to_none=True)
        probabilities, _prediction, loss = model(*inputs)
        logits = probabilities_to_logits(probabilities)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "logits": logits.detach(),
                "label": inputs[4]}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        del loss_fn
        inputs, _metadata = self._inputs(batch, model)
        probabilities, _prediction, loss = model(*inputs)
        return {
            "loss": loss.item(),
            "logits": probabilities_to_logits(probabilities),
            "label": inputs[4],
        }

    @torch.no_grad()
    def eval_step_with_details(self, batch, model):
        """Evaluate one slide and expose semantic TME-token attention."""
        inputs, metadata = self._inputs(batch, model)
        details = model(*inputs, return_details=True)
        details["metadata"] = {
            "slide_id": _one_metadata_value(metadata, "slide_id"),
            "case_id": _one_metadata_value(metadata, "case_id"),
        }
        return details
