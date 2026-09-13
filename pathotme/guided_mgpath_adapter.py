"""MGPATH-owned batch/optimizer bridge for the PathoTME extension."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from methods.mgpath.adapter import MGPathMethod
from pathotme.features import FEATURE_NAMES, transform_nsclc_features, validate_feature_table
from pathotme.guided_mgpath import TMEGuidedMGPath


def one(metadata, key):
    value = metadata[key]
    if isinstance(value, str):
        return value
    if len(value) != 1:
        raise ValueError("expected single-slide metadata")
    return str(value[0])


class TMEGuidedMGPathMethod(MGPathMethod):
    """Freeze the strict native PLIP checkpoint and train only TME parameters."""

    extension_name = "pathotme_mgpath_guided"

    def __init__(self, cfg, device="cuda:0"):
        super().__init__(cfg, device)
        if self.is_encoder_extension or self.backbone_name != "plip":
            raise ValueError("this condition requires native PLIP MGPATH")
        table = validate_feature_table(pd.read_csv(self.cfg["tme_feature_csv"]))
        transformed = transform_nsclc_features(table)
        transformed.insert(0, "slide_id", table["slide_id"].astype(str))
        self.tme = transformed.set_index("slide_id", verify_integrity=True)
        if tuple(self.tme.columns) != FEATURE_NAMES:
            raise ValueError("quantitative feature order changed")

    def build_model(self):
        base = super().build_model()
        fold = int(self.cfg["_fold_index"])
        checkpoint = Path(self.cfg["base_checkpoint_dir"]) / f"fold{fold}_best.pt"
        base.load_state_dict(torch.load(
            checkpoint, map_location=self.device, weights_only=True), strict=True)
        return TMEGuidedMGPath(
            base, tme_mode=self.cfg["tme_mode"],
            hidden_dim=self.cfg["tme_hidden_dim"],
            num_heads=self.cfg["tme_attention_heads"],
            dropout=self.cfg["tme_dropout"],
            initial_gate=self.cfg["tme_initial_gate"]).to(self.device)

    def prepare_fold(self, fold, model, train_loader):
        if fold != self.cfg["_fold_index"]:
            raise ValueError("fold mismatch")
        # Native MGPathDataset exposes frame, not ViLa's df attribute.
        slides = train_loader.dataset.frame["slide_id"].astype(str).tolist()
        values = self.tme.loc[slides, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
        model.standardizer.fit(values)

    def inputs(self, batch):
        low, low_edges, high, high_edges, label = self._unpack(batch, self.device)
        metadata = batch[-2]
        values = self.tme.loc[one(metadata, "slide_id"), list(FEATURE_NAMES)]
        tme = torch.as_tensor(values.to_numpy(dtype=np.float32),
                              device=self.device).unsqueeze(0)
        return (low, low_edges, high, high_edges, tme), label, metadata

    def train_step(self, batch, model, optimizer, loss_fn=None):
        inputs, label, _ = self.inputs(batch)
        optimizer.zero_grad(set_to_none=True)
        logits = model(*inputs)
        loss = (loss_fn or nn.CrossEntropyLoss())(logits, label)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        if any(p.grad is not None for p in model.base.parameters()):
            raise RuntimeError("frozen base received gradients")
        optimizer.step()
        return {"loss": float(loss.detach()), "logits": logits.detach(), "label": label}

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        inputs, label, _ = self.inputs(batch)
        logits = model(*inputs)
        loss = (loss_fn or nn.CrossEntropyLoss())(logits, label)
        return {"loss": float(loss), "logits": logits, "label": label}

    @torch.no_grad()
    def eval_step_with_details(self, batch, model):
        inputs, _, metadata = self.inputs(batch)
        details = model(*inputs, return_details=True)
        details["metadata"] = {k: one(metadata, k) for k in ("slide_id", "case_id")}
        return details
