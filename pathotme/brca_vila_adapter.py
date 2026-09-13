"""Breast panel integration; native prompts, feature bags and labels are retained."""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from methods.vila_mil.adapter import ViLaMILMethod
from pathotme.guided_vila_adapter import TMEGuidedViLaMethod, _load_torch_state
from pathotme.brca_features import panel_spec, transform_rows
from pathotme.brca_vila import BreastGuidedViLaMIL


class BreastGuidedViLaMethod(TMEGuidedViLaMethod):
    """Reuse the native batch bridge, replacing only the panel-specific pieces."""

    extension_name = "pathotme_brca_vila_guided"

    def __init__(self, cfg, device="cuda:0"):
        ViLaMILMethod.__init__(self, cfg, device)
        self.panel = cfg["tme_panel"]
        self.names = panel_spec(self.panel)["feature_names"]
        with Path(cfg["tme_feature_csv"]).open() as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["slide_id", *self.names]:
                raise ValueError("panel feature order mismatch")
            rows = list(reader)
        values = transform_rows(rows, self.panel)
        self._tme = pd.DataFrame(values, columns=self.names,
                                 index=[r["slide_id"] for r in rows])
        if self._tme.index.has_duplicates:
            raise ValueError("duplicate breast TME slide IDs")
        self.fold = int(cfg["_fold_index"])

    def build_model(self):
        base = ViLaMILMethod.build_model(self)
        path = Path(self.cfg["base_checkpoint_dir"]) / f"fold{self.fold}_best.pt"
        base.load_state_dict(_load_torch_state(path, self.device), strict=True)
        return BreastGuidedViLaMIL(
            base, panel=self.panel, tme_mode=self.cfg["tme_mode"],
            hidden_dim=self.cfg["tme_hidden_dim"], num_heads=self.cfg["tme_attention_heads"],
            dropout=self.cfg["tme_dropout"], initial_gate=self.cfg["tme_initial_gate"]
        ).to(self.device)

    def prepare_fold(self, fold, model, train_loader):
        if fold != self.fold:
            raise ValueError("fold mismatch")
        ids = train_loader.dataset.df["slide_id"].astype(str).tolist()
        model.standardizer.fit(self._tme.loc[ids, list(self.names)].to_numpy(dtype=np.float32))

    def _tme_values(self, metadata):
        ids = metadata["slide_id"]
        slide = ids if isinstance(ids, str) else ids[0]
        if not isinstance(ids, str) and len(ids) != 1:
            raise ValueError("single-slide batches required")
        return torch.as_tensor(self._tme.loc[slide, list(self.names)].to_numpy(dtype=np.float32),
                               device=self.device).unsqueeze(0)
