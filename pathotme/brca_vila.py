"""BRCA panel-owned tokenizer over the unchanged native ViLa visual equations."""
from __future__ import annotations

import torch
from torch import nn

from pathotme.brca_features import panel_spec
from pathotme.guided_vila import TMEGuidedViLaMIL


class BreastTMETokenizer(nn.Module):
    """Encode morphology-specific low tokens and shared-family high tokens."""

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        spec = panel_spec("brca_morph64_v1")
        self.LOW_TOKEN_NAMES = tuple(spec["low_tokens"])
        self.HIGH_TOKEN_NAMES = tuple(spec["high_tokens"])
        index = {c: i for i, c in enumerate(spec["feature_names"])}
        self.low_indices = [tuple(index[c] for c in cols) for cols in spec["low_tokens"].values()]
        self.high_indices = [tuple(index[c] for c in cols) for cols in spec["high_tokens"].values()]
        def encoder(width):
            return nn.Sequential(nn.Linear(width, hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.low_encoders = nn.ModuleList(encoder(len(cols)) for cols in self.low_indices)
        self.cell_encoder = encoder(4)
        self.spatial_encoder = encoder(4)
        self.low_identity = nn.Parameter(torch.empty(len(self.low_indices), hidden_dim))
        self.high_identity = nn.Parameter(torch.empty(len(self.high_indices), hidden_dim))
        nn.init.normal_(self.low_identity, std=0.02)
        nn.init.normal_(self.high_identity, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, values):
        """Return six low and ten high tokens; each input is an actual number."""
        if values.ndim != 2 or values.shape[1] != 64:
            raise ValueError("BRCA morphology tokenizer requires [batch,64]")
        low = torch.stack([encoder(values[:, list(cols)]) + self.low_identity[i]
                           for i, (encoder, cols) in enumerate(zip(self.low_encoders, self.low_indices))])
        high = torch.stack([(self.cell_encoder if name.startswith("cell_") else self.spatial_encoder)(
            values[:, list(cols)]) + self.high_identity[i]
            for i, (name, cols) in enumerate(zip(self.HIGH_TOKEN_NAMES, self.high_indices))])
        return self.norm(low), self.norm(high)


class BreastGuidedViLaMIL(TMEGuidedViLaMIL):
    """Keep native ViLa frozen; change only the separately versioned TME path."""

    def __init__(self, base_model, *, panel, **kwargs):
        self.panel = panel
        spec = panel_spec(panel)
        super().__init__(base_model, feature_count=len(spec["feature_names"]), **kwargs)
        if panel == "brca_morph64_v1":
            self.conditioner.tokenizer = BreastTMETokenizer(
                self.conditioner.hidden_dim, dropout=kwargs.get("dropout", 0.1))

    def forward(self, *inputs, bypass_conditioner=False, **kwargs):
        """Support a structural native-bypass check on a training bag only."""
        if bypass_conditioner:
            return self.base(*inputs[:5], **kwargs)
        return super().forward(*inputs, **kwargs)
