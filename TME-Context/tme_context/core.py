"""Small, model-agnostic TME conditioning components."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.resources import files
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class Group:
    name: str
    scale: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class Panel:
    """Ordered measurement and semantic-group contract."""

    name: str
    feature_names: tuple[str, ...]
    groups: tuple[Group, ...]
    percent_indices: tuple[int, ...]
    log1p_indices: tuple[int, ...]

    @classmethod
    def from_mapping(cls, name: str, data: Mapping) -> "Panel":
        features = tuple(data["feature_names"])
        if not features or len(features) != len(set(features)):
            raise ValueError("feature names must be nonempty and unique")
        index = {feature: i for i, feature in enumerate(features)}
        groups = []
        for raw in data["groups"]:
            if raw["scale"] not in {"low", "high"}:
                raise ValueError("group scale must be low or high")
            try:
                indices = tuple(index[feature] for feature in raw["features"])
            except KeyError as error:
                raise ValueError(f"unknown group feature: {error}") from error
            if not indices:
                raise ValueError("groups cannot be empty")
            groups.append(Group(str(raw["name"]), raw["scale"], indices))
        if len(groups) != 16 or len({group.name for group in groups}) != 16:
            raise ValueError("a TME-Context panel needs 16 uniquely named groups")
        scales = tuple(group.scale for group in groups)
        low_count = scales.count("low")
        if not 0 < low_count < 16 or scales != ("low",) * low_count + ("high",) * (16 - low_count):
            raise ValueError("low groups must precede high groups")
        covered = [i for group in groups for i in group.indices]
        if sorted(covered) != list(range(len(features))):
            raise ValueError("groups must cover each measurement exactly once")

        def transform_indices(key: str) -> tuple[int, ...]:
            try:
                return tuple(index[feature] for feature in data.get(key, ()))
            except KeyError as error:
                raise ValueError(f"unknown transform feature: {error}") from error

        percent = transform_indices("percent_to_fraction")
        log1p = transform_indices("log1p")
        if set(percent) & set(log1p):
            raise ValueError("a measurement cannot have both fixed transforms")
        return cls(name, features, tuple(groups), percent, log1p)

    @property
    def low_count(self) -> int:
        return sum(group.scale == "low" for group in self.groups)


def load_panel(name: str = "core62") -> Panel:
    """Load the fixed core62 or morph64 panel without any patient data."""
    data = json.loads(files("tme_context").joinpath("panels.json").read_text())
    if name not in data:
        raise ValueError(f"unknown panel {name!r}; choose from {sorted(data)}")
    return Panel.from_mapping(name, data[name])


class FoldPreprocessor(nn.Module):
    """Apply fixed unit transforms, then fit imputation and scaling on training rows."""

    def __init__(self, panel: Panel) -> None:
        super().__init__()
        self.panel = panel
        width = len(panel.feature_names)
        self.register_buffer("median", torch.zeros(width))
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.register_buffer("fitted", torch.tensor(False))

    def _fixed(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 2 or raw.shape[1] != len(self.panel.feature_names):
            raise ValueError(f"expected [slides, {len(self.panel.feature_names)}] TME measurements")
        values = raw.to(device=self.mean.device, dtype=torch.float64).clone()
        if torch.isinf(values).any() or (values[torch.isfinite(values)] < 0).any():
            raise ValueError("TME measurements must be nonnegative or NaN")
        if self.panel.percent_indices:
            values[:, self.panel.percent_indices] /= 100.0
        if self.panel.log1p_indices:
            values[:, self.panel.log1p_indices] = torch.log1p(
                values[:, self.panel.log1p_indices]
            )
        # The study transforms raw measurements before casting to float32.
        return values.to(dtype=self.mean.dtype)

    @torch.no_grad()
    def fit(self, training_raw: torch.Tensor) -> "FoldPreprocessor":
        """Fit once per fold using only training slides."""
        values = self._fixed(training_raw)
        if values.shape[0] < 2:
            raise ValueError("at least two training slides are required")
        if torch.isnan(values).all(dim=0).any():
            raise ValueError("a training measurement is missing for every slide")
        # Use the midpoint for even counts, matching NumPy's training-fold median.
        values64 = values.to(torch.float64)
        median = torch.nanquantile(values64, 0.5, dim=0)
        imputed = torch.where(torch.isnan(values64), median, values64)
        mean = imputed.mean(dim=0)
        scale = imputed.std(dim=0, unbiased=False)
        scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
        self.median.copy_(median.to(self.median.dtype))
        self.mean.copy_(mean.to(self.mean.dtype))
        self.scale.copy_(scale.to(self.scale.dtype))
        self.fitted.fill_(True)
        return self

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError("fit the preprocessor on training slides first")
        values = self._fixed(raw)
        values = torch.where(torch.isnan(values), self.median, values)
        return (values - self.mean) / self.scale


class SemanticTokenizer(nn.Module):
    """Turn the panel's 16 measurement groups into learned biological tokens."""

    def __init__(self, panel: Panel, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.panel = panel
        self.low_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(len(group.indices), hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for group in panel.groups[:panel.low_count]
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        high_groups = panel.groups[panel.low_count:]
        if any(len(group.indices) != 4 or not group.name.startswith(("cell_", "spatial_"))
               for group in high_groups):
            raise ValueError("high groups must be four-value cell or spatial groups")
        self.low_identity = nn.Parameter(torch.empty(panel.low_count, hidden_dim))
        self.high_identity = nn.Parameter(torch.empty(len(high_groups), hidden_dim))
        nn.init.normal_(self.low_identity, std=0.02)
        nn.init.normal_(self.high_identity, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, standardized_tme: torch.Tensor) -> torch.Tensor:
        if standardized_tme.ndim != 2 or standardized_tme.shape[1] != len(self.panel.feature_names):
            raise ValueError("standardized TME tensor does not match the panel")
        low = torch.stack(
            [encoder(standardized_tme[:, group.indices]) + self.low_identity[i]
             for i, (encoder, group) in enumerate(zip(
                 self.low_encoders, self.panel.groups[:self.panel.low_count]))], dim=1)
        high = torch.stack(
            [(self.cell_encoder if group.name.startswith("cell_") else self.spatial_encoder)(
                standardized_tme[:, group.indices]) + self.high_identity[i]
             for i, group in enumerate(self.panel.groups[self.panel.low_count:])], dim=1)
        return torch.cat((self.norm(low), self.norm(high)), dim=1)


class TMEContext(nn.Module):
    """Add a gated TME-attention residual to existing model queries or centers.

    The caller owns the native model and can freeze it independently. Inputs are
    [batch, queries, model_dim] and [batch, panel_features].
    """

    def __init__(
        self,
        panel: Panel,
        model_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        initial_gate: float = 0.1,
        single_scale: bool = False,
    ) -> None:
        super().__init__()
        if model_dim < 1 or hidden_dim < 1 or heads < 1 or hidden_dim % heads:
            raise ValueError("model and attention dimensions must be positive and compatible")
        if not 0 < initial_gate < 1:
            raise ValueError("initial_gate must lie between zero and one")
        self.panel = panel
        self.model_dim = model_dim
        self.single_scale = single_scale
        self.tokenizer = SemanticTokenizer(panel, hidden_dim, dropout)
        if single_scale:
            self.register_parameter("scale_embedding", None)
        else:
            self.scale_embedding = nn.Parameter(torch.empty(2, hidden_dim))
            nn.init.normal_(self.scale_embedding, std=0.02)
        self.query_projection = nn.Linear(model_dim, hidden_dim, bias=False)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, model_dim, bias=False)
        gate_logit = math.log(initial_gate / (1 - initial_gate))
        self.gate_logits = nn.Parameter(torch.full((1 if single_scale else 3,), gate_logit))

    def forward(
        self,
        queries: torch.Tensor,
        standardized_tme: torch.Tensor,
        *,
        scale: str = "all",
        return_attention: bool = False,
    ):
        if queries.ndim != 3 or queries.shape[-1] != self.model_dim:
            raise ValueError(f"queries must be [batch, queries, {self.model_dim}]")
        if standardized_tme.shape[0] != queries.shape[0]:
            raise ValueError("query and TME batch sizes differ")
        tokens = self.tokenizer(standardized_tme)
        if self.single_scale and scale != "all":
            raise ValueError("single-scale conditioning requires scale='all'")
        if scale == "low":
            assert self.scale_embedding is not None
            tokens, gate_index = tokens[:, :self.panel.low_count] + self.scale_embedding[0], 0
        elif scale == "high":
            assert self.scale_embedding is not None
            tokens, gate_index = tokens[:, self.panel.low_count:] + self.scale_embedding[1], 1
        elif scale == "all":
            gate_index = 0 if self.single_scale else 2
        else:
            raise ValueError("scale must be low, high, or all")
        context, weights = self.attention(
            self.query_projection(queries), tokens, tokens,
            need_weights=return_attention,
        )
        gate = torch.sigmoid(self.gate_logits[gate_index])
        conditioned = queries + gate * self.output_projection(context)
        return (conditioned, weights) if return_attention else conditioned
