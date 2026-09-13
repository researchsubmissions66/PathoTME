"""TME-conditioned prototype queries for the ViLa-MIL extension.

The classes in this module do not modify the vendored ViLa-MIL source.  A
frozen, checkpointed ViLa-MIL model is wrapped and its learned image centers
are conditioned on 5 low-scale and 11 high-scale quantitative phenotype tokens
before the model's native prototype-to-patch attention is evaluated.
"""
from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pathotme.features import (
    CELL_CLASSES,
    CELL_COMPOSITION,
    FEATURE_NAMES,
    GLOBAL_TISSUE,
    LOG1P_NONNEGATIVE,
    PERCENT_TO_FRACTION,
    SPATIAL_CELL_GROUPS,
    SPATIAL_INTERACTIONS,
    TLS_DERIVED,
    WHOLE_TUMOR_TISSUE,
)


GUIDED_PREPROCESSING_CONTRACT = {
    "raw_feature_count": len(FEATURE_NAMES),
    "percent_to_fraction": PERCENT_TO_FRACTION,
    "log1p_nonnegative": LOG1P_NONNEGATIVE,
    "fold_fitted": (
        "training_median_imputation_without_indicators",
        "training_mean_and_population_standard_deviation",
    ),
    "output_feature_count": len(FEATURE_NAMES),
}
GUIDED_PREPROCESSING_SHA256 = hashlib.sha256(json.dumps(
    GUIDED_PREPROCESSING_CONTRACT, separators=(",", ":"), sort_keys=False,
).encode("utf-8")).hexdigest()


class FoldStandardizer(nn.Module):
    """Store training-fold imputation and standardization statistics.

    Statistics are buffers, so an adapter-only checkpoint is self-contained.
    The transformation deliberately produces exactly 62 values: missingness
    indicators are not appended to the in-model experiment.
    """

    def __init__(self, feature_count: int = len(FEATURE_NAMES)) -> None:
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        self.feature_count = int(feature_count)
        self.register_buffer("median", torch.zeros(feature_count))
        self.register_buffer("mean", torch.zeros(feature_count))
        self.register_buffer("scale", torch.ones(feature_count))
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    def fit(self, values: np.ndarray | torch.Tensor) -> None:
        """Fit statistics from training slides only."""
        array = np.asarray(
            values.detach().cpu().numpy() if torch.is_tensor(values) else values,
            dtype=np.float64,
        )
        if array.ndim != 2 or array.shape[1] != self.feature_count:
            raise ValueError(
                f"expected [slides, {self.feature_count}] values, got {array.shape}")
        if array.shape[0] < 2:
            raise ValueError("at least two training slides are required")
        finite_or_missing = np.isfinite(array) | np.isnan(array)
        if not finite_or_missing.all():
            raise ValueError("TME values contain infinity")
        if np.isnan(array).all(axis=0).any():
            missing = np.flatnonzero(np.isnan(array).all(axis=0)).tolist()
            raise ValueError(
                "training fold has all-missing TME columns: "
                + ", ".join(map(str, missing[:8])))

        median = np.nanmedian(array, axis=0)
        imputed = np.where(np.isnan(array), median[None, :], array)
        mean = imputed.mean(axis=0)
        scale = imputed.std(axis=0, ddof=0)
        scale[scale < 1e-8] = 1.0
        for target, source in (
            (self.median, median), (self.mean, mean), (self.scale, scale)
        ):
            target.copy_(torch.as_tensor(source, dtype=target.dtype,
                                         device=target.device))
        self.fitted.fill_(True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Impute and standardize a batch using the frozen fold statistics."""
        if not bool(self.fitted.item()):
            raise RuntimeError("FoldStandardizer must be fitted before use")
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.feature_count:
            raise ValueError(
                f"expected [batch, {self.feature_count}] values, "
                f"got {tuple(values.shape)}")
        if torch.isinf(values).any():
            raise ValueError("TME values contain infinity")
        values = values.to(dtype=self.mean.dtype, device=self.mean.device)
        imputed = torch.where(torch.isnan(values), self.median, values)
        return (imputed - self.mean) / self.scale


class SemanticTMETokenizer(nn.Module):
    """Encode the 62 measurements as 5 low- and 11 high-scale tokens.

    Cell-type tokens share an encoder, as do spatial-interaction tokens. This
    retains phenotype identity through small learned embeddings without using
    textual feature names as model inputs.
    """

    LOW_TOKEN_NAMES = (
        "global_tissue",
        "compartment_extent",
        "tumor_core_composition",
        "invasive_margin_composition",
        "tls",
    )
    HIGH_TOKEN_NAMES = tuple(
        f"cell_{name.lower()}" for name in CELL_CLASSES
    ) + tuple(
        f"spatial_{name.lower()}" for name in SPATIAL_CELL_GROUPS)

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        index = {name: position for position, name in enumerate(FEATURE_NAMES)}
        low_columns = (
            GLOBAL_TISSUE,
            WHOLE_TUMOR_TISSUE[:3],
            WHOLE_TUMOR_TISSUE[3:6],
            WHOLE_TUMOR_TISSUE[6:8],
            TLS_DERIVED,
        )
        cell_columns = tuple(tuple(
            name for name in CELL_COMPOSITION if f"_{cell}_IN_" in name
        ) for cell in CELL_CLASSES)
        spatial_columns = tuple(tuple(
            name for name in SPATIAL_INTERACTIONS
            if f"_OF_{cell_group}_AROUND_" in name
        ) for cell_group in SPATIAL_CELL_GROUPS)
        all_columns = tuple(
            name for group in (*low_columns, *cell_columns, *spatial_columns)
            for name in group)
        if len(all_columns) != len(FEATURE_NAMES) \
                or set(all_columns) != set(FEATURE_NAMES):
            raise ValueError(
                "semantic TME token definitions must cover every feature once")
        if any(len(group) != 4 for group in cell_columns + spatial_columns):
            raise ValueError("cell and spatial TME tokens must each contain four values")

        self.low_indices = tuple(tuple(index[name] for name in group)
                                 for group in low_columns)
        self.cell_indices = tuple(tuple(index[name] for name in group)
                                  for group in cell_columns)
        self.spatial_indices = tuple(tuple(index[name] for name in group)
                                     for group in spatial_columns)

        def encoder(width: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(width, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.low_encoders = nn.ModuleList(
            encoder(len(columns)) for columns in low_columns)
        self.cell_encoder = encoder(4)
        self.spatial_encoder = encoder(4)
        self.low_identity = nn.Parameter(
            torch.empty(len(low_columns), hidden_dim))
        self.cell_identity = nn.Parameter(
            torch.empty(len(cell_columns), hidden_dim))
        self.spatial_identity = nn.Parameter(
            torch.empty(len(spatial_columns), hidden_dim))
        for identity in (
            self.low_identity, self.cell_identity, self.spatial_identity,
        ):
            nn.init.normal_(identity, std=0.02)
        self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _select(values: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        return values[:, list(indices)]

    def forward(
        self, standardized_tme: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[5, batch, hidden]`` and ``[11, batch, hidden]`` tokens."""
        if standardized_tme.ndim != 2 \
                or standardized_tme.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"expected [batch, {len(FEATURE_NAMES)}] standardized TME values")
        low = torch.stack([
            encoder(self._select(standardized_tme, indices))
            + self.low_identity[position]
            for position, (encoder, indices) in enumerate(
                zip(self.low_encoders, self.low_indices))
        ], dim=0)
        cell = torch.stack([
            self.cell_encoder(self._select(standardized_tme, indices))
            + self.cell_identity[position]
            for position, indices in enumerate(self.cell_indices)
        ], dim=0)
        spatial = torch.stack([
            self.spatial_encoder(self._select(standardized_tme, indices))
            + self.spatial_identity[position]
            for position, indices in enumerate(self.spatial_indices)
        ], dim=0)
        return self.norm(low), self.norm(torch.cat((cell, spatial), dim=0))


class TMEPrototypeConditioner(nn.Module):
    """Cross-attend ViLa image centers to semantic TME tokens.

    Five tissue/TLS tokens condition the low-magnification centers and eleven
    cell/spatial tokens condition the high-magnification centers. Attention
    runs in a low-dimensional adapter space to limit 16-shot capacity.
    """

    def __init__(
        self,
        model_dim: int,
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        initial_gate: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim < 1 or hidden_dim < 1:
            raise ValueError("model_dim and hidden_dim must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must lie strictly between zero and one")
        self.model_dim = int(model_dim)
        self.hidden_dim = int(hidden_dim)
        self.tokenizer = SemanticTMETokenizer(hidden_dim, dropout=dropout)
        self.scale_embedding = nn.Parameter(torch.empty(2, hidden_dim))
        nn.init.normal_(self.scale_embedding, std=0.02)

        self.center_projection = nn.Linear(model_dim, hidden_dim, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        self.output_projection = nn.Linear(hidden_dim, model_dim, bias=False)
        gate_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.gate_logits = nn.Parameter(torch.full((2,), gate_logit))

    def forward(
        self,
        centers: torch.Tensor,
        standardized_tme: torch.Tensor,
        *,
        scale: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return phenotype-conditioned centers and group attention weights."""
        if centers.ndim != 3 or centers.shape[2] != self.model_dim:
            raise ValueError(
                f"centers must be [prototypes, batch, {self.model_dim}]")
        if centers.shape[1] != standardized_tme.shape[0]:
            raise ValueError("center and TME batch sizes do not match")
        low_tokens, high_tokens = self.tokenizer(standardized_tme)
        if scale == "low":
            keys, scale_index = low_tokens, 0
        elif scale == "high":
            keys, scale_index = high_tokens, 1
        else:
            raise ValueError("scale must be 'low' or 'high'")

        scale_embedding = self.scale_embedding[scale_index].view(1, 1, -1)
        keys = keys + scale_embedding
        query = self.center_projection(centers)
        attended, weights = self.cross_attention(
            query, keys, keys, need_weights=True, average_attn_weights=True)
        delta = self.output_projection(attended)
        gate = torch.sigmoid(self.gate_logits[scale_index])
        return centers + gate * delta, weights


class TMEGuidedViLaMIL(nn.Module):
    """Wrap a frozen native ViLa-MIL checkpoint with TME cross-attention."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        feature_count: int = len(FEATURE_NAMES),
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        initial_gate: float = 0.1,
        tme_mode: str = "actual",
    ) -> None:
        super().__init__()
        if tme_mode not in {"actual", "zero"}:
            raise ValueError("tme_mode must be 'actual' or 'zero'")
        if not hasattr(base_model, "learnable_image_center"):
            raise TypeError("base_model is not a ViLa-MIL model")
        self.base = base_model
        self.base.requires_grad_(False)
        self.standardizer = FoldStandardizer(feature_count)
        self.conditioner = TMEPrototypeConditioner(
            int(base_model.learnable_image_center.shape[-1]),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            initial_gate=initial_gate,
        )
        self.tme_mode = tme_mode

    def train(self, mode: bool = True):
        """Keep the checkpointed ViLa branch in evaluation mode."""
        super().train(mode)
        self.base.eval()
        return self

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only extension parameters and fold preprocessing buffers."""
        prefixes = ("standardizer.", "conditioner.")
        return {
            key: value for key, value in self.state_dict().items()
            if key.startswith(prefixes)
        }

    def load_adapter_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load a strict adapter-only checkpoint without replacing ViLa."""
        expected = set(self.adapter_state_dict())
        provided = set(state)
        if expected != provided:
            missing = sorted(expected - provided)
            unexpected = sorted(provided - expected)
            raise RuntimeError(
                f"adapter checkpoint mismatch; missing={missing}, "
                f"unexpected={unexpected}")
        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current, strict=True)

    def forward(
        self,
        x_s: torch.Tensor,
        coord_s: torch.Tensor,
        x_l: torch.Tensor,
        coords_l: torch.Tensor,
        label: torch.Tensor,
        tme_values: torch.Tensor,
        *,
        return_details: bool = False,
    ):
        del coord_s, coords_l
        base = self.base
        prompts = base.prompt_learner()
        tokenized_prompts = base.prompt_learner.tokenized_prompts
        text_features = base.text_encoder(prompts, tokenized_prompts)

        standardized = self.standardizer(tme_values)
        if self.tme_mode == "zero":
            standardized = torch.zeros_like(standardized)
        centers_low, tme_attention_low = self.conditioner(
            base.learnable_image_center, standardized, scale="low")
        centers_high, tme_attention_high = self.conditioner(
            base.learnable_image_center, standardized, scale="high")

        patches_low = x_s.float()
        components_low, _ = base.cross_attention_1(
            centers_low, patches_low, patches_low)
        components_low = base.norm(components_low + centers_low)

        patches_high = x_l.float()
        components_high, _ = base.cross_attention_1(
            centers_high, patches_high, patches_high)
        components_high = base.norm(components_high + centers_high)

        hidden_low = components_low.squeeze(1).float()
        attention_low = base.attention_weights(
            base.attention_V(hidden_low) * base.attention_U(hidden_low))
        attention_low = F.softmax(attention_low.transpose(1, 0), dim=1)
        image_features_low = torch.mm(attention_low, hidden_low)

        hidden_high = components_high.squeeze(1).float()
        attention_high = base.attention_weights(
            base.attention_V(hidden_high) * base.attention_U(hidden_high))
        attention_high = F.softmax(attention_high.transpose(1, 0), dim=1)
        image_features_high = torch.mm(attention_high, hidden_high)

        text_features_low = text_features[:base.num_classes]
        image_context_low = torch.cat(
            (components_low.squeeze(1), patches_low), dim=0)
        text_context_low, patch_attention_low = base.cross_attention_2(
            text_features_low.unsqueeze(1), image_context_low,
            image_context_low)
        text_features_low = text_context_low.squeeze(1) + text_features_low

        text_features_high = text_features[base.num_classes:]
        image_context_high = torch.cat(
            (components_high.squeeze(1), patches_high), dim=0)
        text_context_high, patch_attention_high = base.cross_attention_2(
            text_features_high.unsqueeze(1), image_context_high,
            image_context_high)
        text_features_high = text_context_high.squeeze(1) + text_features_high

        logits_low = image_features_low @ text_features_low.T
        logits_high = image_features_high @ text_features_high.T
        logits = logits_low + logits_high
        loss = base.loss_ce(logits, label)
        probabilities = F.softmax(logits, dim=1)
        prediction = torch.topk(probabilities, 1, dim=1)[1]
        if not return_details:
            return probabilities, prediction, loss

        prototype_count = int(base.learnable_image_center.shape[0])

        def patch_part(attention: torch.Tensor, patch_count: int) -> torch.Tensor:
            attention = attention.softmax(dim=-1)
            if attention.ndim == 4:
                attention = attention.mean(dim=1)
            if attention.ndim == 3 and attention.shape[0] == 1:
                attention = attention.squeeze(0)
            expected = prototype_count + patch_count
            if attention.ndim != 2 or attention.shape[1] != expected:
                raise ValueError(
                    "ViLa context attention has unexpected shape "
                    f"{tuple(attention.shape)}; expected class x {expected}")
            return attention[:, prototype_count:]

        return {
            "logits": logits,
            "probabilities": probabilities,
            "prediction": prediction,
            "loss": loss,
            "low_patch_attention": patch_part(
                patch_attention_low, patches_low.shape[0]),
            "high_patch_attention": patch_part(
                patch_attention_high, patches_high.shape[0]),
            "low_tme_group_attention": tme_attention_low,
            "high_tme_group_attention": tme_attention_high,
            "conditioner_gates": torch.sigmoid(
                self.conditioner.gate_logits),
        }
