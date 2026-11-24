"""Frozen, label-free MUSE inference path for a native CLIP-RN50 checkpoint."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .core import TMEContext


class SparseExperts(nn.Module):
    def __init__(self, dim: int, count: int = 8, selected: int = 2) -> None:
        super().__init__()
        self.router = nn.Linear(dim, count)
        self.noise_router = nn.Linear(dim, count)
        self.experts = nn.ModuleList(nn.Linear(dim, dim) for _ in range(count))
        self.selected = selected

    def forward(self, semantics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The native model is frozen in eval mode: no training-time router noise.
        values, indices = self.router(semantics).topk(self.selected, dim=-1)
        weights = values.softmax(dim=-1)
        outputs = torch.stack([expert(semantics) for expert in self.experts], dim=1)
        selected = torch.gather(outputs, 1, indices.unsqueeze(-1).expand(-1, -1, semantics.shape[-1]))
        return selected, weights


class SemanticAggregator(nn.Module):
    def __init__(self, dim: int, heads: int = 8, top_patch_ratio: float = 0.2) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("MUSE embedding width must be divisible by its head count")
        self.num_heads = heads
        self.head_dim = dim // heads
        self.top_patch_ratio = top_patch_ratio
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, patches: torch.Tensor, cues: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        tokens, routes, dim = cues.shape
        patch_count = patches.shape[0]
        queries = cues.reshape(tokens * routes, dim)
        q = self.q_proj(queries).view(tokens * routes, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(patches).view(patch_count, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(patches).view(patch_count, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.einsum("hqd,hnd->hqn", q, k) / math.sqrt(self.head_dim)
        keep = min(patch_count, max(1, int(patch_count * self.top_patch_ratio)))
        indices = scores.topk(keep, dim=-1).indices
        keys = torch.gather(k.unsqueeze(1).expand(-1, tokens * routes, -1, -1), 2,
                            indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        values = torch.gather(v.unsqueeze(1).expand(-1, tokens * routes, -1, -1), 2,
                              indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        # Recompute unscaled scores after top-patch selection, as in native MUSE.
        attention = torch.einsum("hqd,hqnd->hqn", q, keys).softmax(dim=-1)
        attended = (attention.unsqueeze(-1) * values).sum(dim=2)
        attended = attended.transpose(0, 1).reshape(tokens * routes, dim)
        attended = self.out_proj(attended).view(tokens, routes, dim)
        routed = (attended * weights.unsqueeze(-1)).sum(dim=1)
        return routed.mean(dim=0, keepdim=True)


class FrozenMUSE(nn.Module):
    """Native MUSE weights needed by its label-free inference branch."""

    def __init__(self, input_dim: int = 1024, dim: int = 1024) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.register_buffer("class_text_features", torch.zeros(2, dim))
        self.class_context_delta = nn.Parameter(torch.zeros(2, dim))
        self.visual_adapter = nn.Sequential(
            nn.Linear(input_dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(0.25))
        self.moe = SparseExperts(dim)
        self.aggregator = SemanticAggregator(dim)
        self.classifier = nn.Linear(dim, 2)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: torch.device) -> "FrozenMUSE":
        state = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or "prompt_bank" not in state:
            raise ValueError("expected a native MUSE model state_dict with its prompt bank")
        if tuple(state["prompt_bank"].shape[:1]) != (2,):
            raise ValueError("native MUSE checkpoint is not a two-class model")
        input_dim = state["visual_adapter.0.weight"].shape[1]
        dim = state["class_text_features"].shape[1]
        if input_dim != 1024 or dim != 1024:
            raise ValueError("BRCA CLIP-RN50 MUSE requires 1024-wide patch and text features")
        model = cls(input_dim, dim)
        model.load_state_dict({k: v for k, v in state.items() if k != "prompt_bank"}, strict=True)
        return model.to(device).eval().requires_grad_(False)

    def forward(self, features: torch.Tensor, conditioner: TMEContext,
                standardized_tme: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != self.input_dim:
            raise ValueError("one nonempty 1024-wide CLIP-RN50 patch bag is required")
        semantics = self.class_text_features + self.class_context_delta
        conditioned = conditioner(semantics.unsqueeze(0), standardized_tme, scale="all").squeeze(0)
        with torch.no_grad():
            patches = self.visual_adapter(features.float())
        cues, weights = self.moe(conditioned)
        slide = self.aggregator(patches, cues, weights)
        return self.classifier(slide)
