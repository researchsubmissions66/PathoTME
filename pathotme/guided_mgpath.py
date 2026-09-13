"""Quantitative TME-conditioned centers over a frozen PGVL MGPATH model.

This is a PathoTME extension. Native source, graphs, prompt views, projection,
normalization, and Sinkhorn scoring remain unchanged. Only center queries and
their residual term receive a slide-specific additive conditioner.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from methods.mgpath.model import sinkhorn_similarity
from pathotme.guided_vila import FoldStandardizer, TMEPrototypeConditioner


class TMEGuidedMGPath(nn.Module):
    """Wrap native MGPATH with the same 5-low/11-high quantitative tokenizer."""

    def __init__(self, base, *, tme_mode="actual", hidden_dim=128,
                 num_heads=4, dropout=0.1, initial_gate=0.1):
        super().__init__()
        if tme_mode not in {"actual", "zero"}:
            raise ValueError("tme_mode must be actual or zero")
        if base.centers.ndim != 2 or base.centers.shape[0] != 64:
            raise ValueError("expected MGPATH's 64 learned centers")
        self.base = base.requires_grad_(False).eval()
        self.standardizer = FoldStandardizer()
        self.conditioner = TMEPrototypeConditioner(
            base.centers.shape[-1], hidden_dim=hidden_dim,
            num_heads=num_heads, dropout=dropout, initial_gate=initial_gate)
        self.tme_mode = tme_mode
        # The base is frozen and always in eval mode: compute constant text
        # once, without changing its four views or class/scale order.
        with torch.no_grad():
            self.register_buffer("fixed_text", F.normalize(base.prompt(), dim=-1),
                                 persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def adapter_state_dict(self):
        return {k: v for k, v in self.state_dict().items()
                if k.startswith(("standardizer.", "conditioner."))}

    def load_adapter_state_dict(self, state):
        expected = self.adapter_state_dict()
        if set(expected) != set(state):
            raise ValueError("adapter checkpoint keys do not match")
        self.standardizer.load_state_dict({
            k.removeprefix("standardizer."): v for k, v in state.items()
            if k.startswith("standardizer.")}, strict=True)
        self.conditioner.load_state_dict({
            k.removeprefix("conditioner."): v for k, v in state.items()
            if k.startswith("conditioner.")}, strict=True)

    def _aggregate(self, features, edges, centers):
        with torch.no_grad():
            patches = self.base.project(features)
            graph = self.base.graph(patches, edges)
        scale = math.sqrt(patches.shape[-1])
        raw_weights = (centers @ patches.t() / scale).softmax(-1)
        graph_weights = (centers @ graph.t() / scale).softmax(-1)
        ratio = self.base.ratio_graph
        pooled = ((1 - ratio) * (raw_weights @ patches)
                  + ratio * (graph_weights @ graph) + centers)
        pooled = F.normalize(self.base.norm(pooled), dim=-1)
        attention = ((1 - ratio) * raw_weights + ratio * graph_weights).mean(0)
        return pooled, attention

    def forward(self, low, low_edges, high, high_edges, tme, *,
                return_details=False, bypass_conditioner=False):
        standardized = self.standardizer(tme)
        if standardized.shape[0] != 1:
            raise ValueError("MGPATH requires one slide per batch")
        if self.tme_mode == "zero":
            standardized = torch.zeros_like(standardized)
        centers = self.base.centers.unsqueeze(1)
        low_centers, low_tme_attention = self.conditioner(
            centers, standardized, scale="low")
        high_centers, high_tme_attention = self.conditioner(
            centers, standardized, scale="high")
        if bypass_conditioner:
            low_centers = high_centers = centers
        low_tokens, low_attention = self._aggregate(
            low, low_edges, low_centers.squeeze(1))
        high_tokens, high_attention = self._aggregate(
            high, high_edges, high_centers.squeeze(1))
        classes = self.base.n_classes
        logits = (sinkhorn_similarity(
            low_tokens, self.fixed_text[:, :classes], self.base.logit_scale)
            + sinkhorn_similarity(
                high_tokens, self.fixed_text[:, classes:],
                self.base.logit_scale)).unsqueeze(0)
        if not return_details:
            return logits
        return {"logits": logits, "probabilities": logits.softmax(-1),
                "low_tme_group_attention": low_tme_attention,
                "high_tme_group_attention": high_tme_attention,
                "low_patch_attention": low_attention,
                "high_patch_attention": high_attention}
