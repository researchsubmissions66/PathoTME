"""No private features/weights: test the actual MGPATH equations at tiny width."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/path/to/PGVL-Gym")

import numpy as np
import pytest
import torch
from torch import nn

from methods.mgpath.model import MGPathModel
from pathotme.guided_mgpath import TMEGuidedMGPath


class FixedPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("texts", torch.randn(4, 4, 16))

    def forward(self):
        return self.texts


def make(mode="actual"):
    torch.manual_seed(23)
    base = MGPathModel(None, None, [], n_classes=2, source_dim=16,
                       shared_dim=16, prompt_module=FixedPrompt(),
                       projector=nn.Identity(), logit_scale=torch.tensor(1.0))
    model = TMEGuidedMGPath(base, tme_mode=mode, hidden_dim=16, dropout=0)
    model.standardizer.fit(np.stack([np.zeros(62), np.ones(62)]))
    return model.eval()


def inputs():
    torch.manual_seed(9)
    low, high = torch.randn(5, 16), torch.randn(7, 16)
    return low, torch.arange(5).repeat(2, 1), high, torch.arange(7).repeat(2, 1)


def test_bypass_and_zero_residual_reproduce_native():
    model = make(); bags = inputs(); tme = torch.rand(1, 62)
    expected = model.base(*bags)
    torch.testing.assert_close(model(*bags, tme, bypass_conditioner=True), expected)
    with torch.no_grad(): model.conditioner.output_projection.weight.zero_()
    torch.testing.assert_close(model(*bags, tme), expected)


def test_zero_control_is_invariant_to_slide_values():
    model = make("zero"); bags = inputs()
    torch.testing.assert_close(model(*bags, torch.zeros(1, 62)),
                               model(*bags, torch.ones(1, 62)))


def test_real_values_affect_logits_and_attention_has_native_token_roles():
    model = make(); bags = inputs()
    a = model(*bags, torch.zeros(1, 62), return_details=True)
    b = model(*bags, torch.ones(1, 62))
    assert not torch.allclose(a["logits"], b)
    assert a["low_tme_group_attention"].shape == (1, 64, 5)
    assert a["high_tme_group_attention"].shape == (1, 64, 11)
    assert a["low_patch_attention"].shape == (5,)
    assert a["high_patch_attention"].shape == (7,)
    assert torch.isfinite(a["logits"]).all()


def test_only_adapter_receives_gradients_and_base_stays_eval():
    model = make().train(); before = copy.deepcopy(model.base.state_dict())
    model(*inputs(), torch.randn(1, 62)).square().sum().backward()
    assert not model.base.training
    assert all(p.grad is None and not p.requires_grad for p in model.base.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.conditioner.parameters())
    for k, v in model.base.state_dict().items(): torch.testing.assert_close(v, before[k])


def test_adapter_round_trip_does_not_serialize_base_and_modes_match_capacity():
    model = make(); restored = make(); state = model.adapter_state_dict()
    assert not any(k.startswith("base.") for k in state)
    restored.load_adapter_state_dict(state)
    tme = torch.randn(1, 62)
    torch.testing.assert_close(model(*inputs(), tme), restored(*inputs(), tme))
    zero = make("zero")
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == sum(
        p.numel() for p in zero.parameters() if p.requires_grad)
    with pytest.raises(ValueError): restored.load_adapter_state_dict({})


def test_standardization_requires_fit_and_uses_training_statistics():
    model = make(); median = model.standardizer.median.clone()
    model.standardizer(torch.full((1, 62), 1000.0))
    torch.testing.assert_close(model.standardizer.median, median)
    with pytest.raises(ValueError): model(*inputs(), torch.zeros(2, 62))
