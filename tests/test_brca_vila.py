"""Small native ViLa equations with synthetic prompts; no local weights needed."""
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,"/path/to/PGVL-Gym")
from methods.vila_mil.model import ViLa_MIL_Model
from pathotme.brca_vila import BreastGuidedViLaMIL
from pathotme.brca_features import PANELS, panel_spec


def make(panel,mode="actual"):
    torch.manual_seed(42)
    base=ViLa_MIL_Model(SimpleNamespace(input_size=16,hidden_size=8,prototype_number=3),
                        num_classes=2,prompt_features=torch.randn(4,16))
    model=BreastGuidedViLaMIL(base,panel=panel,tme_mode=mode,hidden_dim=16,dropout=0)
    width=len(panel_spec(panel)["feature_names"])
    model.standardizer.fit(np.stack([np.zeros(width),np.ones(width)]))
    return model.eval()


def bags(width):
    torch.manual_seed(7)
    return (torch.randn(5,16),torch.zeros(5,2),torch.randn(8,16),torch.zeros(8,2),
            torch.tensor([1]),torch.rand(1,width))


@pytest.mark.parametrize("panel",PANELS)
def test_native_bypass_and_zero_residual_equivalence(panel):
    m=make(panel); args=bags(m.standardizer.feature_count)
    expected=m.base(*args[:5])[0]
    torch.testing.assert_close(m(*args,bypass_conditioner=True)[0],expected)
    with torch.no_grad():m.conditioner.output_projection.weight.zero_()
    torch.testing.assert_close(m(*args)[0],expected,rtol=1e-5,atol=1e-5)


@pytest.mark.parametrize("panel",PANELS)
def test_quantitative_input_control_gradients_and_roundtrip(panel):
    m=make(panel); zero=make(panel,"zero");args=bags(m.standardizer.feature_count)
    other=(*args[:-1],args[-1]+5)
    assert not torch.allclose(m(*args)[0],m(*other)[0])
    torch.testing.assert_close(zero(*args)[0],zero(*other)[0])
    assert sum(p.numel() for p in m.parameters() if p.requires_grad)==sum(p.numel() for p in zero.parameters() if p.requires_grad)
    m.train();m(*args)[2].backward()
    assert not m.base.training and all(p.grad is None and not p.requires_grad for p in m.base.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.conditioner.parameters())
    m.eval();clone=make(panel);clone.load_adapter_state_dict(m.adapter_state_dict())
    torch.testing.assert_close(m(*args)[0],clone(*args)[0])
    details=m(*args,return_details=True);spec=panel_spec(panel)
    assert details["low_tme_group_attention"].shape[-1]==len(spec["low_tokens"])
    assert details["high_tme_group_attention"].shape[-1]==len(spec["high_tokens"])
    before=m.standardizer.mean.clone();m(*other)
    torch.testing.assert_close(before,m.standardizer.mean)
