"""Small no-weight tests for WSI-only deployment and attribution controls."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from methods.vila_mil.model import ViLa_MIL_Model
from pathotme.controlled_vila import donor_maps, select_fusion
from pathotme.guided_vila import FoldStandardizer
from pathotme.privileged_vila import PrivilegedStudent, auxiliary_loss, distillation_loss


def base():
    torch.manual_seed(11)
    return ViLa_MIL_Model(SimpleNamespace(input_size=16, hidden_size=8, prototype_number=3),
                          num_classes=2, prompt_features=torch.randn(4, 16))


def inputs():
    return torch.randn(5, 16), torch.zeros(5, 2), torch.randn(8, 16), torch.zeros(8, 2)


def test_student_exact_native_no_tme_and_export():
    b = base(); model = PrivilegedStudent(b).eval(); x = inputs()
    expected = b(*x, torch.tensor([0]))[0]
    torch.testing.assert_close(model(*x)["probabilities"], expected)
    torch.testing.assert_close(model(*x, auxiliary=True)["probabilities"], expected)
    clone = base(); clone.load_state_dict(copy.deepcopy(model.base.state_dict()))
    torch.testing.assert_close(clone(*x, torch.tensor([1]))[0], expected)
    assert not b.norm._forward_hooks


def test_auxiliary_gradient_reaches_shared_visual_path():
    model = PrivilegedStudent(base()); x = inputs()
    scaler = FoldStandardizer(64); scaler.fit(np.stack([np.zeros(64), np.ones(64)]))
    raw = torch.ones(1, 64); raw[0, 2] = float("nan")
    loss = auxiliary_loss(model(*x, auxiliary=True)["tme_prediction"], raw, scaler)
    loss.backward()
    assert model.base.learnable_image_center.grad.abs().sum() > 0
    assert model.base.attention_weights.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    p = torch.randn(1, 64, requires_grad=True)
    assert auxiliary_loss(p, torch.full_like(p, float("nan")), scaler).item() == 0


def test_distillation_teacher_detached():
    student = torch.randn(1, 2, requires_grad=True)
    teacher = torch.randn(1, 2, requires_grad=True)
    distillation_loss(student, teacher, 2).backward()
    assert student.grad.abs().sum() > 0 and teacher.grad is None
    assert abs(distillation_loss(teacher, teacher, 2).item()) < 1e-6


def test_shuffle_is_label_blind_bijective_split_local_patient_excluding():
    phases = {p: [{"slide_id": f"{p}{i}", "case_id": f"{p}case{i//2}", "label_id": i % 2}
                  for i in range(12)] for p in ("train", "val", "test")}
    result = donor_maps(phases, 1, 0)
    changed = copy.deepcopy(phases)
    for rows in changed.values():
        for r in rows: r["label_id"] = 500
    assert result == donor_maps(changed, 1, 0)
    for phase, mapping in result.items():
        cases = {r["slide_id"]: r["case_id"] for r in phases[phase]}
        assert set(mapping) == set(mapping.values()) == set(cases)
        assert all(cases[a] != cases[b] for a, b in mapping.items())
    with pytest.raises(ValueError):
        donor_maps({"train": [{"slide_id": "a", "case_id": "same"}, {"slide_id": "b", "case_id": "same"}]}, 1, 0)


def test_fusion_validation_selection_and_tie():
    v = np.array([[.9, .1], [.8, .2]])
    t = np.array([[.1, .9], [.1, .9]])
    assert select_fusion(v, t, [1, 1], [0, .5, 1])[0] == 1
    assert select_fusion(v, v, [0, 0], [1, .5, 0])[0] == 0
    with pytest.raises(ValueError): select_fusion(v*2, t, [0, 1], [0, 1])
