"""Training-only TME supervision; deployed students have a WSI-only API.

No vendored ViLa equations are changed. Auxiliary features are the existing
low/high attention-pooled visual representations captured from its native
LayerNorm outputs. The auxiliary head is absent from exported inference state.
"""
import torch
from torch import nn
from torch.nn import functional as F

from common.models.paired_encoder_extension import project_paired_features
from pathotme.brca_vila_adapter import BreastGuidedViLaMethod
from pathotme.guided_vila_adapter import _one_metadata_value


def visual_inputs(batch, base, device):
    """Project the declared paired features exactly as the PGVL adapter does."""
    if len(batch) != 4 or not isinstance(batch[2], dict):
        raise ValueError("requires low/high/metadata/label batch")
    low, high = (BreastGuidedViLaMethod._slide_bag(x.to(device)) for x in batch[:2])
    low, high = (project_paired_features(base, x) for x in (low, high))
    return (low, torch.zeros(len(low), 2, device=device), high,
            torch.zeros(len(high), 2, device=device), batch[-1].to(device))


class ControlledGuidedMethod(BreastGuidedViLaMethod):
    """Explicit CONCH port retaining its shared-space projection and prompts."""

    def __init__(self, cfg, device="cuda:0", donors=None):
        if cfg["backbone"] != "conch" or cfg["feature_space_id"] != "hf:MahmoodLab/conch":
            raise ValueError("this controlled experiment requires CONCH shared features")
        super().__init__(cfg, device)
        self.donors = donors or {}

    def _tme_values(self, metadata):
        slide = _one_metadata_value(metadata, "slide_id")
        return super()._tme_values({"slide_id": self.donors.get(slide, slide)})

    def _inputs(self, batch, model):
        return (*visual_inputs(batch, model.base, self.device), self._tme_values(batch[2])), batch[2]


class PrivilegedStudent(nn.Module):
    """Native visual classifier with an optional, training-only regression head."""

    def __init__(self, base, feature_count=64, hidden_dim=128):
        super().__init__()
        self.base = base
        width = base.learnable_image_center.shape[-1]
        self.auxiliary = nn.Sequential(nn.Linear(2*width, hidden_dim), nn.GELU(),
                                       nn.Linear(hidden_dim, feature_count))

    def forward(self, low, coord_low, high, coord_high, label=None, *, auxiliary=False):
        """Predict from visual bags only; no TME input is accepted."""
        if label is None:
            label = torch.zeros(1, dtype=torch.long, device=low.device)
        captured = []
        hook = self.base.norm.register_forward_hook(lambda _m, _i, out: captured.append(out)) if auxiliary else None
        try:
            details = self.base(low, coord_low, high, coord_high, label, return_details=True)
        finally:
            if hook is not None:
                hook.remove()
        if auxiliary:
            if len(captured) != 2:
                raise RuntimeError("native two-scale pooling contract changed")
            pooled = []
            for components in captured:
                h = components.squeeze(1).float()
                a = self.base.attention_weights(self.base.attention_V(h)*self.base.attention_U(h))
                pooled.append(a.T.softmax(1) @ h)
            details["tme_prediction"] = self.auxiliary(torch.cat(pooled, dim=1))
        return details


def auxiliary_loss(prediction, raw_target, standardizer):
    """Masked Huber loss on train-standardized observed quantitative values."""
    mask = torch.isfinite(raw_target)
    if not mask.any():
        return prediction.sum()*0
    target = standardizer(raw_target).detach()
    return F.smooth_l1_loss(prediction[mask], target[mask])


def distillation_loss(student_logits, teacher_logits, temperature):
    """KL(teacher || student), retaining conventional temperature-squared scale."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature**2 * F.kl_div(
        F.log_softmax(student_logits/temperature, dim=-1),
        F.softmax(teacher_logits.detach()/temperature, dim=-1), reduction="batchmean")
