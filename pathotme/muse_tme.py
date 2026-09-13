"""TME conditioning of MUSE inference semantics before sparse expert routing."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from methods.muse.adapter import MUSEMethod
from pathotme.focus_tme import TMEQueryConditioner
from pathotme.muse_data import load_tme
from pathotme.guided_vila import FoldStandardizer
from pathotme.muse_contract import PORT, validate_config


class MUSENativeMethod(MUSEMethod):
    """Use the unchanged registered MUSE native training and evaluation recipe."""
    def __init__(self, cfg, device='cuda:0'):
        validate_config(cfg)
        super().__init__(cfg, device)


class TMEGuidedMUSE(nn.Module):
    """Freeze MUSE; condition class semantics using all 16 TME groups.

    The large description bank and ground-truth semantic retrieval are used
    only to train the native baseline. This wrapper always uses MUSE's
    label-free inference branch, including during conditioner training.
    """
    def __init__(self, base, cohort, tme_mode='actual', **kwargs):
        super().__init__()
        if tme_mode not in ('zero', 'actual'):
            raise ValueError('unsupported TME input mode')
        self.base = base.eval().requires_grad_(False)
        self.standardizer = FoldStandardizer(62 if cohort == 'nsclc' else 64)
        self.conditioner = TMEQueryConditioner(base.class_text_features.shape[-1], cohort, **kwargs)
        self.tme_mode = tme_mode

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def adapter_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('base.')}

    def load_adapter_state_dict(self, state):
        if set(state) != set(self.adapter_state_dict()):
            raise ValueError('adapter checkpoint keys differ')
        self.load_state_dict({**self.state_dict(), **state}, strict=True)

    def forward(self, features, label, tme, *, return_details=False):
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if (features.ndim != 2 or features.shape[0] == 0
                or features.shape[1] != self.base.input_dim
                or tme.shape != (1, self.standardizer.feature_count)):
            raise ValueError('one correctly sized feature bag and one TME row required')
        values = self.standardizer(tme)
        if self.tme_mode == 'zero':
            values = torch.zeros_like(values)
        semantics, group_attention = self.conditioner(self.base._adapt_class_semantics(), values)
        patches = self.base.visual_adapter(features.float())
        aggregate = self.base._aggregate(patches, semantics, filter_top_patches=True,
                                         return_attention=return_details)
        if return_details:
            aggregate, patch_attention = aggregate
        logits = self.base.classifier(aggregate)
        # Labels affect only the loss, never semantic retrieval or prediction.
        loss = F.cross_entropy(logits, label)
        probabilities = logits.softmax(1)
        prediction = probabilities.topk(1, dim=1)[1]
        if not return_details:
            return probabilities, prediction, loss
        # The frozen MoE is in eval mode, so these are its actual noiseless
        # route weights. Expert IDs/top-k and patch indices are discrete.
        route_logits = self.base.moe.router(semantics)
        top_values, top_indices = route_logits.topk(self.base.moe.top_k, dim=-1)
        expert_weights = torch.zeros_like(route_logits).scatter(
            1, top_indices, top_values.softmax(-1))
        return dict(logits=logits, probabilities=probabilities, prediction=prediction, loss=loss,
                    tme_group_attention=group_attention, patch_attention=patch_attention,
                    expert_weights=expert_weights)


class MUSETMEMethod(MUSENativeMethod):
    extension_name = PORT

    def __init__(self, cfg, maps, arm, device='cuda:0'):
        if arm not in ('zero', 'actual', 'shuffled'):
            raise ValueError('unknown adapter arm')
        if cfg.get('tme_mode') != ('zero' if arm == 'zero' else 'actual'):
            raise ValueError('TME mode does not match arm')
        super().__init__(cfg, device)
        self.tme = load_tme(cfg['tme_feature_csv'], cfg['task'])
        if arm == 'shuffled':
            original = self.tme.copy(deep=True)
            for mapping in maps.values():
                self.tme.loc[list(mapping)] = original.loc[list(mapping.values())].to_numpy(copy=True)

    def build_model(self):
        base = super().build_model()
        checkpoint = Path(self.cfg['base_checkpoint_dir']) / f"fold{self.cfg['_fold_index']}_best.pt"
        base.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True), strict=True)
        # Encoder/cache construction can consume different RNG draws on its
        # first call. Reset immediately before initializing the matched arms.
        from train import set_seed
        set_seed(self.cfg['seed'] + self.cfg['_fold_index'])
        return TMEGuidedMUSE(base, self.cfg['task'], tme_mode=self.cfg['tme_mode'],
            hidden_dim=self.cfg['tme_hidden_dim'], num_heads=self.cfg['tme_attention_heads'],
            dropout=self.cfg['tme_dropout'], initial_gate=self.cfg['tme_initial_gate']).to(self.device)

    def prepare_fold(self, fold, model, train_loader):
        if fold != self.cfg['_fold_index']:
            raise ValueError('fold mismatch')
        ids = train_loader.dataset.df.slide_id.astype(str).tolist()
        model.standardizer.fit(self.tme.loc[ids].to_numpy(dtype=np.float32))

    def inputs(self, batch, model):
        if len(batch) != 3 or not isinstance(batch[-2], dict):
            raise ValueError('MUSE requires [single bag, metadata, label]')
        metadata = {k: str(v if isinstance(v, str) else v[0]) for k, v in batch[-2].items()}
        tme = torch.as_tensor(self.tme.loc[metadata['slide_id']].to_numpy(dtype=np.float32),
                              device=self.device).unsqueeze(0)
        return (batch[0].to(self.device), batch[-1].to(self.device), tme), metadata

    def train_step(self, batch, model, optimizer, loss_fn=None):
        inputs, _ = self.inputs(batch, model)
        optimizer.zero_grad(set_to_none=True)
        details = model(*inputs, return_details=True)
        details['loss'].backward()
        optimizer.step()
        return dict(loss=details['loss'].item(), logits=details['logits'].detach(), label=inputs[1])

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        inputs, _ = self.inputs(batch, model)
        details = model(*inputs, return_details=True)
        return dict(loss=details['loss'].item(), logits=details['logits'], label=inputs[1])

    @torch.no_grad()
    def eval_step_with_details(self, batch, model):
        inputs, metadata = self.inputs(batch, model)
        return {**model(*inputs, return_details=True), 'metadata': metadata}
