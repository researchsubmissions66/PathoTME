"""FOCUS text-query conditioning; original PGVL runtime is imported unchanged."""
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch import nn

from methods.base import BaseMethod, probabilities_to_logits
from methods.focus.adapter import FOCUSMethod
from common.models.paired_encoder_extension import project_paired_features
from pathotme.guided_vila import FoldStandardizer, SemanticTMETokenizer
from pathotme.brca_vila import BreastTMETokenizer
from pathotme.brca_features import panel_spec, transform_rows
from pathotme.features import FEATURE_NAMES, transform_nsclc_features, validate_feature_table
from pathotme.focus_contract import PORT, validate_config
from pathotme.locked_tcga import rows


class FocusNativeMethod(FOCUSMethod):
    """Existing PLIP port plus an explicit PathoTME-owned RN50 port."""
    backbone_contract = replace(FOCUSMethod.backbone_contract,
        supported_backbones=('plip', 'clip-rn50'), default_backbone='plip',
        feature_dims={'plip': (768,), 'clip-rn50': (1024,)},
        feature_boundaries={'plip': {'hf:vinid/plip#vision-preprojection': (768,)},
                            'clip-rn50': {'openai/clip-rn50@official': (1024,)}})

    def __init__(self, cfg, device='cuda:0'):
        validate_config(cfg)
        if cfg['backbone'] == 'plip':
            FOCUSMethod.__init__(self, cfg, device)
        else:
            BaseMethod.__init__(self, cfg, device)
            self.is_encoder_extension = True


class TMEQueryConditioner(nn.Module):
    """Attend each high-resolution class query to all 16 semantic TME groups.

    Tissue and cell/spatial groups are jointly available because FOCUS has one
    visual scale. The tokenizer's low/high names describe biological roles,
    not a second FOCUS feature bag.
    """
    def __init__(self, model_dim, cohort, hidden_dim=128, num_heads=4,
                 dropout=0.1, initial_gate=0.1):
        super().__init__()
        if cohort not in ('nsclc', 'brca') or hidden_dim % num_heads:
            raise ValueError('invalid cohort or attention dimensions')
        if not 0 < initial_gate < 1:
            raise ValueError('initial gate must be strictly between zero and one')
        cls = SemanticTMETokenizer if cohort == 'nsclc' else BreastTMETokenizer
        self.tokenizer = cls(hidden_dim, dropout=dropout)
        self.token_names = tuple('low_' + n for n in self.tokenizer.LOW_TOKEN_NAMES) + tuple(
            'high_' + n for n in self.tokenizer.HIGH_TOKEN_NAMES)
        self.query_projection = nn.Linear(model_dim, hidden_dim, bias=False)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.output_projection = nn.Linear(hidden_dim, model_dim, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(math.log(initial_gate / (1 - initial_gate))))

    def forward(self, queries, values):
        low, high = self.tokenizer(values)
        tokens = torch.cat((low, high), dim=0)
        update, weights = self.cross_attention(self.query_projection(queries).unsqueeze(1),
                                               tokens, tokens, average_attn_weights=True)
        return queries + self.gate_logit.sigmoid() * self.output_projection(update).squeeze(1), weights


class TMEGuidedFOCUS(nn.Module):
    """Freeze FOCUS and learn a gated residual on its encoded semantic queries."""
    def __init__(self, base, cohort, tme_mode='actual', **kwargs):
        super().__init__()
        if tme_mode not in ('zero', 'actual'):
            raise ValueError('unsupported TME input mode')
        self.base = base.eval().requires_grad_(False)
        self.standardizer = FoldStandardizer(62 if cohort == 'nsclc' else 64)
        self.conditioner = TMEQueryConditioner(base.D, cohort, **kwargs)
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
        if features.ndim != 2 or tme.shape != (1, self.standardizer.feature_count):
            raise ValueError('one feature bag and one TME row required')
        values = self.standardizer(tme)
        if self.tme_mode == 'zero':
            values = torch.zeros_like(values)
        prompts = self.base.prompt_learner()
        queries = self.base.text_encoder(prompts, self.base.prompt_learner.tokenized_prompts)[self.base.num_classes:]
        queries, group_attention = self.conditioner(queries, values)
        visual = self.base.feature_encoder(features.float())
        selected, indices = self.base.adaptive_token_selection(visual, queries)
        compressed, retained = self.base.spatial_token_compression(selected, queries, return_indices=True)
        # Discrete selection has no gradient. Final cross-attention supplies
        # the differentiable path to the conditioned query and its TME tokens.
        attended = self.base.cross_attention(queries.unsqueeze(0), compressed.unsqueeze(0),
                                             compressed.unsqueeze(0), return_weights=return_details)
        if return_details:
            attended, weights = attended
        logits = self.base.classifier(attended.mean(1))
        loss = self.base.loss_ce(logits, label)
        probabilities = logits.softmax(1)
        prediction = probabilities.topk(1, dim=1)[1]
        if not return_details:
            return probabilities, prediction, loss
        patch_attention = visual.new_zeros((len(queries), len(visual)))
        original = indices[retained]
        patch_attention[:, original] = weights.mean(1).squeeze(0)
        return dict(logits=logits, probabilities=probabilities, prediction=prediction, loss=loss,
                    tme_group_attention=group_attention, patch_attention=patch_attention,
                    selected_patch_indices=original)


def load_tme(path, cohort):
    table = pd.read_csv(path)
    if cohort == 'nsclc':
        result = transform_nsclc_features(validate_feature_table(table))
        if tuple(result.columns) != FEATURE_NAMES:
            raise ValueError('core62 order mismatch')
        result.index = table.slide_id.astype(str)
    else:
        names = panel_spec('brca_morph64_v1')['feature_names']
        if list(table.columns) != ['slide_id', *names]:
            raise ValueError('morph64 order mismatch')
        result = pd.DataFrame(transform_rows(rows(path), 'brca_morph64_v1'),
                              columns=names, index=table.slide_id.astype(str))
    if result.index.has_duplicates:
        raise ValueError('duplicate TME rows')
    return result


class FocusTMEMethod(FocusNativeMethod):
    extension_name = PORT

    def __init__(self, cfg, maps, arm, device='cuda:0'):
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
        return TMEGuidedFOCUS(base, self.cfg['task'], tme_mode=self.cfg['tme_mode'],
            hidden_dim=self.cfg['tme_hidden_dim'], num_heads=self.cfg['tme_attention_heads'],
            dropout=self.cfg['tme_dropout'], initial_gate=self.cfg['tme_initial_gate']).to(self.device)

    def prepare_fold(self, fold, model, train_loader):
        if fold != self.cfg['_fold_index']:
            raise ValueError('fold mismatch')
        ids = train_loader.dataset.df.slide_id.astype(str).tolist()
        model.standardizer.fit(self.tme.loc[ids].to_numpy(dtype=np.float32))

    def inputs(self, batch, model):
        if len(batch) != 3 or not isinstance(batch[-2], dict):
            raise ValueError('FOCUS requires [single bag, metadata, label]')
        features, label = self._features_and_label(batch)
        metadata = {k: str(v if isinstance(v, str) else v[0]) for k, v in batch[-2].items()}
        tme = torch.as_tensor(self.tme.loc[metadata['slide_id']].to_numpy(dtype=np.float32),
                              device=self.device).unsqueeze(0)
        return (project_paired_features(model.base, features.to(self.device)),
                label.to(self.device), tme), metadata

    def train_step(self, batch, model, optimizer, loss_fn=None):
        inputs, _ = self.inputs(batch, model)
        optimizer.zero_grad(set_to_none=True)
        probabilities, _, loss = model(*inputs)
        loss.backward()
        optimizer.step()
        return dict(loss=loss.item(), logits=probabilities_to_logits(probabilities).detach(), label=inputs[1])

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        inputs, _ = self.inputs(batch, model)
        probabilities, _, loss = model(*inputs)
        return dict(loss=loss.item(), logits=probabilities_to_logits(probabilities), label=inputs[1])

    @torch.no_grad()
    def eval_step_with_details(self, batch, model):
        inputs, metadata = self.inputs(batch, model)
        return {**model(*inputs, return_details=True), 'metadata': metadata}
