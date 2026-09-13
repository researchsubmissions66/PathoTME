"""Condition DyKo's class queries before both native attention streams."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from methods.base import BaseMethod
from methods.dyko.adapter import DyKoMethod
from common.models.paired_encoder_extension import project_paired_features
from pathotme.dyko_contract import PORT, validate_config
from pathotme.focus_tme import TMEQueryConditioner
from pathotme.guided_vila import FoldStandardizer
from pathotme.muse_data import load_tme


class DyKoNativeMethod(DyKoMethod):
    """Registered PLIP plus a separately owned RN50 port of the same model."""
    backbone_contract = replace(DyKoMethod.backbone_contract,
        supported_backbones=('plip', 'clip-rn50'), default_backbone='plip',
        feature_dims={'plip': (768,), 'clip-rn50': (1024,)},
        feature_boundaries={'plip': {'hf:vinid/plip#vision-preprojection': (768,)},
                            'clip-rn50': {'openai/clip-rn50@official': (1024,)}})

    def __init__(self, cfg, device='cuda:0'):
        validate_config(cfg)
        if cfg['backbone'] == 'plip' and cfg['task'] == 'nsclc':
            super().__init__(cfg, device)
        else:
            BaseMethod.__init__(self, cfg, device)
            self.is_encoder_extension = True
            self._train_batches, self._batch_index = 1, 0


    def build_model(self):
        if self.cfg['task'] == 'nsclc':
            return super().build_model()
        from methods.dyko.model import DyKoModel
        from methods.dyko.prompts import load_prompt_bank
        from common.models.paired_encoder_extension import FeatureSpacePromptLearner, build_paired_feature_projector
        from pathotme.locked_tcga import sha, verify_files
        cfg = self.cfg
        assets = cfg['pathotme_knowledge_bank']
        verify_files({assets[k]: assets[k.replace('_path', '_sha256')]
                      for k in ('class_prompt_path', 'tensor_path', 'source_bank_path', 'encoding_path')})
        bank = load_prompt_bank(cfg['text_prompt_path'], cfg['label_dict'],
                                cfg['prompt_file_classnames'], cfg['prompt_class_bindings'])
        if bank.file_sha256 != cfg['text_prompt_file_sha256']:
            raise ValueError('BRCA class prompt hash mismatch')
        concepts = torch.load(cfg['concept_feature_path'], map_location='cpu', weights_only=True)
        if not isinstance(concepts, torch.Tensor) or concepts.shape != (64, 768) or not torch.isfinite(concepts).all():
            raise ValueError('finite BRCA 64x768 TITAN tensor required')
        encoder = self.load_encoder(weights_path=cfg['backbone_weights']).freeze()
        projector, width = build_paired_feature_projector(encoder,
            feature_space_id=cfg['feature_space_id'], feature_dim=cfg['feature_dim'])
        with torch.no_grad():
            text = encoder.encode_text(bank.descriptions, normalize=True).detach()
        model = DyKoModel(FeatureSpacePromptLearner(text, views=1, class_specific=False),
            concepts, width=width, concept_input_dim=768, n_classes=2,
            **{k: cfg[k] for k in ('visual_prototypes', 'concepts_per_prototype', 'num_heads',
                                    'retrieval_temperature', 'kmeans_iterations', 'kmeans_seed')})
        model.paired_feature_projector = projector
        return model.to(self.device)


class TMEGuidedDyKo(nn.Module):
    """Freeze DyKo; a shared gated TME residual changes its two class queries."""
    def __init__(self, base, cohort, tme_mode='actual', **kwargs):
        super().__init__()
        if cohort not in ('nsclc', 'brca') or tme_mode not in ('zero', 'actual'):
            raise ValueError('only NSCLC/BRCA actual/zero TME inputs are configured')
        self.base = base.eval().requires_grad_(False)
        self.standardizer = FoldStandardizer(62 if cohort == 'nsclc' else 64)
        self.conditioner = TMEQueryConditioner(base.classifier.in_features, cohort, **kwargs)
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
            raise ValueError('one feature bag and one correctly sized TME row required')
        values = self.standardizer(tme)
        if self.tme_mode == 'zero':
            values = torch.zeros_like(values)
        calls = []
        def condition(_module, _inputs, queries):
            if queries.shape != (2, self.base.classifier.in_features):
                raise ValueError('expected exactly two class-ordered DyKo queries')
            conditioned, attention = self.conditioner(queries, values)
            calls.append(attention)
            return conditioned
        # The original forward computes WAKI before this boundary and uses the
        # same conditioned queries for visual and semantic cross-attention.
        handle = self.base.prompt_learner.register_forward_hook(condition)
        try:
            output = self.base(features)
        finally:
            handle.remove()
        if len(calls) != 1:
            raise ValueError('expected exactly one prompt learner call')
        logits = output['logits']
        loss = F.cross_entropy(logits, label)
        probabilities = logits.softmax(1)
        prediction = probabilities.topk(1, dim=1)[1]
        if not return_details:
            return probabilities, prediction, loss
        return {**output, 'probabilities': probabilities, 'prediction': prediction,
                'loss': loss, 'tme_group_attention': calls[0],
                'patch_attention': output['visual_patch_attention'][0]}


class DyKoTMEMethod(DyKoNativeMethod):
    extension_name = PORT

    def __init__(self, cfg, maps, arm, device='cuda:0'):
        super().__init__(cfg, device)
        if arm not in ('zero', 'actual', 'shuffled'):
            raise ValueError('unknown TME arm')
        self.tme = load_tme(cfg['tme_feature_csv'], cfg['task'])
        if arm == 'shuffled':
            original = self.tme.copy(deep=True)
            for mapping in maps.values():
                self.tme.loc[list(mapping)] = original.loc[list(mapping.values())].to_numpy(copy=True)

    def build_model(self):
        base = super().build_model()
        checkpoint = Path(self.cfg['base_checkpoint_dir']) / f"fold{self.cfg['_fold_index']}_best.pt"
        base.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True), strict=True)
        # Native initialization/loading may consume different random draws.
        # Start all three conditioners from exactly the same post-base seed.
        from train import set_seed
        set_seed(self.cfg['seed'] + self.cfg['_fold_index'])
        return TMEGuidedDyKo(base, self.cfg['task'], tme_mode=self.cfg['tme_mode'],
            hidden_dim=self.cfg['tme_hidden_dim'], num_heads=self.cfg['tme_attention_heads'],
            dropout=self.cfg['tme_dropout'], initial_gate=self.cfg['tme_initial_gate']).to(self.device)

    def prepare_fold(self, fold, model, train_loader):
        if fold != self.cfg['_fold_index']:
            raise ValueError('fold mismatch')
        ids = train_loader.dataset.df.slide_id.astype(str).tolist()
        model.standardizer.fit(self.tme.loc[ids].to_numpy(dtype=np.float32))

    def inputs(self, batch, model):
        if len(batch) != 3 or not isinstance(batch[-2], dict):
            raise ValueError('DyKo requires [single bag, metadata, label]')
        features, label = self._unpack(batch, self.device)
        metadata = {k: str(v if isinstance(v, str) else v[0]) for k, v in batch[-2].items()}
        tme = torch.as_tensor(self.tme.loc[metadata['slide_id']].to_numpy(dtype=np.float32),
                              device=self.device).unsqueeze(0)
        return (project_paired_features(model.base, features).squeeze(0), label, tme), metadata

    def train_step(self, batch, model, optimizer, loss_fn=None):
        inputs, _ = self.inputs(batch, model)
        optimizer.zero_grad(set_to_none=True)
        result = model(*inputs, return_details=True)
        result['loss'].backward()
        optimizer.step()
        return dict(loss=result['loss'].item(), logits=result['logits'].detach(), label=inputs[1])

    @torch.no_grad()
    def eval_step(self, batch, model, loss_fn=None):
        inputs, _ = self.inputs(batch, model)
        result = model(*inputs, return_details=True)
        return dict(loss=result['loss'].item(), logits=result['logits'], label=inputs[1])

    @torch.no_grad()
    def eval_step_with_details(self, batch, model):
        inputs, metadata = self.inputs(batch, model)
        return {**model(*inputs, return_details=True), 'metadata': metadata}
